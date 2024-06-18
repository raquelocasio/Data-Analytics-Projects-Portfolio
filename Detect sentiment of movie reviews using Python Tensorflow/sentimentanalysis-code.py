import pandas as pd
import numpy as np
import re
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


#import file into a dataframe
moviedata = pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None, names=['sentence','score'])
print("*** View the dataset ***")
print(moviedata)
print()

# B1 - EDA
# inspect data
print("*** Structure of the dataset ***")
print(moviedata.info())
print()

# detect any duplicate values
print("*** Detect duplicate values ***")
print(moviedata.duplicated().value_counts())
print()

# remove duplicate values
moviedata.drop_duplicates(inplace=True)

# confirm duplicates have been removed
print("*** Confirm duplicates have been removed ***")
print(moviedata.duplicated().value_counts())
print()

# detect any missing values
print("*** Detect missing values ***")
print(moviedata.isnull().sum())
print()

# B1a - detect and remove unusual characters
def process_unusual_chars(sen):
    #remove HTML tags
    sentence = remove_tags(sen)

    #remove punctuation and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    #remove single characters
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    #remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    #convert all characters to lowercase
    sentence = sentence.lower()

    return sentence

tag_re = re.compile(r'<[^>]+>')

def remove_tags(text):
    return tag_re.sub('', text)

moviereviews = []
sentences = list(moviedata['sentence'])
for sen in sentences:
    moviereviews.append(process_unusual_chars(sen))

# display a movie review that has been processed
print("*** Sample movie review without unusual characters ***")
print(moviereviews[1])
print()

# add processed text to original dataframe
moviedata['reviews_processed'] = moviereviews
print("*** View dataset with processed text column ***")
print(moviedata.head())
print()

# split data intro train, test, and validation sets of 70/15/15
# split data into train and temp sets
movie_train, temp_df = train_test_split(moviedata, test_size=0.30, random_state=42)

# split temp set into test and validation sets of equal size
movie_test, movie_val = train_test_split(temp_df, test_size=0.5, random_state=42)

# Printing the sizes of the splits
print("Train set size:", len(movie_train))
print("Test set size:", len(movie_test))
print("Validation set size:", len(movie_val))
print()

#B1b - vocabulary size
tokenizer = Tokenizer()
tokenizer.fit_on_texts(movie_train['sentence'])
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size: ", vocab_size)
print()

# B1c - word embedding length
max_sequence_embedding = int(round(np.sqrt(np.sqrt(vocab_size)), 0))
print("Proposed word embedding length: ", max_sequence_embedding)
print()

#B1d - maximum sequence length
review_length = []
for char_len in movie_train['sentence']:
    review_length.append(len(char_len.split(" ")))

max_sequence_length = int(round(np.mean(review_length), 0))
print("Chosen maximum sequence length: ", max_sequence_length)
print()


# B2 - tokenization
# Tokenize training set
movie_train['tokenized_text'] = tokenizer.texts_to_sequences(movie_train['reviews_processed'])
# inspect data with new column
print("*** View training dataset with tokens column ***")
print(movie_train.head())
print()

# Tokenize test set
movie_test['tokenized_text'] = tokenizer.texts_to_sequences(movie_test['reviews_processed'])
# inspect data with new column
print("*** View test dataset with tokens column ***")
print(movie_test.head())
print()

# Tokenize validation set
movie_val['tokenized_text'] = tokenizer.texts_to_sequences(movie_val['reviews_processed'])
# inspect data with new column
print("*** View validation dataset with tokens column ***")
print(movie_val.head())
print()

# # B3 - padding
# Pad the training set
padded_train = tf.keras.preprocessing.sequence.pad_sequences(movie_train['tokenized_text'], maxlen=max_sequence_length, padding='post')

# Add column with the padded sequences
movie_train['padded_tokenized_text'] = padded_train.tolist()

# inspect training data with new padded sequence column
print("*** View training dataset with padded sequence column ***")
print(movie_train.head())
print()

# Pad the test set
padded_test = tf.keras.preprocessing.sequence.pad_sequences(movie_test['tokenized_text'], maxlen=max_sequence_length, padding='post')

# Add column with the padded sequences
movie_test['padded_tokenized_text'] = padded_test.tolist()

# inspect test data with new padded sequence column
print("*** View test dataset with padded sequence column ***")
print(movie_test.head())
print()

# Pad the test set
padded_val = tf.keras.preprocessing.sequence.pad_sequences(movie_val['tokenized_text'], maxlen=max_sequence_length, padding='post')

# Add column with the padded sequences
movie_val['padded_tokenized_text'] = padded_val.tolist()

# inspect validation data with new padded sequence column
print("*** View validation dataset with padded sequence column ***")
print(movie_val.head())
print()

# display a single padded sequence
print("*** View single padded sequence ***")
print(movie_train['padded_tokenized_text'].iloc[1])
print()


# B6
# save prepared datasets to CSV format
movie_train.to_csv('movie_train_prepared.csv')
movie_test.to_csv('movie_test_prepared.csv')
movie_val.to_csv('movie_val_prepared.csv')


# C1 network architecture
activation = 'softmax'
loss = 'categorical_crossentropy'
optimizer = 'adam'
num_epochs = 20

# Define early stopping monitor
early_stopping_monitor = EarlyStopping(patience=2)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, output_dim=max_sequence_embedding),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(2, activation=activation)
])

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
print(model.summary())