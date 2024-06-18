import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import roc_auc_score

# create a dataframe for the data
churn_data = pd.read_csv("churn_clean.csv")

# inspect data
print("*** Structure of the dataset ***")
print(churn_data.info())
print()

# detect any duplicate values
print("*** Detect duplicates ***")
print(churn_data.duplicated().value_counts())
print()

# detect any missing values
print("*** Detect missing values ***")
print(churn_data.isnull().sum())
print()

# perform univariate imputation on InternetService variable
churn_data['InternetService'] = churn_data['InternetService'].fillna(churn_data['InternetService'].mode()[0])
# verify values were imputed
print("*** Confirm missing values were imputed ***")
print(churn_data.isnull().sum())
print()

# detect outliers
# - function to calculate z-score and display count of outliers
def find_outliers(df, column_name):
    # Calculate the z-scores for the specified column
    z_scores = np.abs(stats.zscore(df[column_name]))

    # Count how many z-scores are greater than 3 or less than -3
    extreme_count = np.sum((z_scores > 3) | (z_scores < -3))

    print(f"{column_name} has {extreme_count} outliers")

print("*** Find outliers ***")
find_outliers(churn_data, 'Outage_sec_perweek')
find_outliers(churn_data, 'Email')
find_outliers(churn_data, 'Contacts')
find_outliers(churn_data, 'Yearly_equip_failure')
find_outliers(churn_data, 'Tenure')
find_outliers(churn_data, 'MonthlyCharge')
find_outliers(churn_data, 'Bandwidth_GB_Year')
print()

# detect unique values in categorical variables
# - function to detect and output unique values
def print_unique_values_in_columns(dataframe, categorical_columns):
    for column_name in categorical_columns:
        if column_name in dataframe.columns:
            unique_values = dataframe[column_name].unique()
            print(f"Variable {column_name} has unique values: {', '.join(unique_values)}")
        else:
            print(f"Variable {column_name} does not exist in the DataFrame.")

categorical_columns = ['Churn', 'Techie', 'Contract', 'Port_modem', 'Tablet', 'InternetService', 'Phone', 'Multiple', 'OnlineSecurity', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']

print('*** Detect unique values in categorical variables ***')
print_unique_values_in_columns(churn_data, categorical_columns)
print()

# Re-express categorical variables
# - function to label encode categoricals with two values
def factorize_categoricals(dataframe, columns):
    for column in columns:
        if column in dataframe.columns:
            factorized_column, _ = pd.factorize(dataframe[column])
            dataframe[column + '_Numerical'] = factorized_column

columns_to_factorize = ['Churn', 'Techie', 'Port_modem', 'Tablet', 'InternetService', 'Phone', 'Multiple', 'OnlineSecurity', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
factorize_categoricals(churn_data, columns_to_factorize)

# One-hot encode categoricals with more than two values
encoded_data = pd.get_dummies(churn_data, columns=['Contract', 'PaymentMethod'], prefix=['Contract', 'PaymentMethod'], drop_first=False)

# View the resulting DataFrame to detect datatypes
print(encoded_data)

# Convert one-hot encoded values from Boolean to numeric
encoded_data['Contract_Month-to-month'] = encoded_data['Contract_Month-to-month'].astype(int)
encoded_data['Contract_One year'] = encoded_data['Contract_One year'].astype(int)
encoded_data['Contract_Two Year'] = encoded_data['Contract_Two Year'].astype(int)
encoded_data['PaymentMethod_Bank Transfer(automatic)'] = encoded_data['PaymentMethod_Bank Transfer(automatic)'].astype(int)
encoded_data['PaymentMethod_Credit Card (automatic)'] = encoded_data['PaymentMethod_Credit Card (automatic)'].astype(int)
encoded_data['PaymentMethod_Electronic Check'] = encoded_data['PaymentMethod_Electronic Check'].astype(int)
encoded_data['PaymentMethod_Mailed Check'] = encoded_data['PaymentMethod_Mailed Check'].astype(int)
# View the updated DataFrame to
print(encoded_data)

# Save cleaned dataset to CSV
encoded_data.to_csv('encoded_data_prepared.csv')


#KNN
# Split data into training and test sets 
# (Larose & Larose, 2019, p. 5.2.1)
X_train, X_test, Y_train, Y_test = train_test_split(encoded_data[['Outage_sec_perweek', 'Contacts', 'Yearly_equip_failure', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two Year', 'Tenure', 'MonthlyCharge']], encoded_data['Churn'], random_state=0)
print(X_test)
print()

# Save sets to CSV
X_train.to_csv('PartD1_encoded_Xtrain.csv')
X_test.to_csv('PartD1_encoded_Xtest.csv')
Y_train.to_csv('PartD1_encoded_Ytrain.csv')
Y_test.to_csv('PartD1_encoded_Ytest.csv')

# apply scaling to the datasets
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#setup the KNN model
knn = KNeighborsClassifier(n_neighbors = 5) 
#fitting the KNN
knn.fit(X_train_scaled, Y_train)

# Predict on dataset which model has not seen before 
print('Expected values for Churn')
print(knn.predict(X_test_scaled))
print()

#Checking accuracy on the training set
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, Y_train)))
#Checking accuracy on the test set
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, Y_test)))

# Area under the curve
y_pred_prob = knn.predict_proba(X_test_scaled)[::,1]
auc = metrics.roc_auc_score(Y_test, y_pred_prob)
print('The Area Under The Curve is', auc)