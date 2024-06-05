import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statistics
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# create a dataframe for the data
medical = pd.read_csv("medical_clean.csv")

# inspect data
print("*** Structure of the dataset ***")
print(medical.info())

# detect any duplicate values
print()
print("*** Detect duplicates ***")
print(medical.duplicated().value_counts())

# detect any missing values
print()
print("*** Detect missing values ***")
print(medical.isnull().sum())

print()

# Detect outliers
# function to detect and display outliers
def count_and_print_extreme_z_scores(df, column_name):
    # Calculate the z-scores for the specified column
    z_scores = np.abs(stats.zscore(df[column_name]))

    # Count how many z-scores are outliers
    extreme_count = np.sum((z_scores > 3) | (z_scores < -3))

    print(f"Variable {column_name} has {extreme_count} outliers.")

count_and_print_extreme_z_scores(medical, 'Population')
count_and_print_extreme_z_scores(medical, 'Children')
count_and_print_extreme_z_scores(medical, 'Age')
count_and_print_extreme_z_scores(medical, 'Income')
count_and_print_extreme_z_scores(medical, 'VitD_levels')
count_and_print_extreme_z_scores(medical, 'Doc_visits')
count_and_print_extreme_z_scores(medical, 'Full_meals_eaten')
count_and_print_extreme_z_scores(medical, 'vitD_supp')
count_and_print_extreme_z_scores(medical, 'Initial_days')
count_and_print_extreme_z_scores(medical, 'TotalCharge')
count_and_print_extreme_z_scores(medical, 'Additional_charges')
print()

# Summary statistics
# dependent variable
print(medical['Initial_days'].describe())
# independent variables
print(medical[['Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'vitD_supp']].describe())

# Univariate visualizations
# function to create histogram
def uni_histograms(dataframe, column_names):
    for column_name in column_names:
        if column_name not in dataframe.columns:
            print(f"Warning: Column '{column_name}' not found in the DataFrame.")
            continue
        
        plt.figure(figsize=(8, 6))
        plt.hist(dataframe[column_name], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histogram for {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
# create histograms
columns_to_plot = ['Initial_days', 'Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'vitD_supp']
uni_histograms(medical, columns_to_plot)

# Bivariate visualizations
# function to create scatterplot
def bi_scatterplot(dataframe, x_col, y_col, title='', x_label='', y_label=''):    
    # Extract the X and Y columns
    x_values = dataframe[x_col]
    y_values = dataframe[y_col]
    # Create a scatterplot
    plt.scatter(x_values, y_values, label='Scatterplot', color='b', marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    # Show the legend
    plt.legend()
    # Display the plot
    plt.show()
# create scatterplots
bi_scatterplot(medical, 'Children', 'Initial_days', 'Initial_days vs Children', 'Children', 'Initial_days')
bi_scatterplot(medical, 'Age', 'Initial_days', 'Initial_days vs Age', 'Age', 'Initial_days')
bi_scatterplot(medical, 'Income', 'Initial_days', 'Initial_days vs Income', 'Income', 'Initial_days')
bi_scatterplot(medical, 'VitD_levels', 'Initial_days', 'Initial_days vs VitD_levels', 'VitD_levels', 'Initial_days')
bi_scatterplot(medical, 'Doc_visits', 'Initial_days', 'Initial_days vs Doc_visits', 'Doc_visits', 'Initial_days')
bi_scatterplot(medical, 'vitD_supp', 'Initial_days', 'Initial_days vs vitD_supp', 'vitD_supp', 'Initial_days')


# Multiple linear regression
# (Larose & Larose, 2019, sec. 11.4)
X = pd.DataFrame(medical[['Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'vitD_supp']])
y = pd.DataFrame(medical[['Initial_days']])
# add a constant term b to predictor variables
X = sm.add_constant(X)
# run the model
model_initial = sm.OLS(y, X).fit()
# get results of the model
print(model_initial.summary())
print()

# Feature selection
# (Stepwise Regression in Python: A Comprehensive Guide | Saturn Cloud Blog, 2023)
# Function to perform backward stepwise regression
def backward_regression(X, y, threshold_out):
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        p_values = model.pvalues.iloc[1:]
        worst_p_value = p_values.max()
        if worst_p_value > threshold_out:
            changed = True
            worst_feature = p_values.idxmax()
            included.remove(worst_feature)
        if not changed:
            break
    return included

# Set the significance level for removing variables
threshold = 0.05

# Perform backward stepwise regression
selected_features = backward_regression(X, y, threshold)

# Create the final regression model with selected features
final_model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

# Print the summary of the final model
print(final_model.summary())


# Residual plot
# (GeeksforGeeks, 2022)
# modify figure size
fig = plt.figure(figsize=(14, 8))

# create regression plots
fig = sm.graphics.plot_regress_exog(final_model, 'Children', fig=fig)

# Residual standard error
# (DSC Data Science Concepts, 2021)
print()
print('The residual standard error is', np.sqrt(final_model.mse_resid))