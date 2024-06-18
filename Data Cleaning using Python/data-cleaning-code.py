import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statistics
import numpy as np
from sklearn.decomposition import PCA

# create a dataframe for the data
medical_raw_data = pd.read_csv("medical_raw_data.csv")

# inspect data
print("*** Structure of the dataset ***")
print(medical_raw_data.info())

# detect any duplicate values
print("*** Detect duplicates ***")
print(medical_raw_data.duplicated().value_counts())

# detect any missing values
print("*** Detect missing values ***")
print(medical_raw_data.isnull().sum())

# detect inconsistent presentation of data in categorical variables
print("*** Detect unique values in ReAdmis variable ***")
print(medical_raw_data.ReAdmis.unique())

print("*** Detect unique values in Soft_drink variable ***")
print(medical_raw_data.Soft_drink.unique())

print("*** Detect unique values in Initial_admin variable ***")
print(medical_raw_data.Initial_admin.unique())

print("*** Detect unique values in HighBlood variable ***")
print(medical_raw_data.HighBlood.unique())

print("*** Detect unique values in Stroke variable ***")
print(medical_raw_data.Stroke.unique())

print("*** Detect unique values in Complication_risk variable ***")
print(medical_raw_data.Complication_risk.unique())

print("*** Detect unique values in Overweight variable ***")
print(medical_raw_data.Overweight.unique())

print("*** Detect unique values in Arthritis variable ***")
print(medical_raw_data.Arthritis.unique())

print("*** Detect unique values in Diabetes variable ***")
print(medical_raw_data.Diabetes.unique())

print("*** Detect unique values in Hyperlipidemia variable ***")
print(medical_raw_data.Hyperlipidemia.unique())

print("*** Detect unique values in BackPain variable ***")
print(medical_raw_data.BackPain.unique())

print("*** Detect unique values in Anxiety variable ***")
print(medical_raw_data.Anxiety.unique())

print("*** Detect unique values in Allergic_rhinitis variable ***")
print(medical_raw_data.Allergic_rhinitis.unique())

print("*** Detect unique values in Reflux_esophagitis variable ***")
print(medical_raw_data.Reflux_esophagitis.unique())

print("*** Detect unique values in Asthma variable ***")
print(medical_raw_data.Asthma.unique())

print("*** Detect unique values in Services variable ***")
print(medical_raw_data.Services.unique())

print("*** Detect unique values in Item1 variable ***")
print(medical_raw_data.Item1.unique())

print("*** Detect unique values in Item2 variable ***")
print(medical_raw_data.Item2.unique())

print("*** Detect unique values in Item3 variable ***")
print(medical_raw_data.Item3.unique())

print("*** Detect unique values in Item4 variable ***")
print(medical_raw_data.Item4.unique())

print("*** Detect unique values in Item5 variable ***")
print(medical_raw_data.Item5.unique())

print("*** Detect unique values in Item6 variable ***")
print(medical_raw_data.Item6.unique())

print("*** Detect unique values in Item7 variable ***")
print(medical_raw_data.Item7.unique())

print("*** Detect unique values in Item8 variable ***")
print(medical_raw_data.Item8.unique())

# Treatment of missing values on quantitative variables
# Children variable
# create histogram of variable before imputation
plt.hist(medical_raw_data['Children'])
plt.title('Children_before')
plt.show()
# impute data
medical_raw_data['Children'].fillna(medical_raw_data['Children'].median(), inplace=True)
# verify missing values were imputed
print(medical_raw_data.isnull().sum())
# create histogram of variable after imputation
plt.hist(medical_raw_data['Children'])
plt.title('Children_after')
plt.show()

# Age variable
# create histogram of variable before imputation
plt.hist(medical_raw_data['Age'])
plt.title('Age_before')
plt.show()
# check mean and median values before imputation
print(medical_raw_data.describe())
print(medical_raw_data[['Age']].median())
# impute data
medical_raw_data['Age'].fillna(medical_raw_data['Age'].mean(), inplace=True)
# convert Age values to int format
medical_raw_data = medical_raw_data.astype({'Age':'int'})
# verify missing values were imputed
print(medical_raw_data.isnull().sum())
# display values of Age column after
print()
print('Age column values after')
print(medical_raw_data['Age'].to_string(index=False))
# create histogram of variable after imputation
plt.hist(medical_raw_data['Age'])
plt.title('Age_after')
plt.show()
# check mean and median values after imputation
print(medical_raw_data.describe())
print(medical_raw_data[['Age']].median())

# Income variable
# create histogram of variable before imputation
plt.hist(medical_raw_data['Income'])
plt.title('Income_before')
plt.show()
# impute data
medical_raw_data['Income'].fillna(medical_raw_data['Income'].median(), inplace=True)
# verify missing values were imputed
print(medical_raw_data.isnull().sum())
# create histogram of variable after imputation
plt.hist(medical_raw_data['Income'])
plt.title('Income_after')
plt.show()

# Initial_days variable
# create histogram of variable before imputation
plt.hist(medical_raw_data['Initial_days'])
plt.title('Initial_days_before')
plt.show()
# impute data
medical_raw_data['Initial_days'].fillna(medical_raw_data['Initial_days'].median(), inplace=True)
# verify missing values were imputed
print(medical_raw_data.isnull().sum())
# create histogram of variable after imputation
plt.hist(medical_raw_data['Initial_days'])
plt.title('Initial_days_after')
plt.show()


# Treatment of missing values on qualitative variables
# Soft_drink variable
# re-express the variable
# duplicate the column
medical_raw_data['Soft_drink_numeric'] = medical_raw_data['Soft_drink']
# create a dictionary
dict_softdrink = {"Soft_drink_numeric": {"No": 0, "Yes": 1}}
# replace values in the new column with dictionary
medical_raw_data.replace(dict_softdrink, inplace=True)
# create histogram of variable before imputation
plt.hist(medical_raw_data['Soft_drink_numeric'])
plt.title('Soft_drink_numeric_before')
plt.show()
# impute data
medical_raw_data["Soft_drink_numeric"] = medical_raw_data ['Soft_drink_numeric'].fillna(medical_raw_data['Soft_drink_numeric'].mode()[0])
# verify missing values were imputed
print(medical_raw_data.isnull().sum())
# create histogram of variable after imputation
plt.hist(medical_raw_data['Soft_drink_numeric'])
plt.title('Soft_drink_numeric_after')
plt.show()

# Overweight variable
# re-express the variable
# duplicate the column
medical_raw_data['Overweight_numeric'] = medical_raw_data['Overweight']
# create a dictionary
dict_overweight = {"Overweight_numeric": {"No": 0, "Yes": 1}}
# replace values in the new column with dictionary
medical_raw_data.replace(dict_overweight, inplace=True)
# create histogram of variable before imputation
plt.hist(medical_raw_data['Overweight_numeric'])
plt.title('Overweight_numeric_before')
plt.show()
# impute data
medical_raw_data["Overweight_numeric"] = medical_raw_data ['Overweight_numeric'].fillna(medical_raw_data['Overweight_numeric'].mode()[0])
# verify missing values were imputed
print(medical_raw_data.isnull().sum())
# create histogram of variable after imputation
plt.hist(medical_raw_data['Overweight_numeric'])
plt.title('Overweight_numeric_after')
plt.show()

# Anxiety variable
# re-express the variable
# duplicate the column
medical_raw_data['Anxiety_numeric'] = medical_raw_data['Anxiety']
# create a dictionary
dict_anxiety = {"Anxiety_numeric": {"No": 0, "Yes": 1}}
# replace values in the new column with dictionary
medical_raw_data.replace(dict_anxiety, inplace=True)
# create histogram of variable before imputation
plt.hist(medical_raw_data['Anxiety_numeric'])
plt.title('Anxiety_numeric_before')
plt.show()
# impute data
medical_raw_data["Anxiety_numeric"] = medical_raw_data ['Anxiety_numeric'].fillna(medical_raw_data['Anxiety_numeric'].mode()[0])
# verify missing values were imputed
print(medical_raw_data.isnull().sum())
# create histogram of variable after imputation
plt.hist(medical_raw_data['Anxiety_numeric'])
plt.title('Anxiety_numeric_after')
plt.show()


# Detect outliers
# Population variable
# calculate z-score
medical_raw_data['population_z'] = stats.zscore(medical_raw_data['Population'])
# detect outliers
population_outliers = medical_raw_data.query('population_z > 3 | population_z < -3')
# display outliers
row_count = len(population_outliers)
print(f'The Population variable has {row_count} outliers.')
print('Values range from', population_outliers['Population'].min(), 'to', population_outliers['Population'].max())
print('Z-scores range from', population_outliers['population_z'].min(), 'to', population_outliers['population_z'].max())
print()

# Children variable
# calculate z-score
medical_raw_data['children_z'] = stats.zscore(medical_raw_data['Children'])
# find outliers
children_outliers = medical_raw_data.query('children_z > 3 | children_z < -3')
# display outliers
row_count = len(children_outliers)
print(f'The Children variable has {row_count} outliers.')
print('Values range from', children_outliers['Children'].min(), 'to', children_outliers['Children'].max())
print('Z-scores range from', children_outliers['children_z'].min(), 'to', children_outliers['children_z'].max())
print()

# Age variable
# calculate z-score
medical_raw_data['age_z'] = stats.zscore(medical_raw_data['Age'])
# find outliers
age_outliers = medical_raw_data.query('age_z > 3 | age_z < -3')
# display outliers
row_count = len(age_outliers)
print(f'The Age variable has {row_count} outliers.')
print()

# Income variable
# calculate z-score
medical_raw_data['income_z'] = stats.zscore(medical_raw_data['Income'])
# find outliers
income_outliers = medical_raw_data.query('income_z > 3 | income_z < -3')
# display outliers
row_count = len(income_outliers)
print(f'The Income variable has {row_count} outliers.')
print('Values range from', income_outliers['Income'].min(), 'to', income_outliers['Income'].max())
print('Z-scores range from', income_outliers['income_z'].min(), 'to', income_outliers['income_z'].max())
print()

# VitD_levels variable
# calculate z-score
medical_raw_data['vitdlevels_z'] = stats.zscore(medical_raw_data['VitD_levels'])
# find outliers
vitdlevels_outliers = medical_raw_data.query('vitdlevels_z > 3 | vitdlevels_z < -3')
# display outliers
row_count = len(vitdlevels_outliers)
print(f'The VitD_levels variable has {row_count} outliers.')
print('Values range from', vitdlevels_outliers['VitD_levels'].min(), 'to', vitdlevels_outliers['VitD_levels'].max())
print('Z-scores range from', vitdlevels_outliers['vitdlevels_z'].min(), 'to', vitdlevels_outliers['vitdlevels_z'].max())
print()

# Doc_visits variable
# calculate z-score
medical_raw_data['docvisits_z'] = stats.zscore(medical_raw_data['Doc_visits'])
# find outliers
docvisits_outliers = medical_raw_data.query('docvisits_z > 3 | docvisits_z < -3')
# display outliers
row_count = len(docvisits_outliers)
print(f'The Doc_visits variable has {row_count} outliers.')
print('Values range from', docvisits_outliers['Doc_visits'].min(), 'to', docvisits_outliers['Doc_visits'].max())
print('Z-scores range from', docvisits_outliers['docvisits_z'].min(), 'to', docvisits_outliers['docvisits_z'].max())
print()

# Full_meals_eaten variable
# calculate z-score
medical_raw_data['fullmealseaten_z'] = stats.zscore(medical_raw_data['Full_meals_eaten'])
# find outliers
fullmealseaten_outliers = medical_raw_data.query('fullmealseaten_z > 3 | fullmealseaten_z < -3')
# display outliers
row_count = len(fullmealseaten_outliers)
print(f'The Full_meals_eaten variable has {row_count} outliers.')
print('Values range from', fullmealseaten_outliers['Full_meals_eaten'].min(), 'to', fullmealseaten_outliers['Full_meals_eaten'].max())
print('Z-scores range from', fullmealseaten_outliers['fullmealseaten_z'].min(), 'to', fullmealseaten_outliers['fullmealseaten_z'].max())
print()

# VitD_supp variable
# calculate z-score
medical_raw_data['vitdsupp_z'] = stats.zscore(medical_raw_data['VitD_supp'])
# find outliers
vitdsupp_outliers = medical_raw_data.query('vitdsupp_z > 3 | vitdsupp_z < -3')
# display outliers
row_count = len(vitdsupp_outliers)
print(f'The VitD_supp variable has {row_count} outliers.')
print('Values range from', vitdsupp_outliers['VitD_supp'].min(), 'to', vitdsupp_outliers['VitD_supp'].max())
print('Z-scores range from', vitdsupp_outliers['vitdsupp_z'].min(), 'to', vitdsupp_outliers['vitdsupp_z'].max())
print()

# Initial_days variable
# calculate z-score
medical_raw_data['initialdays_z'] = stats.zscore(medical_raw_data['Initial_days'])
# find outliers
initialdays_outliers = medical_raw_data.query('initialdays_z > 3 | initialdays_z < -3')
# display outliers
row_count = len(initialdays_outliers)
print(f'The Initial_days variable has {row_count} outliers.')
print()

# TotalCharge variable
# calculate z-score
medical_raw_data['totalcharge_z'] = stats.zscore(medical_raw_data['TotalCharge'])
# find outliers
totalcharge_outliers = medical_raw_data.query('totalcharge_z > 3 | totalcharge_z < -3')
# display outliers
row_count = len(totalcharge_outliers)
print(f'The TotalCharge variable has {row_count} outliers.')
print('Values range from', totalcharge_outliers['TotalCharge'].min(), 'to', totalcharge_outliers['TotalCharge'].max())
print('Z-scores range from', totalcharge_outliers['totalcharge_z'].min(), 'to', totalcharge_outliers['totalcharge_z'].max())
print()

# Additional_charges variable
# calculate z-score
medical_raw_data['additionalcharges_z'] = stats.zscore(medical_raw_data['Additional_charges'])
# find outliers
additionalcharges_outliers = medical_raw_data.query('additionalcharges_z > 3 | additionalcharges_z < -3')
# display outliers
row_count = len(additionalcharges_outliers)
print(f'The Additional_charges variable has {row_count} outliers.')

# round variables with monetary values to two decimal places
medical_raw_data.TotalCharge = medical_raw_data.TotalCharge.round(2)
medical_raw_data.Additional_charges = medical_raw_data.Additional_charges.round(2)

# save cleaned data to CSV format
medical_raw_data.to_csv('medical_raw_data_cleaned.csv', index=False)


# Principal Component Analysis
# create new dataframe of quantitative continuous variables
medical_vars_numerical = medical_raw_data[['Income', 'VitD_levels', 'Initial_days', 'TotalCharge', 'Additional_charges']]
# normalize quantitative variables
medical_vars_numerical_normalized = (medical_vars_numerical-medical_vars_numerical.mean())/medical_vars_numerical.std()
# apply PCA
pca = PCA(n_components=medical_vars_numerical.shape[1])
print(pca)
# fit PCA onto normalized data
pca.fit(medical_vars_numerical_normalized)
# transform fit into a dataframe
medical_raw_data_pca = pd.DataFrame(pca.transform(medical_vars_numerical_normalized))
# assign column names to new dataframe
columns=['PC1','PC2','PC3','PC4','PC5']
# PCA loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC1','PC2','PC3','PC4','PC5'], index=medical_vars_numerical.columns)
print(loadings)
# save PCA loading matrix to CSV format
loadings.to_csv('medical_raw_data_PCAmatrix.csv', index=False)
# Select principal components
# calculate covariance
cov_matrix = np.dot(medical_vars_numerical_normalized.T, medical_vars_numerical_normalized)/ medical_vars_numerical.shape[0]
# calculate eigenvalues
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]
# display calculated eigenvalues
print()
print("Eigenvalues")
print(eigenvalues)
# create scree plot of eigenvalues
plt.plot(eigenvalues)
plt.xlabel('number of components')
plt.ylabel('eigenvalue')
plt.axhline(y=1, color="red")
plt.show()