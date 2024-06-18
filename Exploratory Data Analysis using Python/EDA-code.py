import pandas as pd
from scipy.stats import chi2_contingency

# import csv to a dataframe
medical = pd.read_csv("medical_clean.csv")

# create contingency table
# (Zach, 2021)
observed = pd.crosstab(index=medical['Marital'], columns=medical['Initial_admin'])
print('Contingency Table')
print(observed)

# conduct Chi-Square Test
chi2, p, dof, expected = chi2_contingency(observed)

# interpret results
alpha = 0.05
print()
print("The p-value is", p)
if p < alpha:
    print("There is a significant association between the variables.")
else:
    print("There is no significant association between the variables.")

print()
print("Expected frequencies:\n", expected)

# identify unique values for categorical variables
# for use in univariate and bivariate statistics
print()
print('Unique values for Area variable')
print(medical.Area.unique())
print('Unique values for Services variable')
print(medical.Services.unique())