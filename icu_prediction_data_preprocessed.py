# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_excel("C:/Users/junch/OneDrive/Documents/BigData/Project/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")

# Understand the data structured
print('\n', data.head())

# Data Preparation: Turn categorical variables into numerics
data['AGE_PERCENTIL'] = data['AGE_PERCENTIL'].str.replace('Above ', '').str.extract(r'(.+?)th')
data['WINDOW'] = data['WINDOW'].str.replace('ABOVE_12', '12-more').str.extract(r'(.+?)-')

#check the missing value percentage
# Calculate the number and percentage of missing values in each column
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100

missing_data_summary = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
print('\n', missing_data_summary)

# Create a new feature for missingness within each row
data['row_missing_columns'] = data.isnull().sum(axis=1)

# Mean imputation for missing values
mean_impute = SimpleImputer(strategy='mean')
imputed_data = mean_impute.fit_transform(data)

# Convert the imputed data back to a DataFrame
imputed_data = pd.DataFrame(imputed_data, columns=data.columns)

# Verify if all missing values are handled
missing_values_after = imputed_data.isnull().sum()
missing_data_summary_after = pd.DataFrame({'Missing Values': missing_values_after})
print('\n', missing_data_summary_after)

#check is there duplicate data
duplicate_count = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")
#since no duplicate data, hence no need to handle it

# Display the first few rows of the preprocessed data
print('\n', imputed_data.head())

# Save the preprocessed data to a new CSV file
imputed_data.to_csv("C:/Users/junch/OneDrive/Documents/BigData/Project/Preprocess/Kaggle_Sirio_preprocessed.csv", index=False)