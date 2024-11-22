import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('C:/Users/User/OneDrive/Documents/Degree/SEM 4/Machine Learning/Bengaluru_House_Data.csv')

# Clean 'total_sqft' to handle ranges
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            vals = x.split('-')
            return (float(vals[0]) + float(vals[1])) / 2
        return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

# Handle missing values
df['total_sqft'].fillna(df['total_sqft'].median(), inplace=True)
df['bath'].fillna(df['bath'].median(), inplace=True)
df['balcony'].fillna(df['balcony'].median(), inplace=True)

# Drop unnecessary columns: 'society' and 'availability'
df = df.drop(['society', 'availability'], axis=1)

# Outlier removal using IQR
Q1 = df['price'].quantile(0.25)
print("Q1 =", Q1)
Q3 = df['price'].quantile(0.75)
print("Q3 =", Q3)
IQR = Q3 - Q1
df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR)))]
print("IQR =", IQR)

# Feature engineering: Add new feature 'price_per_sqft'
df['price_per_sqft'] = df['price'] / df['total_sqft']

# Fit LabelEncoder on the remaining categorical columns
label_encoder = LabelEncoder()
for col in ['area_type', 'location', 'size']:
    df[col] = label_encoder.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop(['price'], axis=1)
y = df['price']  # Use the original price for training



# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluate Random Forest and XGBoost before hyperparameter tuning

# Random Forest Regressor (Default Parameters)
rf_base_model = RandomForestRegressor(random_state=42)  # Default Random Forest
rf_base_model.fit(X_train, y_train)
y_pred_rf_base = rf_base_model.predict(X_test)

# Evaluate Random Forest before tuning
mse_rf_base = mean_squared_error(y_test, y_pred_rf_base)
rmse_rf_base = np.sqrt(mse_rf_base)
r2_rf_base = r2_score(y_test, y_pred_rf_base)

print("=== Random Forest Before Tuning ===")
print(f"Mean Squared Error: {mse_rf_base}")
print(f"Root Mean Squared Error: {rmse_rf_base}")
print(f"R-squared: {r2_rf_base}\n")

# XGBoost Regressor (Default Parameters)
xgb_base_model = XGBRegressor(random_state=42)  # Default XGBoost
xgb_base_model.fit(X_train, y_train)
y_pred_xgb_base = xgb_base_model.predict(X_test)

# Evaluate XGBoost before tuning
mse_xgb_base = mean_squared_error(y_test, y_pred_xgb_base)
rmse_xgb_base = np.sqrt(mse_xgb_base)
r2_xgb_base = r2_score(y_test, y_pred_xgb_base)

print("=== XGBoost Before Tuning ===")
print(f"Mean Squared Error: {mse_xgb_base}")
print(f"Root Mean Squared Error: {rmse_xgb_base}")
print(f"R-squared: {r2_xgb_base}\n")

# Expanded hyperparameter search space for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10, 12, 15],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4]
}
random_rf = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_grid_rf, 
                               n_iter=20, scoring='neg_mean_squared_error', cv=5, verbose=2, random_state=42, n_jobs=-1)
random_rf.fit(X_train, y_train)
print("Best params for Random Forest:", random_rf.best_params_)

# Train Random Forest Regressor with the best hyperparameters
rf_model = RandomForestRegressor(**random_rf.best_params_, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest after tuning
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest Root Mean Squared Error: {rmse_rf}")
print(f"Random Forest R-squared: {r2_rf}")
print("\n")

# Expanded hyperparameter search space for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample': [0.7, 0.8, 1.0]
}
random_xgb = RandomizedSearchCV(XGBRegressor(random_state=42), param_distributions=param_grid_xgb, 
                                n_iter=20, scoring='neg_mean_squared_error', cv=5, verbose=2, random_state=42, n_jobs=-1)
random_xgb.fit(X_train, y_train)
print("Best params for XGBoost:", random_xgb.best_params_)

# Train XGBoost Regressor with the best hyperparameters
xgb_model = XGBRegressor(**random_xgb.best_params_, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost after tuning
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Mean Squared Error: {mse_xgb}")
print(f"XGBoost Root Mean Squared Error: {rmse_xgb}")
print(f"XGBoost R-squared: {r2_xgb}")


# Data for before and after tuning
models = ['Random Forest', 'XGBoost']
mse_before = [20.42, 24.46]
rmse_before = [4.52, 4.95]
r2_before = [0.9883, 0.9860]

mse_after = [18.19, 24.31]
rmse_after = [4.26, 4.93]
r2_after = [0.9896, 0.9861]

# Plotting MSE before and after tuning
plt.figure(figsize=(10, 6))
x = np.arange(len(models))

# Bar width
width = 0.3

# Plotting MSE
plt.bar(x - width, mse_before, width, label='MSE Before Tuning', color='blue')
plt.bar(x, mse_after, width, label='MSE After Tuning', color='green')

plt.xticks(x, models)
plt.xlabel("Models")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE Comparison Before and After Tuning")
plt.legend()
plt.show()

# Plotting RMSE before and after tuning
plt.figure(figsize=(10, 6))
plt.bar(x - width, rmse_before, width, label='RMSE Before Tuning', color='orange')
plt.bar(x, rmse_after, width, label='RMSE After Tuning', color='purple')

plt.xticks(x, models)
plt.xlabel("Models")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("RMSE Comparison Before and After Tuning")
plt.legend()
plt.show()

# Plotting R-squared before and after tuning
plt.figure(figsize=(10, 6))
plt.bar(x - width, r2_before, width, label='R² Before Tuning', color='red')
plt.bar(x, r2_after, width, label='R² After Tuning', color='cyan')

plt.xticks(x, models)
plt.xlabel("Models")
plt.ylabel("R-squared (R²)")
plt.title("R² Comparison Before and After Tuning")
plt.legend()
plt.show()

