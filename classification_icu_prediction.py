import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv('C:/Users/User/.spyder-py3/Kaggle_Sirio_preprocessed.csv')

# Define independent and dependent variables
X = data[list(data.columns)[:-1]].values
y = data[data.columns[-1]].values

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(np.nan_to_num(X_train))
X_test = scaler.transform(np.nan_to_num(X_test))

# Train Random Forest Classifier before tuning
rf_model_before = RandomForestClassifier(random_state=42)  # Default parameters
rf_model_before.fit(X_train, y_train)
y_pred_rf_before = rf_model_before.predict(X_test)

# Evaluate Random Forest before tuning
acc_rf_before = metrics.accuracy_score(y_test, y_pred_rf_before)
print('Random Forest Accuracy (Before Tuning):', acc_rf_before)

# Train XGBoost Classifier before tuning
xgb_model_before = XGBClassifier(random_state=42)  # Default parameters
xgb_model_before.fit(X_train, y_train)
y_pred_xgb_before = xgb_model_before.predict(X_test)

# Evaluate XGBoost before tuning
acc_xgb_before = metrics.accuracy_score(y_test, y_pred_xgb_before)
print("XGBoost Accuracy (Before Tuning):", acc_xgb_before)

# Random Forest Hyperparameter Tuning
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid_rf,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X_train, y_train)
print("Best Parameters for Random Forest:", random_search_rf.best_params_)

# Train Random Forest with Best Parameters
rf_model_after = random_search_rf.best_estimator_
rf_model_after.fit(X_train, y_train)
y_pred_rf_after = rf_model_after.predict(X_test)

# Evaluate Random Forest after tuning
acc_rf_after = metrics.accuracy_score(y_test, y_pred_rf_after)
print('Random Forest Accuracy (After Tuning):', acc_rf_after)

# XGBoost Hyperparameter Tuning
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'subsample': [0.6, 0.8, 1.0]
}

random_search_xgb = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_distributions=param_grid_xgb,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

random_search_xgb.fit(X_train, y_train)
print("Best Parameters for XGBoost:", random_search_xgb.best_params_)

# Train XGBoost with Best Parameters
xgb_model_after = random_search_xgb.best_estimator_
xgb_model_after.fit(X_train, y_train)
y_pred_xgb_after = xgb_model_after.predict(X_test)

# Evaluate XGBoost after tuning
acc_xgb_after = metrics.accuracy_score(y_test, y_pred_xgb_after)
print("XGBoost Accuracy (After Tuning):", acc_xgb_after)

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf_after)
print("Random Forest Confusion Matrix:\n", cm_rf)

# Confusion Matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb_after)
print("XGBoost Confusion Matrix:\n", cm_xgb)

# Visualize Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy values for Random Forest and XGBoost before and after tuning
accuracy_before = [0.97, 0.96]  # Random Forest and XGBoost before tuning
accuracy_after = [0.96, 0.96]   # Random Forest and XGBoost after tuning
models = ['Random Forest', 'XGBoost']

# Create the bar chart
x = range(len(models))
width = 0.4

plt.bar([i - width/2 for i in x], accuracy_before, width=width, label='Accuracy Before Tuning', color='blue')
plt.bar([i + width/2 for i in x], accuracy_after, width=width, label='Accuracy After Tuning', color='green')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Before and After Tuning')
plt.xticks(x, models)
plt.legend()


ax[0].imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
ax[0].set_title('Random Forest Confusion Matrix')
ax[0].set_xticks(np.arange(len(label_encoder.classes_)))
ax[0].set_yticks(np.arange(len(label_encoder.classes_)))
ax[0].set_xticklabels(label_encoder.classes_, rotation=45)
ax[0].set_yticklabels(label_encoder.classes_)
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

ax[1].imshow(cm_xgb, interpolation='nearest', cmap=plt.cm.Blues)
ax[1].set_title('XGBoost Confusion Matrix')
ax[1].set_xticks(np.arange(len(label_encoder.classes_)))
ax[1].set_yticks(np.arange(len(label_encoder.classes_)))
ax[1].set_xticklabels(label_encoder.classes_, rotation=45)
ax[1].set_yticklabels(label_encoder.classes_)
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
