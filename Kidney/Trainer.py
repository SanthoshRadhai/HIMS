import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import xgboost as xgb

# Load your dataset
data = pd.read_csv('kidney_disease.csv')  # Replace with your actual dataset path

# Drop the 'id' column since it's not a feature
data = data.drop('id', axis=1)

# Clean the data: Replace non-numeric values like '\t?' with NaN and handle them
data.replace(r'^\s*\?+\s*$', pd.NA, regex=True, inplace=True)  # Replace '\t?' with pd.NA

# Convert pd.NA to np.nan for XGBoost compatibility
data = data.astype(object).where(pd.notnull(data), np.nan)

# Handle missing values (fill with mean for numerical, mode for categorical)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Split features and target
X = data.drop('classification', axis=1)  # Drop the target column (classification)
y = data['classification']  # Target column

# Encode the target column ('classification')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode 'ckd' and 'notckd'

# Encode categorical features
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# XGBoost can handle missing values natively, so no need for explicit imputation

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', missing=np.nan, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the trained model using joblib
joblib.dump(model, 'ckd_prediction_model_xgb.joblib')

# Save the scaler and label encoders
joblib.dump(scaler, 'ckd_scaler.joblib')
joblib.dump(label_encoders, 'ckd_label_encoders.joblib')
joblib.dump(label_encoder, 'classification_encoder.joblib')  # Save the target encoder

print('Model, scaler, label encoders, and classification encoder saved successfully!')
