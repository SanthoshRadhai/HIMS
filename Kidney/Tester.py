import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load the pre-trained model, scaler, and label encoders
model = joblib.load('ckd_prediction_model_xgb.joblib')
scaler = joblib.load('ckd_scaler.joblib')
label_encoders = joblib.load('ckd_label_encoders.joblib')
classification_encoder = joblib.load('classification_encoder.joblib')

# Load the new dataset for prediction (without the 'classification' column)
new_data = pd.read_csv('TestData.csv')  # Replace with your new dataset path

new_data = new_data.drop('id',axis = 1)

# Handle missing values in the new data
new_data.replace(r'^\s*\?+\s*$', pd.NA, regex=True, inplace=True)  # Replace '\t?' with pd.NA
new_data = new_data.astype(object).where(pd.notnull(new_data), np.nan)  # Convert pd.NA to np.nan

# Encode categorical features using the saved label encoders
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

for col in categorical_cols:
    le = label_encoders[col]
    
    # Fill missing values with the most frequent category from the new data
    if new_data[col].mode().size > 0:
        most_frequent = new_data[col].mode()[0]  # Get the most frequent value from the new data
        new_data[col] = new_data[col].fillna(most_frequent)  # Fill missing with most frequent value
        
        # Check if any unseen labels exist and map them to the most frequent class seen during training
        unseen_values = new_data[col][~new_data[col].isin(le.classes_)]
        if not unseen_values.empty:
            print(f"Warning: Unseen values {unseen_values.unique()} in column {col}. Filling with most frequent class.")
            new_data[col][~new_data[col].isin(le.classes_)] = most_frequent
        
        # Transform the column using the label encoder
        new_data[col] = le.transform(new_data[col])
    else:
        # If mode is not available, fill with the most common value seen during training
        most_frequent_train_value = le.classes_[0]
        new_data[col] = new_data[col].fillna(most_frequent_train_value)
        new_data[col] = le.transform(new_data[col])

# Scale numerical features using the saved scaler
new_data_scaled = scaler.transform(new_data)

# Make probability predictions using the pre-trained model
probabilities = model.predict_proba(new_data_scaled)

# Extract probabilities for the positive class (assuming 'ckd' is the positive class)
positive_class_probabilities = probabilities[:, 1]  # Get probabilities for the 'ckd' class

# Display the probabilities
for i, prob in enumerate(positive_class_probabilities):
    print(f"Patient {i+1}: Probability of CKD = {prob:.2f}")
