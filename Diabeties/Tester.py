import pandas as pd
import joblib

# Load the saved model, scaler, and separate label encoders
model = joblib.load('diabetes_prediction_model.joblib')
scaler = joblib.load('scaler.joblib')
gender_encoder = joblib.load('gender_encoder.joblib')
smoking_encoder = joblib.load('smoking_history_encoder.joblib')

# Load the new dataset for prediction
# Ensure the new data file has the same structure as the training data (excluding the 'diabetes' column)
new_data = pd.read_csv('DiaTest.csv')

# Preprocessing (apply the same transformations as done during training)
# Encode categorical columns ('gender' and 'smoking_history')

# Handle unseen categories by setting them to a default value, if necessary
new_data['gender'] = new_data['gender'].apply(lambda x: x if x in gender_encoder.classes_ else 'Female')
new_data['smoking_history'] = new_data['smoking_history'].apply(lambda x: x if x in smoking_encoder.classes_ else 'never')

# Encode the categorical columns with the correct encoders
new_data['gender'] = gender_encoder.transform(new_data['gender'])
new_data['smoking_history'] = smoking_encoder.transform(new_data['smoking_history'])

# Standardize numerical columns ('age', 'bmi', 'HbA1c_level', 'blood_glucose_level')
new_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(
    new_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])

# Get predicted probabilities
probabilities = model.predict_proba(new_data)

# We are interested in the probability of the second class (Diabetes = 1)
diabetes_probabilities = probabilities[:, 1]  # Get the probability of having diabetes (class 1)

# Print the predicted probabilities on a scale from 0 to 1
print(f'Predicted Diabetes Probabilities: {diabetes_probabilities}')
