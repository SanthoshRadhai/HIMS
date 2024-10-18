import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('heart_disease_model_rf.joblib')  # Replace with the actual path of your saved model
scaler = joblib.load('scaler.joblib')  # Replace with the actual path of your saved scaler

# Load new data for prediction (replace with the path to your new data)
new_data = pd.read_csv('heartTest.csv')  # Ensure that this data has the same features as the training data

# Scale the new data using the loaded scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)

# If you want probability estimates, use:
probability_predictions = model.predict_proba(new_data_scaled)

# Output the predictions
print('Predictions:', predictions)
print('Probability Predictions:', probability_predictions)
