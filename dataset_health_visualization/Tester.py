import pandas as pd
import joblib

# Load the saved XGBoost model and Label Encoder
loaded_model = joblib.load('xgboost_model.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

# Load the new CSV file (replace 'your_new_data.csv' with the actual file path)
# The CSV file is assumed to have the same headers as the training data except the 'prognosis' column
new_data = pd.read_csv('Testing001.csv')

# Ensure that the 'prognosis' column is not present in the new data (if present, drop it)
# if 'prognosis' in new_data.columns:
#     new_data = new_data.drop('prognosis', axis=1)

new_data = new_data.drop('prognosis', axis=1)  

# Make sure the data types match the model input
# Use the loaded model to make predictions
y_pred = loaded_model.predict(new_data)

# Decode the predictions back to the original class names (optional)
decoded_predictions = loaded_label_encoder.inverse_transform(y_pred)

# Display the decoded predictions
print(f"Predicted prognosis (encoded): {y_pred}")
print(f"Predicted prognosis (decoded): {decoded_predictions}")
