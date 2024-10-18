import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import numpy as np

# Load your dataset
data = pd.read_csv('General/Training.csv')

# Check if 'Unnamed: 133' exists and drop it if present
if 'Unnamed: 133' in data.columns:
    data = data.drop('Unnamed: 133', axis=1)

# Split features and target
X = data.drop('prognosis', axis=1)  # Drop the target column from the features
y = data['prognosis']               # Target column (prognosis)

# Label encode the target column (prognosis)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(y_encoded)

print(X.shape)
print(y.shape)

# Now 'y_encoded' contains numerical values instead of disease names

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
print(X_train.shape)
print(y_train.shape)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model (you can use metrics like accuracy_score or confusion_matrix)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# If you need to map back the encoded labels to original disease names, you can use:
original_labels = label_encoder.inverse_transform(y_pred)

# Save the trained model using joblib
joblib.dump(model, 'xgboost_model.joblib')

# Save the label encoder as well (in case you need to decode predictions later)
joblib.dump(label_encoder, 'label_encoder.joblib')

print('Model and label encoder saved successfully!')

# --- Perform prediction on a random row from X_train ---
# Select a random row from X_train
random_index = np.random.randint(0, X_train.shape[0])
random_row = X_train.iloc[[random_index]]

# Perform prediction
random_prediction_encoded = model.predict(random_row)

# Decode the prediction back to the original prognosis label
random_prediction = label_encoder.inverse_transform(random_prediction_encoded)

print(f'Random row index: {random_index}')
print(f'Features of random row: \n{random_row}')
print(f'Predicted prognosis (encoded): {random_prediction_encoded}')
print(f'Predicted prognosis (decoded): {random_prediction}')
