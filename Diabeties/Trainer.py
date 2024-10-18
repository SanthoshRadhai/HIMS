import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
data = pd.read_csv('DiaData.csv')  # Replace with your actual data path

# Split features and target
X = data.drop('diabetes', axis=1)  # Drop the target column from the features
y = data['diabetes']               # Target column (diabetes)

# Encode categorical columns
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

# Encode 'gender' and 'smoking_history' columns
X['gender'] = gender_encoder.fit_transform(X['gender'])
X['smoking_history'] = smoking_encoder.fit_transform(X['smoking_history'])

# Scale numerical features
scaler = StandardScaler()
X[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.fit_transform(
    X[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)  # More estimators and deeper trees
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the trained model using joblib
joblib.dump(model, 'diabetes_prediction_model_rf.joblib')

# Save the label encoders
joblib.dump(gender_encoder, 'gender_encoder.joblib')
joblib.dump(smoking_encoder, 'smoking_history_encoder.joblib')

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

print('Model, encoders, and scaler saved successfully!')
