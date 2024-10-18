import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.svm import SVC


# Load your dataset
data = pd.read_csv('heart.csv')  # Replace with your actual dataset path

# Split features and target
X = data.drop('target', axis=1)  # Drop the target column
y = data['target']               # Target column (heart disease)

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the SVM model
model = SVC(kernel='rbf', probability=True)  # 'rbf' kernel is powerful for non-linear problems
model.fit(X_train, y_train)


# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the trained model using joblib
joblib.dump(model, 'heart_disease_model_rf.joblib')

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

print('Model and scaler saved successfully!')
