import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the saved XGBoost model and Label Encoder
loaded_model = joblib.load('xgboost_model.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

# Load the new CSV file (replace 'Testing001.csv' with the actual file path)
new_data = pd.read_csv('Testing001.csv')

# Drop the 'prognosis' column if present
new_data = new_data.drop('prognosis', axis=1, errors='ignore')  

# Get predicted probabilities for each class
proba = loaded_model.predict_proba(new_data)

# Sort probabilities and get the top N classes (for example, top 3)
top_n = 3
top_n_indices = np.argsort(proba[0])[-top_n:][::-1]
top_n_predictions = loaded_label_encoder.inverse_transform(top_n_indices)

# Print the top N predicted classes
print(f'Top {top_n} predictions: {top_n_predictions}')

# Display the decoded predictions
print(f"Predicted prognosis (encoded): {top_n_indices}")
print(f"Predicted prognosis (decoded): {top_n_predictions}")

# Plotting the disease classes in a square
disease_classes = loaded_label_encoder.classes_  # Get all disease classes
n_classes = len(disease_classes)

# Create square coordinates for the classes
x = np.linspace(-1, 1, int(np.ceil(np.sqrt(n_classes))))  # Create x coordinates
y = np.linspace(-1, 1, int(np.ceil(np.sqrt(n_classes))))  # Create y coordinates
X, Y = np.meshgrid(x, y)  # Create a meshgrid

# Flatten the coordinates
coordinates = np.vstack([X.ravel(), Y.ravel()]).T

# Generate a color map for the classes
colors = plt.cm.get_cmap('tab20', n_classes)  # Use a colormap with distinct colors

# Create a scatter plot for disease classes
plt.figure(figsize=(10, 10))
for i, disease in enumerate(disease_classes):
    plt.scatter(coordinates[i, 0], coordinates[i, 1], s=100, color=colors(i), label=disease)

# Draw parabolic curves between the predicted diseases
for i in range(top_n):
    for j in range(i + 1, top_n):  # Connect each pair of top predictions
        x_start, y_start = coordinates[top_n_indices[i], 0], coordinates[top_n_indices[i], 1]
        x_end, y_end = coordinates[top_n_indices[j], 0], coordinates[top_n_indices[j], 1]

        # Calculate points for the parabolic curve
        t = np.linspace(0, 1, 100)
        # Parabola formula
        x_curve = (1 - t) * x_start + t * x_end
        y_curve = -4 * (x_curve - (x_start + x_end) / 2) ** 2 + (y_start + y_end) / 2  # Inverted parabola

        # Plot the parabolic curve
        plt.plot(x_curve, y_curve, color='blue', alpha=0.5)

# Highlight the predicted diseases
for index in top_n_indices:
    plt.scatter(coordinates[index, 0], coordinates[index, 1], s=200, edgecolor='red', linewidth=2)

# Add labels for all disease classes
for i, disease in enumerate(disease_classes):
    plt.annotate(disease, (coordinates[i, 0], coordinates[i, 1]), fontsize=10, ha='center', color=colors(i))

# Add labels for the predicted diseases
for index in top_n_indices:
    plt.annotate(disease_classes[index], (coordinates[index, 0], coordinates[index, 1]), fontsize=12, ha='center', color='red')

# Create a text box on the right side of the plot to display all disease labels
plt.text(1.2, 0, '\n'.join(disease_classes), fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

plt.title('Top 3 Disease Predictions with Interconnected Associations')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid()
plt.show()
