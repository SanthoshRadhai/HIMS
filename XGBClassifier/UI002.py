import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Load the saved XGBoost model and Label Encoder
loaded_model = joblib.load('xgboost_model.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

# Streamlit interface
st.title('Disease Prediction with XGBoost')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file
    new_data = pd.read_csv(uploaded_file)
    
    # Drop the 'prognosis' column if present
    new_data = new_data.drop('prognosis', axis=1, errors='ignore')  

    # Get predicted probabilities for each class
    proba = loaded_model.predict_proba(new_data)

    # Sort probabilities and get the top N classes (for example, top 3)
    top_n = 3
    top_n_indices = np.argsort(proba[0])[-top_n:][::-1]
    top_n_predictions = loaded_label_encoder.inverse_transform(top_n_indices)

    # Display the top N predictions
    st.write(f'Top {top_n} predictions: {top_n_predictions}')

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
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, disease in enumerate(disease_classes):
        ax.scatter(coordinates[i, 0], coordinates[i, 1], s=100, color=colors(i), label=disease)

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
            ax.plot(x_curve, y_curve, color='blue', alpha=0.5)

    # Highlight the predicted diseases
    for index in top_n_indices:
        ax.scatter(coordinates[index, 0], coordinates[index, 1], s=200, edgecolor='red', linewidth=2)

    # Add labels for all disease classes
    for i, disease in enumerate(disease_classes):
        ax.annotate(disease, (coordinates[i, 0], coordinates[i, 1]), fontsize=10, ha='center', color=colors(i))

    # Add labels for the predicted diseases
    for index in top_n_indices:
        ax.annotate(disease_classes[index], (coordinates[index, 0], coordinates[index, 1]), fontsize=12, ha='center', color='red')

    # Add labels for all disease classes in a text box
    st.write('\n'.join(disease_classes))

    # Display the plot in Streamlit
    st.pyplot(fig)
