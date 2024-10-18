# pages/disease_prediction.py
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# Load the saved XGBoost model and Label Encoder
loaded_model = joblib.load('xgboost_model.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

st.header("Disease Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    # Drop the 'prognosis' column if present
    new_data = new_data.drop('prognosis', axis=1, errors='ignore')  

    # Get predicted probabilities for each class
    proba = loaded_model.predict_proba(new_data)

    # Sort probabilities and get the top N classes (for example, top 3)
    top_n = 3
    top_n_indices = np.argsort(proba[0])[-top_n:][::-1]
    top_n_predictions = loaded_label_encoder.inverse_transform(top_n_indices)

    # Display the top N predicted classes
    st.write(f'Top {top_n} predictions: {top_n_predictions}')

    # Plotting the disease classes in a square
    disease_classes = loaded_label_encoder.classes_
    n_classes = len(disease_classes)

    # Create square coordinates for the classes
    x = np.linspace(-1, 1, int(np.ceil(np.sqrt(n_classes))))
    y = np.linspace(-1, 1, int(np.ceil(np.sqrt(n_classes))))
    X, Y = np.meshgrid(x, y)

    coordinates = np.vstack([X.ravel(), Y.ravel()]).T

    colors = plt.cm.get_cmap('tab20', n_classes)

    plt.figure(figsize=(10, 10))
    for i, disease in enumerate(disease_classes):
        plt.scatter(coordinates[i, 0], coordinates[i, 1], s=100, color=colors(i), label=disease)

    # Draw parabolic curves between the predicted diseases
    for i in range(top_n):
        for j in range(i + 1, top_n):
            x_start, y_start = coordinates[top_n_indices[i], 0], coordinates[top_n_indices[i], 1]
            x_end, y_end = coordinates[top_n_indices[j], 0], coordinates[top_n_indices[j], 1]

            t = np.linspace(0, 1, 100)
            x_curve = (1 - t) * x_start + t * x_end
            y_curve = -4 * (x_curve - (x_start + x_end) / 2) ** 2 + (y_start + y_end) / 2

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

    plt.title('Top 3 Disease Predictions with Interconnected Associations')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.grid()

    # Show the plot in Streamlit
    st.pyplot(plt)
