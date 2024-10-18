import pandas as pd

# Load datasets
df_diabet = pd.read_csv('diabetes_prediction_dataset.csv')
df_heart = pd.read_csv('heart.csv')
df_kidney = pd.read_csv('kidney_disease.csv')
df_lung = pd.read_csv('lungcancer.csv')
df_stroke = pd.read_csv('stroke-data.csv')

# Concatenate them along the columns axis (axis=1)
df_combined = pd.concat([df_diabet, df_heart], axis=1)

# Fill missing values with NaN or some default
df_combined.fillna(0, inplace=True)

print(df_combined)