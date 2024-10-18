import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sqlite3

# Load the saved XGBoost model and Label Encoder
loaded_model = joblib.load('xgboost_model.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

# Define symptoms for the checkbox UI
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 
            'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 
            'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 
            'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 
            'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 
            'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 
            'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 
            'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 
            'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 
            'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
            'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
            'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness']
import sqlite3

def create_table_if_not_exists():
    conn = sqlite3.connect('healthcare_data.db')
    cursor = conn.cursor()
    
    # Create the table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_data (
        id TEXT PRIMARY KEY,
        name TEXT,
        itching INTEGER,
        skin_rash INTEGER,
        nodal_skin_eruptions INTEGER,
        continuous_sneezing INTEGER,
        shivering INTEGER,
        chills INTEGER,
        joint_pain INTEGER,
        stomach_pain INTEGER,
        acidity INTEGER,
        ulcers_on_tongue INTEGER,
        muscle_wasting INTEGER,
        vomiting INTEGER,
        burning_micturition INTEGER,
        spotting_urination INTEGER,
        fatigue INTEGER,
        weight_gain INTEGER,
        anxiety INTEGER,
        cold_hands_and_feets INTEGER,
        mood_swings INTEGER,
        weight_loss INTEGER,
        restlessness INTEGER,
        lethargy INTEGER,
        patches_in_throat INTEGER,
        irregular_sugar_level INTEGER,
        cough INTEGER,
        high_fever INTEGER,
        sunken_eyes INTEGER,
        breathlessness INTEGER,
        sweating INTEGER,
        dehydration INTEGER,
        indigestion INTEGER,
        headache INTEGER,
        yellowish_skin INTEGER,
        dark_urine INTEGER,
        nausea INTEGER,
        loss_of_appetite INTEGER,
        pain_behind_the_eyes INTEGER,
        back_pain INTEGER,
        constipation INTEGER,
        abdominal_pain INTEGER,
        diarrhoea INTEGER,
        mild_fever INTEGER,
        yellow_urine INTEGER,
        yellowing_of_eyes INTEGER,
        acute_liver_failure INTEGER,
        fluid_overload INTEGER,
        swelling_of_stomach INTEGER,
        swelled_lymph_nodes INTEGER,
        malaise INTEGER,
        blurred_and_distorted_vision INTEGER,
        phlegm INTEGER,
        throat_irritation INTEGER,
        redness_of_eyes INTEGER,
        sinus_pressure INTEGER,
        runny_nose INTEGER,
        congestion INTEGER,
        chest_pain INTEGER,
        weakness_in_limbs INTEGER,
        fast_heart_rate INTEGER,
        pain_during_bowel_movements INTEGER,
        pain_in_anal_region INTEGER,
        bloody_stool INTEGER,
        irritation_in_anus INTEGER,
        neck_pain INTEGER,
        dizziness INTEGER,
        cramps INTEGER,
        bruising INTEGER,
        obesity INTEGER,
        swollen_legs INTEGER,
        swollen_blood_vessels INTEGER,
        puffy_face_and_eyes INTEGER,
        enlarged_thyroid INTEGER,
        brittle_nails INTEGER,
        swollen_extremities INTEGER,
        excessive_hunger INTEGER,
        extra_marital_contacts INTEGER,
        drying_and_tingling_lips INTEGER,
        slurred_speech INTEGER,
        knee_pain INTEGER,
        hip_joint_pain INTEGER,
        muscle_weakness INTEGER,
        stiff_neck INTEGER,
        swelling_joints INTEGER,
        movement_stiffness INTEGER,
        spinning_movements INTEGER,
        loss_of_balance INTEGER,
        unsteadiness INTEGER,
        weakness_of_one_body_side INTEGER,
        loss_of_smell INTEGER,
        bladder_discomfort INTEGER,
        foul_smell_of_urine INTEGER,
        continuous_feel_of_urine INTEGER,
        passage_of_gases INTEGER,
        internal_itching INTEGER,
        toxic_look_typhos INTEGER,
        depression INTEGER,
        irritability INTEGER,
        muscle_pain INTEGER,
        altered_sensorium INTEGER,
        red_spots_over_body INTEGER,
        belly_pain INTEGER,
        abnormal_menstruation INTEGER,
        dischromic_patches INTEGER,
        watering_from_eyes INTEGER,
        increased_appetite INTEGER,
        polyuria INTEGER,
        family_history INTEGER,
        mucoid_sputum INTEGER,
        rusty_sputum INTEGER,
        lack_of_concentration INTEGER,
        visual_disturbances INTEGER,
        receiving_blood_transfusion INTEGER,
        receiving_unsterile_injections INTEGER,
        coma INTEGER,
        stomach_bleeding INTEGER,
        distention_of_abdomen INTEGER,
        history_of_alcohol_consumption INTEGER,
        blood_in_sputum INTEGER,
        prominent_veins_on_calf INTEGER,
        palpitations INTEGER,
        painful_walking INTEGER,
        pus_filled_pimples INTEGER,
        blackheads INTEGER,
        scurring INTEGER,
        skin_peeling INTEGER,
        silver_like_dusting INTEGER,
        small_dents_in_nails INTEGER,
        inflammatory_nails INTEGER,
        blister INTEGER,
        red_sore_around_nose INTEGER,
        yellow_crust_ooze INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()



# Function to save user data in a simple SQLite database
def save_to_database(patient_id, patient_name, data, prediction):
    conn = sqlite3.connect('healthcare_data.db')
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_data (
        id TEXT PRIMARY KEY,
        name TEXT,
        symptoms TEXT,
        prediction TEXT
    )''')

    # Insert user data
    cursor.execute('''
    INSERT INTO patient_data (id, name, symptoms, prediction)
    VALUES (?, ?, ?, ?)''', (patient_id, patient_name, str(data.tolist()), prediction))

    print(f"patient_id: {patient_id}, type: {type(patient_id)}")

    
    conn.commit()
    conn.close()

def check_patient_id_exists(patient_id):
    conn = sqlite3.connect('healthcare_data.db')
    cursor = conn.cursor()
    
    # Execute the query with patient_id passed as a tuple (with a comma after patient_id)
    cursor.execute('SELECT id FROM patient_data WHERE id=?', (patient_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    # Return whether the patient ID exists
    return result is not None


# Create a title for the Streamlit app
st.title("Disease Prediction Tool")

# Input fields for Patient's Name and ID
st.subheader("Enter Patient Details:")
patient_name = st.text_input("Patient Name")
patient_id = st.text_input("Patient ID")

# Create a checkbox for each symptom
st.subheader("Please select the symptoms you are experiencing:")
user_data = []
for symptom in symptoms:
    value = st.checkbox(symptom.replace('_', ' '))
    user_data.append(1 if value else 0)


# Call this function at the start of your code
create_table_if_not_exists()


# Create a button to run the prediction
if st.button('Predict'):
    if not patient_name or not patient_id:
        st.error("Please provide both Patient Name and ID!")
    elif check_patient_id_exists(patient_id):
        st.error(f"Patient ID '{patient_id}' already exists! Please use a unique ID.")
    else:
        new_data = pd.DataFrame([user_data], columns=symptoms)

        # Get predicted probabilities for each class
        proba = loaded_model.predict_proba(new_data)

        # Sort probabilities and get the top N classes (for example, top 3)
        top_n = 3
        top_n_indices = np.argsort(proba[0])[-top_n:][::-1]
        top_n_predictions = loaded_label_encoder.inverse_transform(top_n_indices)

        # Display the top N predicted classes
        st.subheader(f'Top {top_n} Predicted Diseases:')
        for i, disease in enumerate(top_n_predictions):
            st.write(f'{i + 1}. {disease}')

        # Save data to database
        save_to_database(patient_id, patient_name, new_data, str(top_n_predictions))

        # Success message
        st.success("Your data has been saved and predictions are shown above.")

# Option to search patient data by ID
st.subheader("Search Patient Data by ID")
search_id = st.text_input("Enter Patient ID to Search")

if st.button('Search'):
    if not search_id:
        st.error("Please enter a Patient ID to search!")
    else:
        conn = sqlite3.connect('healthcare_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patient_data WHERE id=?', (search_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            st.subheader(f"Patient Data for ID: {search_id}")
            st.write(f"Name: {result[1]}")
            st.write(f"Symptoms: {result[2]}")
            st.write(f"Prediction: {result[3]}")
        else:
            st.error(f"No data found for Patient ID: {search_id}")
