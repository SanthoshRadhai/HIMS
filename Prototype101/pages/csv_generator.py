import sqlite3
import streamlit as st

# Connect to the database
conn = sqlite3.connect('symptom_data.db')
c = conn.cursor()

# Create the symptoms_data table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS symptoms_data (
    id INTEGER,
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
    swollen_extremeties INTEGER,
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
    yellow_crust_ooze INTEGER,
    FOREIGN KEY (id) REFERENCES patients(id) ON DELETE CASCADE
)
''')

# Define symptoms
symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", 
    "spotting_urination", "fatigue", "weight_gain", "anxiety", 
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", 
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", 
    "high_fever", "sunken_eyes", "breathlessness", "sweating", 
    "dehydration", "indigestion", "headache", "yellowish_skin", 
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", 
    "back_pain", "constipation", "abdominal_pain", "diarrhoea", 
    "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", 
    "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", 
    "malaise", "blurred_and_distorted_vision", "phlegm", 
    "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", 
    "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", 
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", 
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", 
    "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", 
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", 
    "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", 
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", 
    "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", 
    "movement_stiffness", "spinning_movements", "loss_of_balance", 
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", 
    "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine", 
    "passage_of_gases", "internal_itching", "toxic_look_typhos", 
    "depression", "irritability", "muscle_pain", "altered_sensorium", 
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", 
    "dischromic_patches", "watering_from_eyes", "increased_appetite", 
    "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", 
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", 
    "receiving_unsterile_injections", "coma", "stomach_bleeding", 
    "distention_of_abdomen", "history_of_alcohol_consumption", 
    "blood_in_sputum", "prominent_veins_on_calf", "palpitations", 
    "painful_walking", "pus_filled_pimples", "blackheads", 
    "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", 
    "inflammatory_nails", "blister", "red_sore_around_nose", 
    "yellow_crust_ooze"
]

# Create a dictionary to store user input
data = {symptom: st.selectbox(symptom, options=[0, 1]) for symptom in symptoms}

# Input for user ID
user_id = st.number_input("Enter your ID", min_value=1)

# Add a button to insert data into the database
if st.button("Submit Data to Database"):
    # Check if the user ID exists in the patients table
    c.execute('SELECT COUNT(*) FROM patients WHERE id = ?', (user_id,))
    exists = c.fetchone()[0]

    if exists:
        # Prepare the insert query
        columns = ', '.join([f'"{col}"' for col in data.keys()])
        placeholders = ', '.join(['?'] * len(data))
        query = f'INSERT INTO symptoms_data (id, {columns}) VALUES (?, {placeholders})'

        # Execute the insert query
        c.execute(query, (user_id, *data.values()))
        conn.commit()
        st.success("Data submitted successfully!")
    else:
        st.error("The entered ID does not exist in the patients table.")

# Close the connection when done
conn.close()
