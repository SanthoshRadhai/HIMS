import streamlit as st
import sqlite3

# Connect to the existing symptoms_data database
conn = sqlite3.connect('symptom_data.db')
c = conn.cursor()

# Create a patients table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY,
        name TEXT,
        FOREIGN KEY (id) REFERENCES symptoms_data (id) ON DELETE CASCADE
    )
''')
conn.commit()

# Home page
def main():
    st.title("Welcome to the Medical Data App")
    
    # Registration Form
    st.header("Register Your Details")
    name = st.text_input("Enter your name")
    user_id = st.number_input("Enter your ID", min_value=1)  # Ensuring ID is a positive integer
    
    if st.button("Register"):
        # Insert user data into the database
        if name and user_id:
            try:
                c.execute('INSERT INTO patients (id, name) VALUES (?, ?)', (user_id, name))
                conn.commit()
                st.success(f"User {name} (ID: {user_id}) registered successfully!")
            except sqlite3.IntegrityError:
                st.error(f"User ID {user_id} already exists. Please use a different ID.")
        else:
            st.error("Please enter both name and ID.")
    
    # Navigation to other pages
    if st.button("Go to Disease Prediction"):
        st.query_params(page="disease_prediction")
    if st.button("Go to CSV Generator"):
        st.query_params(page="csv_generator")

if __name__ == "__main__":
    main()
