import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('symptom_data.db')  # Update with your actual database path

# Function to fetch data from the database
def fetch_data():
    query = "SELECT * FROM symptoms_data"
    df = pd.read_sql_query(query, conn)
    return df

# Main function to display data
def main():
    try:
        # Fetch the data
        data = fetch_data()

        # Display the data
        print("Symptom Data:")
        print(data)

        # Optional: Save the data to a CSV file
        data.to_csv('symptom_data.csv', index=False)
        print("\nData has been saved to 'symptom_data.csv'.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the database connection
        conn.close()

# Run the main function
if __name__ == "__main__":
    main()
