import pandas as pd

print("--- Starting Data Exploration for Indian Car Dataset ---")

# --- 1. Load the Data ---
try:
    # We are using 'car data.csv'
    df = pd.read_csv('car data.csv')
    print("Successfully loaded car data.csv")
except FileNotFoundError:
    print("Error: car data.csv not found. Please make sure the file is in the correct folder.")
    exit()

# --- 2. Initial Inspection ---

# Print the shape of the dataset (rows, columns)
print("\n--- Dataset Shape ---")
print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# Print the first 5 rows to see the data structure
print("\n--- First 5 Rows of Data ---")
print(df.head())

# Print a concise summary of the dataframe
print("\n--- Technical Summary & Data Types ---")
df.info()

# Print the count of missing values for each column
print("\n--- Count of Missing Values per Column ---")
print(df.isnull().sum())

print("\n--- Exploration Script Finished ---")