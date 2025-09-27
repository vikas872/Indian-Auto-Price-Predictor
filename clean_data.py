import pandas as pd
import datetime

print("--- Starting Data Cleaning and Preparation ---")

# --- 1. Load the Data ---
try:
    df = pd.read_csv('car data.csv')
    print(f"Successfully loaded car data.csv with {df.shape[0]} rows.")
except FileNotFoundError:
    print("Error: car data.csv not found.")
    exit()

# --- 2. Feature Engineering: Create 'Car_Age' ---
# The current year is 2025, which we'll use as our reference.
current_year = 2025
df['Car_Age'] = current_year - df['Year']
print("Created 'Car_Age' feature.")

# --- 3. Feature Selection: Drop Unnecessary Columns ---
# We drop the original 'Year' column as 'Car_Age' is more useful.
# We drop 'Car_Name' as it has too many unique values for this small dataset (high cardinality).
df.drop(columns=['Year', 'Car_Name'], inplace=True)
print("Dropped 'Year' and 'Car_Name' columns.")

# --- 4. Final Check and Save ---
print("\n--- Preparation Summary ---")
print(f"Shape of the final, prepared dataset: {df.shape}")
print("\nFinal dataset columns:")
print(df.columns)
print("\nFirst 5 rows of prepared data:")
print(df.head())


# Save the cleaned dataframe to a new CSV file.
df.to_csv('cleaned_car_data.csv', index=False)
print("\nPrepared data has been successfully saved to 'cleaned_car_data.csv'")

print("\n--- Data Preparation Finished ---")