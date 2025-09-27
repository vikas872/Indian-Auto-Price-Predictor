# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib  # For saving Python objects (our model and columns)

print("--- Starting Model Training for Indian Car Dataset ---")

# --- 1. Load the Prepared Data ---
# Try to load the dataset created by clean_data.py.
# If the file is not found, exit the script with an error message.
try:
    df = pd.read_csv('cleaned_car_data.csv')
    print("Successfully loaded cleaned_car_data.csv")
except FileNotFoundError:
    print("Error: cleaned_car_data.csv not found. Please run clean_data.py first.")
    exit()

# --- 2. One-Hot Encoding for Categorical Features ---
# Machine learning models require all input features to be numerical.
# We use pd.get_dummies() to convert categorical columns (like 'Fuel_Type')
# into numerical format. 'drop_first=True' avoids redundant columns.
df_encoded = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

print(f"Data shape after one-hot encoding: {df_encoded.shape}")

# --- 3. Define Features (X) and Target (y) ---
# 'X' contains all our input features (everything except the price).
# 'y' is the target variable we want to predict (the selling price).
X = df_encoded.drop('Selling_Price', axis=1)
y = df_encoded['Selling_Price']

# --- 4. Save the Model Columns ---
# This is a crucial step for deployment. We save the exact order and names
# of the columns that the model was trained on. This prevents errors in the Flask app.
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_columns_indian.pkl')
print("\nSuccessfully saved the model columns to 'model_columns_indian.pkl'")

# --- 5. Split Data into Training and Testing Sets ---
# We split the data to train the model on one part (80%) and test its
# performance on unseen data (20%). 'random_state=42' ensures the split is the same every time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 6. Train the Random Forest Model ---
print("\n--- Training Random Forest Model ---")
# Initialize the RandomForestRegressor. 'n_estimators' is the number of trees in the forest.
# 'random_state=42' ensures the model's randomness is the same each time for reproducibility.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model using our training data.
rf_model.fit(X_train, y_train)

# --- 7. Evaluate the Model ---
# Make predictions on the unseen test data.
y_pred = rf_model.predict(X_test)

# Calculate performance metrics.
# R-squared (R²) measures how much of the price variance the model can explain.
r2 = r2_score(y_test, y_pred)
# Mean Absolute Error (MAE) is the average error of the predictions in Lakhs.
mae = mean_absolute_error(y_test, y_pred)

print(f"Model R-squared (R²): {r2:.4f}")
print(f"Model Mean Absolute Error (MAE): {mae:.4f} Lakhs")
# Convert MAE from Lakhs to Rupees for a more understandable interpretation.
print(f"This means the model's price predictions are, on average, off by about ₹{mae*100000:,.2f}.")

# --- 8. Save the Trained Model ---
# We save the trained model object to a file using joblib.
# This allows our Flask app to load and use the pre-trained model.
joblib.dump(rf_model, 'random_forest_model_indian.pkl')
print("\nSuccessfully saved the trained model to 'random_forest_model_indian.pkl'")

# --- 9. Display Feature Importances ---
# We can inspect the trained model to see which features it found most important.
print("\n--- Top 5 Most Important Features ---")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feature_importances.nlargest(5).to_string())

print("\n--- Model Training Finished ---")

