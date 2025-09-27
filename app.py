import pandas as pd
import joblib
from flask import Flask, render_template, request

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Trained Model and Data Columns ---
# Load the pre-trained Random Forest model for the Indian market
model = joblib.load('random_forest_model_indian.pkl')

# Load the column names that the model was trained on
data_columns = joblib.load('model_columns_indian.pkl')


# --- Define the Routes ---

@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives user input, prepares it for the model, and returns the prediction."""
    
    # Get all the input values from the form as a dictionary
    input_features = request.form.to_dict()

    # --- Prepare the input data for the model ---
    # Create a dataframe with a single row of zeros, using the loaded column names
    prediction_df = pd.DataFrame(columns=data_columns)
    prediction_df.loc[0] = 0

    # The model expects 'Present_Price' in Lakhs, but the user inputs it in Rupees.
    # We convert the input from Rupees to Lakhs.
    present_price_rupees = float(input_features.get('Present_Price', 0))
    prediction_df['Present_Price'] = present_price_rupees / 100000.0

    # Update the other numerical features
    prediction_df['Kms_Driven'] = int(input_features.get('Kms_Driven', 0))
    prediction_df['Owner'] = int(input_features.get('Owner', 0))
    prediction_df['Car_Age'] = int(input_features.get('Car_Age', 0))

    # Update the one-hot encoded categorical features
    # For each categorical feature, find its corresponding column and set the value to 1
    for feature, value in input_features.items():
        if feature in ['Fuel_Type', 'Seller_Type', 'Transmission']:
            column_name = f"{feature}_{value}"
            if column_name in data_columns:
                prediction_df[column_name] = 1
    
    # Ensure all columns are in the correct order as during training
    prediction_df = prediction_df[data_columns]
                
    # --- Make the Prediction ---
    # The model will predict the price in Lakhs.
    prediction_in_lakhs = model.predict(prediction_df)
    
    # Convert the prediction from Lakhs to Rupees for display
    prediction_in_rupees = prediction_in_lakhs[0] * 100000
    
    # Format the output with the Rupee symbol and commas
    output = f"₹{prediction_in_rupees:,.2f}"

    # ** THIS IS THE UPDATED LINE **
    # Render the home page again, passing back the prediction and the original form data
    return render_template('index.html', prediction_text=f'Predicted Car Price: {output}', form_data=input_features)

# This line is essential to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)