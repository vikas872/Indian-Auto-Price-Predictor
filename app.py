import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('random_forest_model_indian.pkl')

data_columns = joblib.load('model_columns_indian.pkl')



@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives user input, prepares it for the model, and returns the prediction."""
    
    # Get all the input values from the form as a dictionary
    input_features = request.form.to_dict()

    # Create a dataframe with a single row of zeros, using the loaded column names
    prediction_df = pd.DataFrame(columns=data_columns)
    prediction_df.loc[0] = 0

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
    prediction_in_lakhs = model.predict(prediction_df)
    
    # Convert the prediction from Lakhs to Rupees for display
    prediction_in_rupees = prediction_in_lakhs[0] * 100000
    
    # Format the output with the Rupee symbol and commas
    output = f"₹{prediction_in_rupees:,.2f}"

    return render_template('index.html', prediction_text=f'Predicted Car Price: {output}', form_data=input_features)

if __name__ == "__main__":
    pass
