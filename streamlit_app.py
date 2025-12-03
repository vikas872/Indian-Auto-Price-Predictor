import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Model and Columns ---
# We use @st.cache_resource to load these only once, making the app faster
@st.cache_resource
def load_data():
    model = joblib.load('random_forest_model_indian.pkl')
    model_columns = joblib.load('model_columns_indian.pkl')
    return model, model_columns

model, data_columns = load_data()

# --- 2. App Title and Description ---
st.title("🇮🇳 Indian Used Car Price Predictor")
st.write("Enter the car details below to get an estimated resale price.")

# --- 3. Sidebar for Inputs ---
st.sidebar.header("Car Details")

# Numerical Inputs
present_price = st.sidebar.number_input("Current Showroom Price (₹)", min_value=0, value=850000, step=10000, help="The price of the car if bought new today.")
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, value=45000, step=1000)
car_age = st.sidebar.number_input("Car Age (Years)", min_value=0, value=5, step=1)
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 3], format_func=lambda x: "First Owner" if x == 0 else x)

# Categorical Inputs
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# --- 4. Prediction Logic ---
if st.sidebar.button("Predict Price"):
    # Create a dataframe with a single row of zeros
    prediction_df = pd.DataFrame(columns=data_columns)
    prediction_df.loc[0] = 0

    # Process Inputs (Same logic as your Flask app)
    prediction_df['Present_Price'] = present_price / 100000.0 # Convert to Lakhs
    prediction_df['Kms_Driven'] = kms_driven
    prediction_df['Owner'] = owner
    prediction_df['Car_Age'] = car_age

    # One-Hot Encoding Logic
    input_dict = {
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission
    }

    for feature, value in input_dict.items():
        column_name = f"{feature}_{value}"
        if column_name in data_columns:
            prediction_df[column_name] = 1
    
    # Ensure column order matches
    prediction_df = prediction_df[data_columns]

    # Make Prediction
    prediction_lakhs = model.predict(prediction_df)[0]
    prediction_rupees = prediction_lakhs * 100000

    # --- 5. Display Result ---
    st.subheader("Prediction Result")
    
    # Display a nice metric card
    st.metric(label="Estimated Resale Value", value=f"₹{prediction_rupees:,.2f}")
    
    # Optional: Show a progress bar or fun element
    if prediction_lakhs > 5:
        st.success("This car holds significant value!")
    else:
        st.info("This is a budget-friendly option.")

else:
    st.info("Adjust the details in the sidebar and click 'Predict Price' to see the result.")