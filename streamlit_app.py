import streamlit as st
import pandas as pd
import joblib
import time

# Set page configuration
st.set_config_page(
    page_title="Indian Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load the Model and Columns ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('random_forest_model_indian.pkl')
        model_columns = joblib.load('model_columns_indian.pkl')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'random_forest_model_indian.pkl' and 'model_columns_indian.pkl' are in the directory.")
        return None, None

model, data_columns = load_data()

if model and data_columns:
    # --- 2. App Layout ---
    
    # Header Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚗 Indian Used Car Price Predictor")
        st.markdown("### Accurate Resale Value Estimation AI")
        st.markdown("Unlock the hidden value of your vehicle with our advanced Machine Learning model. Perfect for buyers and sellers looking for a fair deal.")
    
    with col2:
        # Placeholder for an image or logo if available, otherwise just spacing
        st.write("") 
        st.info("💡 **Pro Tip:** Lower kms and fewer owners significantly boost resale value.")

    st.markdown("---")

    # --- 3. Sidebar for Inputs ---
    st.sidebar.header("📝 Enter Car Details")
    
    with st.sidebar.form("prediction_form"):
        st.subheader("Vehicle Specs")
        
        present_price = st.number_input(
            "Current Showroom Price (₹)", 
            min_value=0, 
            value=850000, 
            step=10000, 
            help="The price of the car if bought new today in Rupees."
        )
        
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            kms_driven = st.number_input("Kms Driven", min_value=0, value=45000, step=1000)
        with col_sb2:
            car_age = st.number_input("Car Age (Years)", min_value=0, value=5, step=1)
            
        owner = st.selectbox(
            "Previous Owners", 
            [0, 1, 2, 3], 
            format_func=lambda x: "First Owner" if x == 0 else (f"{x+1}nd Owner" if x==1 else f"{x+1}rd/th Owner")
        )
        
        st.subheader("Configuration")
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        
        submit_btn = st.form_submit_button("🚀 Predict Price")

    # --- 4. Main Content Area (Results & Insights) ---
    
    if submit_btn:
        with st.spinner("Analyzing market trends..."):
            time.sleep(1) # Added for effect
            
            # Prediction Logic
            prediction_df = pd.DataFrame(columns=data_columns)
            prediction_df.loc[0] = 0

            prediction_df['Present_Price'] = present_price / 100000.0 # Convert to Lakhs
            prediction_df['Kms_Driven'] = kms_driven
            prediction_df['Owner'] = owner
            prediction_df['Car_Age'] = car_age

            input_dict = {
                'Fuel_Type': fuel_type,
                'Seller_Type': seller_type,
                'Transmission': transmission
            }

            for feature, value in input_dict.items():
                column_name = f"{feature}_{value}"
                if column_name in data_columns:
                    prediction_df[column_name] = 1
            
            prediction_df = prediction_df[data_columns]
            
            prediction_lakhs = model.predict(prediction_df)[0]
            prediction_rupees = prediction_lakhs * 100000
            
            # Display Result
            st.markdown("### 📊 Valuation Report")
            
            res_col1, res_col2 = st.columns([2, 3])
            
            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#555;">Estimated Value</h3>
                    <h1 style="margin:0; color:#2ecc71; font-size: 3em;">₹{prediction_rupees:,.0f}</h1>
                    <p style="margin-top:10px; font-size:0.9em; color:#888;">Based on current market data</p>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                # Insights/Factors
                st.caption("Why this price?")
                factors = {
                    "Depreciation": f"-{(car_age * 10):.1f}% estimated due to age",
                    "Brand Value": "Calculated from showroom price",
                    "Condition": "Inferred from Kms driven"
                }
                
                for key, val in factors.items():
                    st.text(f"• {key}: {val}")
                    
                if prediction_lakhs < 0:
                    st.warning("⚠️ The estimated value is negative. This might be due to unrealistic inputs (e.g., extremely high age for a cheap car). Please check your inputs.")
                elif prediction_lakhs > 50:
                    st.balloons()
                    st.success("🌟 This is a luxury vehicle with high resale potential!")
                else:
                    st.info("✔ A solid deal for this category.")


    # --- 5. Footer/About ---
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit & Scikit-Learn")

else:
    st.warning("Application is waiting for model files to be loaded.")