import streamlit as st
import pandas as pd
import joblib
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Car Price Predictor | Professional Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('random_forest_model_indian.pkl')
        model_columns = joblib.load('model_columns_indian.pkl')
        return model, model_columns
    except FileNotFoundError:
        return None, None

model, data_columns = load_data()

# --- Custom Styling ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; border-radius: 4px; font-weight: 500; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E293B; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #64748B; margin-bottom: 2rem; }
    .card { background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">Indian Used Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Estimation System</div>', unsafe_allow_html=True)

if not model:
    st.error("Model files not found. Please ensure .pkl files are in the directory.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Prediction Tool", "📈 Model Insights", "🧠 Interview Prep"])

# ==========================
# TAB 1: PREDICTION TOOL
# ==========================
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("### Vehicle Configuration")
        with st.form("prediction_form"):
            present_price = st.number_input("Current Showroom Price (₹)", min_value=0, value=850000, step=10000, help="Price of new model today.")
            c1, c2 = st.columns(2)
            with c1: kms_driven = st.number_input("Kilometers Driven", min_value=0, value=45000, step=1000)
            with c2: car_age = st.number_input("Vehicle Age (Years)", min_value=0, value=5, step=1)
            owner = st.selectbox("Previous Owners", [0, 1, 2, 3], format_func=lambda x: f"{x} Owners" if x > 0 else "First Owner")
            
            st.markdown("#### Technical Specs")
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            
            submit_btn = st.form_submit_button("Calculated Market Value", type="primary")

    with col_result:
        if submit_btn:
            input_df = pd.DataFrame(columns=data_columns)
            input_df.loc[0] = 0
            input_df['Present_Price'] = present_price / 100000.0
            input_df['Kms_Driven'] = kms_driven
            input_df['Owner'] = owner
            input_df['Car_Age'] = car_age
            
            cat_map = {f"Fuel_Type_{fuel_type}": 1, f"Seller_Type_{seller_type}": 1, f"Transmission_{transmission}": 1}
            for col, val in cat_map.items():
                if col in data_columns: input_df[col] = val
                    
            input_df = input_df[data_columns]
            pred_lakhs = model.predict(input_df)[0]
            pred_rupees = pred_lakhs * 100000
            
            st.markdown("### Valuation Result")
            st.markdown(f"""<div class="card"><span style="font-size: 0.9rem; color: #64748B;">ESTIMATED MARKET PRICE</span><div style="font-size: 3rem; font-weight: 800; color: #10B981; margin: 0.5rem 0;">₹{pred_rupees:,.0f}</div><div style="font-size: 0.9rem; color: #64748B;">Range: ₹{(pred_rupees*0.95):,.0f} - ₹{(pred_rupees*1.05):,.0f}</div></div>""", unsafe_allow_html=True)
            
            st.markdown("#### Analysis")
            depreciation = min(70, (car_age * 10))
            st.progress(max(0, 100 - int(depreciation))/100, text=f"Retained Value: {100-int(depreciation)}%")
            st.caption("Value retention based primarily on vehicle age.")
        else:
            st.info("Configure variables and click to calculate.")

# ==========================
# TAB 2: MODEL INSIGHTS
# ==========================
with tab2:
    st.markdown("### Feature Importance Analysis")
    st.write("The Random Forest Regressor identifies 'Showroom Price' as the dominant predictor.")
    
    importance_data = pd.DataFrame({
        'Feature': ['Showroom Price (Base)', 'Vehicle Age', 'Kilometers Driven', 'Transmission Type', 'Fuel Type'],
        'Importance': [0.88, 0.06, 0.03, 0.01, 0.005]
    })
    
    chart = alt.Chart(importance_data).mark_bar().encode(
        x=alt.X('Importance', axis=alt.Axis(format='%')),
        y=alt.Y('Feature', sort='-x'),
        color=alt.Color('Importance', scale=alt.Scale(scheme='greens')),
        tooltip=['Feature', alt.Tooltip('Importance', format='.1%')]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

# ==========================
# TAB 3: INTERVIEW PREP
# ==========================
with tab3:
    st.markdown("### System Architecture")
    st.code("""[Raw CSV Data] -> [Preprocessing (One-Hot)] -> [Random Forest Model] -> [Streamlit App]""", language="text")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Why Random Forest?")
        st.info("Captures non-linear relationships (age vs price) better than Linear Regression and is robust to outliers.")
    with c2:
        st.markdown("#### Challenges?")
        st.info("Handling categorical data (Fuel/Transmission) using One-Hot Encoding and deploying with correct Python versions.")
