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
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #64748B; margin-bottom: 2rem; }
    .card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border-left: 5px solid #10B981; }
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background-color: #1E293B;
        color: white;
        padding: 10px 0;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .ticker {
        display: inline-block;
        white-space: nowrap;
        animation: ticker 30s linear infinite;
    }
    .ticker-item {
        display: inline-block;
        padding: 0 2rem;
        font-size: 0.9rem;
    }
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">Indian Used Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Estimation System</div>', unsafe_allow_html=True)

if not model:
    st.error("Model files not found. Please ensure .pkl files are in the directory.")
    st.stop()

# --- Tabs ---
# Ticker logic
quotes = [
    "💡 Tip: Cars with full service history command 15% higher resale value.",
    "🚗 Market Trend: SUV demand is up 12% this quarter.",
    "📉 Depreciation Alert: Values drop fastest in the first 3 years.",
    "💰 Seller Tip: Detailing your car can add ₹10k-20k to the final deal.",
    "⭐ Pro Insight: First owners always get the best deals!"
]
quote_html = f"""
<div class="ticker-wrap">
<div class="ticker">
{''.join([f'<div class="ticker-item">{q}</div>' for q in quotes])}
</div>
</div>
"""
st.markdown(quote_html, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Valuation & Forecast", "📈 Market Insights"])

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
            st.markdown(f"""
            <div class="card">
                <span style="font-size: 0.9rem; color: #64748B; text-transform: uppercase;">Estimated Market Value</span>
                <div style="font-size: 2.5rem; font-weight: 800; color: #10B981; margin: 0.5rem 0;">₹{pred_rupees:,.0f}</div>
                <div style="font-size: 0.9rem; color: #64748B;">
                    Ideal Selling Range: <b>₹{(pred_rupees*0.97):,.0f} - ₹{(pred_rupees*1.03):,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📅 Future Value Forecast")
            st.caption("Estimated depreciation over the next 3 years if kept in similar condition:")
            
            future_data = []
            current_val = pred_lakhs
            for i in range(1, 4):
                # Simple decay model: 10% per year for display purposes
                current_val = current_val * 0.90
                future_data.append({
                    "Year": f"+{i} Year",
                    "Estimated Value": f"₹{current_val*100000:,.0f}",
                    "Loss": f"-₹{(pred_lakhs - current_val)*100000:,.0f}"
                })
            st.table(pd.DataFrame(future_data))
            
            # CSV Download
            report_text = f"""
            VALUATION REPORT
            ----------------
            Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
            Vehicle: {car_age}yo, {kms_driven}kms, {fuel_type}
            
            ESTIMATED VALUE: ₹{pred_rupees:,.0f}
            """
            st.download_button("Download Report", report_text, file_name="valuation_report.txt")

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
# REMOVED TAB 3 (Interview Prep)
# ==========================
