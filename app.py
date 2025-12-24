import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- 1. Page Config & Theme ---
st.set_page_config(page_title="Churn AI Expert", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- 2. Preprocessing Function (FULL LOGIC) ---
def preprocess_data(df_input):
    df = df_input.copy()
    
    # Numeric Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['tenure'] = df['tenure'].astype(float)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(float)
    
    # Mappings
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    for col in service_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0}).fillna(0).astype(float)
    
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0}).fillna(0).astype(float)
    
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['Contract_encoded'] = df['Contract'].map(contract_map).fillna(0).astype(float)
    
    # Engineering
    df['Monthly_x_Tenure'] = (df['MonthlyCharges'] * df['tenure']).astype(float)
    df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(float)
    df['InternetService_No'] = (df['InternetService'] == 'No').astype(float)
    
    df['tenure_bin_13-24'] = ((df['tenure'] > 12) & (df['tenure'] <= 24)).astype(float)
    df['tenure_bin_25-48'] = ((df['tenure'] > 24) & (df['tenure'] <= 48)).astype(float)
    df['tenure_bin_49+'] = (df['tenure'] > 48).astype(float)
    
    df['MonthlyCharges_bin_Medium'] = ((df['MonthlyCharges'] > 35) & (df['MonthlyCharges'] <= 70)).astype(float)
    df['MonthlyCharges_bin_High'] = (df['MonthlyCharges'] > 70).astype(float)
    
    df['PaymentMethod_combined_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(float)
    df['PaymentMethod_combined_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(float)
    df['PaymentMethod_combined_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(float)
    df['PaymentMethod_Automatic'] = df['PaymentMethod'].apply(lambda x: 1 if 'automatic' in str(x).lower() else 0).astype(float)

    # Scaling
    numeric_cols = ['Monthly_x_Tenure', 'MonthlyCharges', 'TotalCharges', 'tenure']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    final_cols = [
        'SeniorCitizen', 'tenure', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'Contract_encoded', 
        'InternetService_Fiber optic', 'InternetService_No', 'Monthly_x_Tenure', 
        'tenure_bin_13-24', 'tenure_bin_25-48', 'tenure_bin_49+', 
        'MonthlyCharges_bin_Medium', 'MonthlyCharges_bin_High', 
        'PaymentMethod_combined_Credit card (automatic)', 'PaymentMethod_combined_Electronic check', 
        'PaymentMethod_combined_Mailed check', 'PaymentMethod_Automatic'
    ]
    
    return df[final_cols].values.astype(np.float32)

# --- 3. UI Layout ---
st.title("🚀 Customer Churn Intelligence Dashboard")
st.markdown("Identify high-risk customers and protect your revenue.")

uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    
    if st.button('🔍 Run Deep Analysis'):
        try:
            X_array = preprocess_data(raw_df)
            preds = model.predict(X_array)
            probs = model.predict_proba(X_array)[:, 1]
            
            raw_df['Churn_Risk'] = ["🔴 High Risk" if p == 1 else "🟢 Safe" for p in preds]
            raw_df['Risk_Score'] = (probs * 100).round(1)
            
            # --- KPI Metrics ---
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Customers", len(raw_df))
            m2.metric("At-Risk", (preds == 1).sum())
            m3.metric("Avg Risk Score", f"{raw_df['Risk_Score'].mean():.1f}%")

            # --- Visuals ---
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(raw_df, names='Churn_Risk', title="Overall Risk Status",
                                 color='Churn_Risk', color_discrete_map={'🔴 High Risk':'#ff4b4b', '🟢 Safe':'#00d488'})
                st.plotly_chart(fig_pie)
            with col2:
                fig_hist = px.histogram(raw_df, x="Risk_Score", title="Risk Score Distribution", nbins=20)
                st.plotly_chart(fig_hist)

            st.subheader("📋 Priority Report")
            st.dataframe(raw_df[['customerID', 'Churn_Risk', 'Risk_Score', 'Contract', 'MonthlyCharges']].sort_values(by='Risk_Score', ascending=False))
            
        except Exception as e:
            st.error(f"Error: {e}")