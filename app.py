import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("car_loan_model.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Car Loan Eligibility",
    page_icon="🚗",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🚗 Car Loan Eligibility Prediction")
st.write("Enter the applicant details to predict loan eligibility.")

st.divider()


# ==========================================
# INPUTS
# ==========================================

st.subheader("👤 Applicant Details")

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=20,
    value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)


st.subheader("💰 Financial Details")

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000,
    step=10000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000,
    step=10000
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1,
    max_value=50,
    value=10
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)


st.subheader("🏠 Asset Details")

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=1000000,
    step=10000
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=500000,
    step=10000
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=500000,
    step=10000
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=500000,
    step=10000
)


st.divider()


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Check Loan Eligibility", use_container_width=True):

    # Convert frontend values to the same format
    # used during model training

    education_value = 0 if education == "Graduate" else 1

    self_employed_value = 1 if self_employed == "Yes" else 0


    # Create input dataframe
    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "education": [education_value],
        "self_employed": [self_employed_value],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]
    })


    # Prediction
    prediction = model.predict(input_data)[0]


    # ==========================================
    # RESULT
    # ==========================================

    st.subheader("📋 Prediction Result")

    if prediction == 0:
        st.error("❌ Loan Rejected")
        st.write("The applicant may not be eligible for the loan.")
    else:
        st.success("✅ Loan Approved")
        st.write("The applicant may be eligible for the loan.")