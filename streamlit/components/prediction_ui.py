import streamlit as st


def render_sidebar():

    st.sidebar.header("Customer Information")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])

    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])

    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])

    internet_service = st.sidebar.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.sidebar.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    monthly_charges = st.sidebar.slider(
        "Monthly Charges",
        10.0,
        150.0,
        70.0
    )

    total_charges = st.sidebar.slider(
        "Total Charges",
        0.0,
        10000.0,
        2000.0
    )

    return {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": "No",
        "internet_service": internet_service,
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": contract,
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }
