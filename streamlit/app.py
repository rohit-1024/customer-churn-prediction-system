import streamlit as st

from utils.model_loader import load_model
from utils.preprocessing import preprocess_input
from utils.prediction import predict
from components.prediction_ui import render_sidebar


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("Customer Churn Prediction System")

st.write(
"""
This application predicts whether a telecom customer
is likely to churn using a trained machine learning model.
"""
)

model, training_columns = load_model()

user_input = render_sidebar()

if st.button("Predict"):

    input_df = preprocess_input(user_input, training_columns)

    prediction, probability = predict(model, input_df)

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("Customer likely to churn")

    else:

        st.success("Customer likely to stay")

    st.metric("Churn Probability", f"{probability:.2%}")
