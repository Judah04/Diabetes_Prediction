import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))


# Prediction function
def predict_diabetes(input_data, model, scaler):
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    std_data = scaler.transform(input_df)
    prediction = model.predict(std_data)
    return (
        "The person IS diabetic" if prediction[0] == 1 else "The person is NOT diabetic"
    )


# Streamlit App
st.title(":red[Diabetes Prediction App]")
st.write(
    """
This app predicts whether a person is diabetic based on their health parameters.
"""
)
st.divider()

st.write(
    """
Fill in patient's details below.
"""
)
# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 200, 0)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 0)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 0)
insulin = st.number_input("Insulin", 0, 1000, 0)
bmi = st.number_input("BMI", 0.0, 70.0, 0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.0)
age = st.number_input("Age", 0, 120, 0)

# Predict button
if st.button("Predict"):
    input_data = (
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    )
    prediction = predict_diabetes(input_data, model, scaler)
    st.success(prediction)
