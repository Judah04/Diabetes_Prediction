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
t.title(":red[Diabetes Prediction App]")
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

# streamlit run your_script_name.py
# To run the app, save this code in a file (e.g., app.py) and run the following command in your terminal:
# streamlit run app.py
# Note: Ensure you have the required libraries installed:
# pip install streamlit pandas numpy scikit-learn
# Note: The model and scaler files (diabetes_model.pkl and scaler.pkl) should be in the same directory as this script.
# This code is a simple Streamlit application for predicting diabetes based on user input.
# The model and scaler should be pre-trained and saved as pickle files.
# The app takes user inputs for various health parameters and uses the loaded model to predict diabetes.
# The prediction result is displayed to the user.
# The app is user-friendly and allows for easy interaction.
# The code includes necessary imports, model loading, prediction function, and Streamlit UI components.
# The app is designed to be run in a Streamlit environment, providing a web-based interface for users.
# The app is a great example of how to deploy machine learning models using Streamlit.
# The app is a simple yet effective tool for diabetes prediction, making it accessible to users without technical expertise.
# The app can be further enhanced with additional features, such as data visualization and user feedback.
# The app can be used in healthcare settings to assist in diabetes screening and awareness.
# The app can be integrated with other health applications for a comprehensive health monitoring solution.
# The app can be extended to include more health parameters and advanced prediction algorithms.
# The app can be used for educational purposes to demonstrate the use of machine learning in healthcare.
# The app can be shared with healthcare professionals for feedback and improvement.
# The app can be deployed on cloud platforms for wider accessibility and scalability.
# The app can be customized with different themes and layouts to enhance user experience.
# The app can be tested with various input scenarios to ensure accuracy and reliability.
# The app can be used to raise awareness about diabetes and promote healthy living.
# The app can be a valuable resource for individuals at risk of diabetes to monitor their health.
