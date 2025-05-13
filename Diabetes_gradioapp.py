import gradio as gr
import pickle
import numpy as np
import pandas as pd

# Load saved model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))


# Prediction function
def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
):

    input_data = np.array(
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
        ]
    ).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        return "The person IS diabetic"
    else:
        return "The person is NOT diabetic"


# Gradio interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs="text",
    title="Diabetes Prediction App",
    description="Enter the health parameters to predict if the person is diabetic.",
)

iface.launch()
