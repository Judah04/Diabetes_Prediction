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
# This code creates a Gradio app for diabetes prediction using a pre-trained model and scaler.
# The app takes user inputs for various health parameters and outputs whether the person is diabetic or not.
# The app is launched using the `iface.launch()` method.
# The app is user-friendly and allows for easy interaction with the model.
# The input fields are labeled for clarity, and the output is displayed as text.
# The app can be run in a web browser, making it accessible to users without requiring any installation.
# The Gradio library simplifies the process of creating web interfaces for machine learning models.
# The app is a great example of how to deploy machine learning models using Gradio.
# The app is a simple yet effective tool for diabetes prediction, making it accessible to users without technical expertise.
# The app is designed to be run in a Gradio environment, providing a web-based interface for users.
# The app is a great example of how to deploy machine learning models using Gradio.
# The app is a simple yet effective tool for diabetes prediction, making it accessible to users without technical expertise.
# The app is designed to be run in a Gradio environment, providing a web-based interface for users.
# The app is a great example of how to deploy machine learning models using Gradio.
# The app is a simple yet effective tool for diabetes prediction, making it accessible to users without technical expertise.
