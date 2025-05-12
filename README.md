# 🩺 Diabetes Prediction App using SVM

This project uses a Support Vector Machine (SVM) classifier to predict whether a person is diabetic based on several medical parameters. It is built using scikit-learn for machine learning and includes both a **Gradio** and **Streamlit** interface for interaction.

## 🚀 Features

- Based on the Pima Indians Diabetes Dataset.
- Preprocessing using StandardScaler.
- Classification using Support Vector Machine (SVM).
- Streamlit web app for interactive prediction.
- Alternatively, Gradio interface for interactive prediction.

## 📊 Dataset

The dataset used is from the **Pima Indians Diabetes Database**, which includes medical data for female patients of at least 21 years old of Pima Indian heritage.

Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target:
- Outcome (`0` means _Not diabetic_ and `1` means _Diabetic_)

## ⚙️ How to Use

### 🛠️ Requirements

- Install all dependencies:
```bash
pip install -r requirements.txt
```

### 🔧 In Jupyter Notebook

1. Train the model by running all cells in the notebook.
2. Save the model and scaler using `pickle`.
3. Test predictions with new input data.

### 🌐 As a Web App (Streamlit or Gradio)

1. Run the Streamlit app:

```bash
streamlit run diabetes_streamlitapp.py
```

2. Run the Gradio interface:
```bash
python diabetes_gradioapp.py
```