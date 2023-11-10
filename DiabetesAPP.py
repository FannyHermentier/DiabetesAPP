import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load the saved model
model_filename = 'diabetes_classifier.h5'
model = tf.keras.models.load_model(model_filename)

# Load the saved scaler
scaler_filename = 'scaler.pkl'
scaler = pickle.load(open(scaler_filename, 'rb'))

# Function to predict diabetes
def predict_diabetes(Pregnancies, Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = np.array([Pregnancies, Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age])
    input_data = scaler.transform(input_data.reshape(1, -1))  # Reshape the input data and scale it
    prediction = model.predict(input_data)
    return prediction

# Create the Streamlit web app
st.set_page_config(layout="wide")
st.title("Diabetes Prediction App")
st.markdown("Welcome to the Diabetes Prediction App. Enter the following information to predict diabetes.")
st.sidebar.image("diab.png", use_column_width=True, caption="Picture - Source: https://www.arkanalabs.com/diabetes-mellitus/")
st.sidebar.image("diabetes_symptoms.jpg", caption="Diabetes Symptoms - Source: https://my.clevelandclinic.org/health/diseases/7104-diabetes")

# Input fields for user data
Pregnancies = st.slider('Number of Pregnancies', min_value=0, max_value=20, value=4, step=1)
Glucose = st.slider('Glucose level', min_value=0, max_value=199, value=60, step=1)
Insulin = st.slider('Insulin level (mu U/ml)', min_value=0, max_value=1000, value=400, step=1)
st.markdown("**BMI (Body Mass Index)** is a measure of body fat based on height and weight.")
BMI = st.slider('BMI value', min_value=0.0, max_value=70.0, value=33.3, step=0.01)
DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', min_value=0.000, max_value=3.0, value=0.045, step=0.001)
Age = st.slider('Age', min_value=10, max_value=100, value=21, step=1)

# When the 'Predict' button is clicked, make the prediction and display the result
if st.button("Predict"):
    prediction = predict_diabetes(Pregnancies, Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age)
    if prediction[0][0] > 0.5:
        st.error("Based on the given data, the person may have diabetes.")
    else:
        st.success("Based on the given data, the person may not have diabetes.")
        st.balloons()

# Add a footer with your name, affiliation, and a link to the source code
st.markdown(
    """
    ---
    Created by Fanny HERMENTIER *Student at IE University*
    Source code available on [GitHub](https://github.com/FannyHermentier)
    """,
    unsafe_allow_html=True,
)
