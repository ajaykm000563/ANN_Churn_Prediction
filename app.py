import numpy as np
import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer



# Loading the model and preprocessor objects
model = tf.keras.models.load_model('churn_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))


# Streamlit app
st.title("Customer Churn Prediction")


# User input
st.header("Enter Customer Details")
CreditScore = st.number_input("Credit Score", min_value=300, max_value=1000, value=600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0,1]) 
is_active_member = st.selectbox("Is Active Member", [0,1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Preprocess user input
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Apply preprocessor to the input data
input_processed = preprocessor.transform(input_data)

# Apply scaler to the processed input data
input_scaled = scaler.transform(input_processed)

# Make prediction
prediction = model.predict(input_scaled)
if prediction[0][0] > 0.5:
    st.error("The customer is likely to churn with a probability of {:.2f}%".format(prediction[0][0] * 100))
else:
    st.success("The customer is unlikely to churn with a probability of {:.2f}%".format(prediction[0][0] * 100))