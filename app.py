import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

# Create input fields in a 3x3 grid
col1, col2, col3 = st.columns(3)

# First row
with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
with col2:
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
with col3:
    age = st.slider('Age', 18, 92)

# Second row
with col1:
    balance = st.number_input('Balance', min_value=0.0)
with col2:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
with col3:
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

# Third row
with col1:
    tenure = st.slider('Tenure', 0, 10)
with col2:
    num_of_products = st.slider('Number of Products', 1, 4)
with col3:
    has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])

# Length slider
length = st.slider('Select Length', min_value=1, max_value=100, value=50)

# Button to trigger prediction
if st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [0],  # Placeholder; adjust based on your use case
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Show loading spinner while predicting
    with st.spinner('Predicting...'):
        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

    # Display results after prediction
    st.success(f'Churn Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')
