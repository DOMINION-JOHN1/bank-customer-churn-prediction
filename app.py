
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image

# Load the Random Forest Classifier model
with open('rfc_model.pkl', 'rb') as model_file:
    rfc_model = pickle.load(model_file)

# Define dictionaries to map user-friendly text options to labels
geography_mapping = {'Spain': 0, 'France': 1, 'Germany': 2}
gender_mapping = {'Female': 0, 'Male': 1}
yes_no_mapping = {'No': 0, 'Yes': 1}

# Streamlit app header
st.title('Customer Churn Prediction')
image = Image.open("bank.jpg")
st.image(image, use_column_width=True)

# Sidebar with user input
st.sidebar.header('User Input')

# Create input fields for the new features
credit_score = st.sidebar.slider('Credit Score', min_value=300, max_value=850, value=500)
geography = st.sidebar.selectbox('Country', options=['Spain', 'France', 'Germany'])
gender = st.sidebar.selectbox('Gender', options=['Female', 'Male'])
age = st.sidebar.slider('Age', min_value=18, max_value=100, value=35)
tenure = st.sidebar.slider('Tenure', min_value=0, max_value=10, value=5)
balance = st.sidebar.slider('Account balance', min_value=0, max_value=250000, value=50000)
num_of_products = st.sidebar.slider('Num Of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.sidebar.selectbox('Uses Credit card', options=['No', 'Yes'])
is_active_member = st.sidebar.selectbox('Active member?', options=['No', 'Yes'])
estimated_salary = st.sidebar.slider('Annual Salary', min_value=0, max_value=200000, value=50000)

# Convert user-friendly text options to labels
geography = geography_mapping[geography]
gender = gender_mapping[gender]
has_cr_card = yes_no_mapping[has_cr_card]
is_active_member = yes_no_mapping[is_active_member]

# Create a button to make predictions
if st.sidebar.button('Predict'):
    # Preprocess the user input (standardization)
    user_input = [credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform([user_input])

    # Make predictions using the Random Forest Classifier model
    prediction = rfc_model.predict(user_input_scaled)

    # Display the prediction result
    if prediction[0][0] >= 0.2:   # Assuming 1 represents churn
        st.sidebar.success('This customer is at a higher risk of leaving or discontinuing their services.')
    else:
        st.sidebar.error('This customer is likely to continue using their services.')
