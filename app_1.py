import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the SVM model
with open('svc_model.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)

# Streamlit app header
st.title('Customer Churn Prediction')

# Sidebar with user input
st.sidebar.header('User Input')

# Create input fields for the new features
features = {
    'BalanceSalaryRatio': st.sidebar.slider('BalanceSalaryRatio', min_value=0.0, max_value=10.0, value=5.0),
    'CreditScoreAgeRatio': st.sidebar.slider('CreditScoreAgeRatio', min_value=0.0, max_value=10.0, value=5.0),
    'TenureAgeRatio': st.sidebar.slider('TenureAgeRatio', min_value=0.0, max_value=10.0, value=5.0),
    'CreditScoreGivenSalary': st.sidebar.slider('CreditScoreGivenSalary', min_value=0.0, max_value=10.0, value=5.0),
    'NumOfProductsGivenAge': st.sidebar.slider('NumOfProductsGivenAge', min_value=0.0, max_value=10.0, value=5.0),
    'BalanceGivenAge': st.sidebar.slider('BalanceGivenAge', min_value=0.0, max_value=10.0, value=5.0),
    'BalanceGivenCreditScore': st.sidebar.slider('BalanceGivenCreditScore', min_value=0.0, max_value=10.0, value=5.0),
    'TenureGivenAge': st.sidebar.slider('TenureGivenAge', min_value=0.0, max_value=10.0, value=5.0)
}

# Create a button to make predictions
if st.sidebar.button('Predict'):
    # Convert the user input to a NumPy array
    user_input = np.array(list(features.values())).reshape(1, -1)

    # Preprocess the user input (standardization)
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    # Make predictions using the SVM model
    prediction = svc_model.predict(user_input_scaled)

    # Display the prediction result
    if prediction[0] == 1:   # Assuming 1 represents churn
        st.sidebar.success('Prediction: Churn (C)')
    else:
        st.sidebar.error('Prediction: Non-Churn (NC)')

# Run the Streamlit app
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.run()

    # Preprocess the user input (standardization)
    user_input = np.array([features])
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    # Make predictions using the SVM model
    prediction = knn_model.predict(user_input_scaled)

    # Display the prediction result
    if prediction[0] == '1':   # Assuming '1' represents churn
        st.sidebar.success('Prediction: Churn (C)')
    else:
        st.sidebar.error('Prediction: NON Churn (NC)')


# Run the Streamlit app
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.run()
