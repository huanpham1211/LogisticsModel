import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler

# Function to load the data from an uploaded Excel file
def load_data(file):
    if file is not None:
        return pd.read_excel(file)
    return None

# Prepare the data for prediction
def prepare_data(data):
    categorical_vars = ['GIOI', 'PRIOR.TIA.STROKE', 'THA', 'DTD', 'RN', 'KHOIPHAT', 'TUYENTRUOC', 'RTPA', 'TICIG', 'sICH.SITS', 'ARTERY.DSA.FINAL', 'TOAST']
    continuous_vars = ['TUOI', 'NIHSSNVIEN', 'ONSET-REP', 'HIRnew', 'ADC<620', 'MM.VOL', 'ASPECT/MRI']

    # One-hot encode the categorical variables
    data_encoded = pd.get_dummies(data[categorical_vars], drop_first=True)

    # Impute missing values in continuous variables
    data[continuous_vars] = data[continuous_vars].fillna(data[continuous_vars].mean())

    # Combine with continuous variables
    X = pd.concat([data_encoded, data[continuous_vars]], axis=1)

    return X, continuous_vars

# Simulate a trained logistic regression model
def train_model(data):
    X, continuous_vars = prepare_data(data)
    y = data['MRS90DAY'].apply(lambda x: 1 if x < 3 else 0)  # 1 = Good (0-2), 0 = Bad (3-6)
    
    # Fit the RobustScaler on training data
    scaler = RobustScaler()
    X[continuous_vars] = scaler.fit_transform(X[continuous_vars])

    # Fit the logistic regression model
    logit_model = sm.Logit(y, X).fit()

    # Return both the model, fitted scaler, and continuous_vars
    return logit_model, scaler, continuous_vars

# Streamlit App
st.title("MRS Prediction Tool")
st.write("Upload your Excel file and enter patient data to predict the probability of a good MRS score (0-2).")

# Upload Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        # Train the model with the uploaded data
        logit_model, scaler, continuous_vars = train_model(data)

        # Input fields for user inputs
        tuoi = st.number_input('Age (TUOI)', min_value=0, max_value=120, value=65)
        nihssnvien = st.number_input('NIHSSNVIEN', min_value=0, max_value=42, value=10)
        onset_rep = st.number_input('ONSET-REP', min_value=0.0, value=5.0)
        hirnew = st.number_input('HIRnew', min_value=0.0, value=0.5)
        adc_620 = st.number_input('ADC<620', min_value=0.0, value=50.0)
        mm_vol = st.number_input('MM.VOL', min_value=0.0, value=100.0)
        aspect_mri = st.number_input('ASPECT/MRI', min_value=0.0, value=7.0)

        # Categorical inputs as select boxes
        gioi = st.selectbox('Sex (1=Male, 0=Female)', [1, 0])
        prior_tia_stroke = st.selectbox('PRIOR.TIA.STROKE (1=Yes, 0=No)', [1, 0])
        tha = st.selectbox('THA (1=Yes, 0=No)', [1, 0])
        ticig = st.selectbox('TICIG (0=Reperfusion, 1=NoReperfusion)', [0, 1])
        artery_dsa_final = st.selectbox('ARTERY.DSA.FINAL', [1, 2, 3, 4])
        toast = st.selectbox('TOAST Classification', [1, 2, 3, 4])

        # Convert user inputs into a DataFrame
        input_data = pd.DataFrame({
            'TUOI': [tuoi],
            'NIHSSNVIEN': [nihssnvien],
            'ONSET-REP': [onset_rep],
            'HIRnew': [hirnew],
            'ADC<620': [adc_620],
            'MM.VOL': [mm_vol],
            'ASPECT/MRI': [aspect_mri],
            'GIOI': [gioi],
            'PRIOR.TIA.STROKE': [prior_tia_stroke],
            'THA': [tha],
            'TICIG': [ticig],
            'ARTERY.DSA.FINAL': [artery_dsa_final],
            'TOAST': [toast]
        })

        # Standardize continuous variables using the previously fitted scaler
        input_data[continuous_vars] = scaler.transform(input_data[continuous_vars])

        # Predict probability
        predicted_prob = logit_model.predict(input_data)[0]

        st.write(f"Predicted Probability of a Good MRS Outcome (0-2): {predicted_prob:.2f}")
else:
    st.write("Please upload an Excel file to proceed.")
