import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, RobustScaler

# Load the trained logistic regression model (you can load the model if it is saved, or retrain it within the script)
# Example of a trained model being loaded if saved:
# import pickle
# with open('trained_logit_model.pkl', 'rb') as f:
#     logit_model = pickle.load(f)

# Simulating logistic regression model fitting based on previous instructions:
def load_data():
    file_path = 'v1 NCS_FINAL.xlsx'  # Path to the data file
    return pd.read_excel(file_path)

def prepare_data(data):
    categorical_vars = ['GIOI', 'PRIOR.TIA.STROKE', 'THA', 'DTD', 'RN', 'KHOIPHAT', 'TUYENTRUOC', 'RTPA', 'TICIG', 'sICH.SITS', 'ARTERY.DSA.FINAL', 'TOAST']
    continuous_vars = ['TUOI', 'NIHSSNVIEN', 'ONSET-REP', 'HIRnew', 'ADC<620', 'MM.VOL', 'ASPECT/MRI']

    # One-hot encode the categorical variables
    data_encoded = pd.get_dummies(data[categorical_vars], drop_first=True)

    # Impute missing values in continuous variables
    data[continuous_vars] = data[continuous_vars].fillna(data[continuous_vars].mean())

    # Combine with continuous variables
    X = pd.concat([data_encoded, data[continuous_vars]], axis=1)
    
    # Standardize continuous variables
    scaler = RobustScaler()
    X[continuous_vars] = scaler.fit_transform(X[continuous_vars])

    return X

# Simulate a trained logistic regression model (for demonstration purposes, we can refit the model)
def train_model():
    data = load_data()
    X = prepare_data(data)
    y = data['MRS90DAY'].apply(lambda x: 1 if x < 3 else 0)  # 1 = Good (0-2), 0 = Bad (3-6)
    
    logit_model = sm.Logit(y, X).fit()
    return logit_model

# Load or train model
logit_model = train_model()

# Streamlit App
st.title("MRS Prediction Tool")
st.write("Enter the patient data to predict the probability of having a good MRS score (0-2).")

# Create input fields for user inputs
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

# Standardize continuous variables
scaler = RobustScaler()
input_data[continuous_vars] = scaler.transform(input_data[continuous_vars])

# Predict probability
predicted_prob = logit_model.predict(input_data)[0]

st.write(f"Predicted Probability of a Good MRS Outcome (0-2): {predicted_prob:.2f}")
