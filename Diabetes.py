import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load dataset
dt = pd.read_csv("diabetes.csv")
x = dt.iloc[:, :-1].values
y = dt.iloc[:, -1].values

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(x_train, y_train)

# Streamlit app
st.title('Diabetes Checker')
st.sidebar.header('User Input Parameters')

# Function to capture user input
def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', float(dt['Pregnancies'].min()), float(dt['Pregnancies'].max()), float(dt['Pregnancies'].mean()))
    Glucose = st.sidebar.slider('Glucose', float(dt['Glucose'].min()), float(dt['Glucose'].max()), float(dt['Glucose'].mean()))
    BloodPressure = st.sidebar.slider('BloodPressure', float(dt['BloodPressure'].min()), float(dt['BloodPressure'].max()), float(dt['BloodPressure'].mean()))
    SkinThickness = st.sidebar.slider('SkinThickness', float(dt['SkinThickness'].min()), float(dt['SkinThickness'].max()), float(dt['SkinThickness'].mean()))
    Insulin = st.sidebar.slider('Insulin', float(dt['Insulin'].min()), float(dt['Insulin'].max()), float(dt['Insulin'].mean()))
    BMI = st.sidebar.slider('BMI', float(dt['BMI'].min()), float(dt['BMI'].max()), float(dt['BMI'].mean()))
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', float(dt['DiabetesPedigreeFunction'].min()), float(dt['DiabetesPedigreeFunction'].max()), float(dt['DiabetesPedigreeFunction'].mean()))
    Age = st.sidebar.slider('Age', float(dt['Age'].min()), float(dt['Age'].max()), float(dt['Age'].mean()))

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_data = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_data)

# Make predictions
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display the results
st.subheader('Prediction')
st.write('Diabetic' if prediction[0] == 1 else 'Not Diabetic')

st.subheader('Prediction Probability')
st.write(prediction_proba)
