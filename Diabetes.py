import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Importing Models
from sklearn.ensemble import RandomForestClassifier

# Importing Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

dt = pd.read_csv("diabetes.csv")
x = dt.iloc[:, :-1].values
y = dt.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)

st.title('Diabetes Checker')
st.sidebar.header('User Input Parameters')

def user_input_features():
        Pregnancies = st.sidebar.slider('Pregnancies', float(dt['Pregnancies'].min()), float(dt['Pregnancies'].max()), float(dt['Pregnancies'].mean()))
        Glucose = st.sidebar.slider('Glucose', float(dt['Glucose'].min()), float(dt['Glucose'].max()), float(dt['Glucose'].mean()))
        BloodPressure = st.sidebar.slider('BloodPressure', float(dt['BloodPressure'].min()), float(dt['BloodPressure'].max()), float(dt['BloodPressure'].mean()))
        SkinThickness = st.sidebar.slider('SkinThickness', float(dt['SkinThickness'].min()), float(dt['SkinThickness'].max()), float(dt['SkinThickness'].mean()))
        Insulin = st.sidebar.slider('Insulin', float(dt['Insulin'].min()), float(dt['Insulin'].max()), float(dt['Insulin'].mean()))
        BMI = st.sidebar.slider('BMI', float(dt['BMI'].min()), float(dt['BMI'].max()), float(dt['BMI'].mean()))
        DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', float(dt['DiabetesPedigreeFunction'].min()), float(dt['DiabetesPedigreeFunction'].max()), float(dt['DiabetesPedigreeFunction'].mean()))
        Age = st.sidebar.slider('Age', float(dt['Age'].min()), float(dt['Age'].max()), float(dt['Age'].mean()))
    
        data = {'Pregnancies': Pregnancies, 
            'Glucose': Glucose, 
            'BloodPressure': BloodPressure, 
            'SkinThickness': SkinThickness, 
            'Insulin': Insulin, 
            'BMI': BMI, 
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 
            'Age': Age}
    
        features = pd.DataFrame(data, index=[0])
        return features
