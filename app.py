import os  
import subprocess  

# Install missing packages  
packages = ["joblib", "xgboost", "scikit-learn", "numpy", "pandas", "streamlit"]  
for package in packages:  
    try:  
        __import__(package)  
    except ImportError:  
        subprocess.run(["pip", "install", package])  

# Now import the necessary libraries  
import joblib  
import numpy as np  
import pandas as pd  
import streamlit as st  
import streamlit as st
import joblib
import numpy as np

# Load the trained model
xgb_model = joblib.load("titanic_xgboost_model.pkl")

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival:")

# Get user input
age = st.number_input("Age", min_value=1, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
embarked = st.selectbox("Embarked From", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

# Convert inputs for the model
sex = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Queenstown (Q)" else 0
embarked_S = 1 if embarked == "Southampton (S)" else 0

# ✅ Define the features before using them
features = np.array([[age, fare, sex, pclass, embarked_Q, embarked_S]])

# ✅ Now print the features
st.write("🔍 Model Input Features:", features)

# Make prediction when user clicks the button
if st.button("Predict Survival"):
    prediction = xgb_model.predict(features)[0]
    
    # Show result
    if prediction == 1:
        st.success("🎉 This passenger **survived**!")
    else:
        st.error("⚠️ This passenger **did not survive**.")


