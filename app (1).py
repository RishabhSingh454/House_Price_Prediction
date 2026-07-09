import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter house details")

area = st.number_input("Area (sq ft)", 500, 10000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
stories = st.number_input("Stories", 1, 5)

if st.button("Predict Price"):

    data = np.array([[area, bedrooms, bathrooms, stories]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")
