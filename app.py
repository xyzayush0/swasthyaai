import streamlit as st
import numpy as np

st.title("🫀 SwasthyaAI - AI Health Risk Predictor")

st.write("AI-based preventive health screening tool")

st.header("Enter Health Details")

age = st.number_input("Age", 1, 100, 30)
bp = st.number_input("Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 400, 200)

# Simple ML-like risk formula
risk_score = (age * 0.3) + (bp * 0.2) + (chol * 0.1)

st.subheader("Risk Score")
st.progress(int(min(risk_score, 100)))

if st.button("Predict Risk"):
    if risk_score > 80:
        st.error("High Risk ⚠️")
    elif risk_score > 50:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk ✅")

    st.write(f"Calculated Risk Score: {round(risk_score,2)}")
