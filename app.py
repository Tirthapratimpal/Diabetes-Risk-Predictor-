# app.py

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="🩺 Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# ============================================
# BACKGROUND IMAGE USING CSS
# ============================================
background_image = """
<style>
body {
background-image: url("https://images.unsplash.com/photo-1581090700227-1e37b190418e?auto=format&fit=crop&w=1950&q=80");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
color: white;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# ============================================
# PAGE TITLE
# ============================================
st.markdown("<h1 style='text-align: center; color: #00FFAB;'>🩺 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #00C4FF;'>Predict your diabetes risk with your medical records</h4>", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND SCALER
# ============================================
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are in the same folder.")
    st.stop()

# ============================================
# SIDEBAR USER INPUT
# ============================================
st.sidebar.header("⚙️ Enter Patient Details")

preg = st.sidebar.number_input('🤰 Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.sidebar.number_input('🍬 Glucose Level', min_value=0, max_value=300, step=1)
bp = st.sidebar.number_input('💓 Blood Pressure', min_value=0, max_value=200, step=1)
skin = st.sidebar.number_input('🧴 Skin Thickness', min_value=0, max_value=100, step=1)
insulin = st.sidebar.number_input('💉 Insulin Level', min_value=0, max_value=900, step=1)
bmi = st.sidebar.number_input('⚖️ BMI', min_value=0.0, max_value=70.0, format="%.2f")
dpf = st.sidebar.number_input('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=2.5, format="%.3f")
age = st.sidebar.number_input('🎂 Age', min_value=0, max_value=120, step=1)

# ============================================
# PREDICT BUTTON
# ============================================
if st.sidebar.button("🚨 Predict Diabetes Risk"):
    # Prepare data
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("---")
    st.markdown("## 🧪 Prediction Result:")

    # Show the result
    if prediction[0] == 1:
        st.error("🚨 **High Risk Detected! Please consult a doctor 👨‍⚕️**")

        # Flashing Warning Text
        for i in range(6):
            color = "#FF4B4B" if i % 2 == 0 else "#FFD700"
            st.markdown(f"<h4 style='color:{color}; text-align:center;'>⚠️ HIGH RISK ⚠️</h4>", unsafe_allow_html=True)
            time.sleep(0.4)

        st.snow()  # Snow animation

        risk = ["High Risk", "Low Risk"]
        sizes = [80, 20]
        colors = ['#FF4B4B', '#00FFAB']

    else:
        st.success("✅ **Low Risk. Keep maintaining a healthy lifestyle 🌱**")
        st.balloons()  # Balloons animation

        risk = ["Low Risk", "High Risk"]
        sizes = [80, 20]
        colors = ['#00FFAB', '#FF4B4B']

    # ============================================
    # PIE CHART
    # ============================================
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=risk, colors=colors, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)

# ============================================
# FOOTER
# ============================================
st.write("---")
st.markdown("<div style='text-align: center;'>Made by 👽<b>Tirthapratim Pal</b></div>", unsafe_allow_html=True)


