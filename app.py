import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
try:
    model = pickle.load(open('heart_disease_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Please run 'train_model.py' first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# --- UI Design ---
st.title("❤️ Cardiovascular Disease Prediction")
st.markdown("Enter the patient's medical details below to predict the risk of heart disease.")

# --- Input Form ---
# We create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], help="0: Normal, 1: Fixed Defect, 2: Reversable Defect")

# --- Prediction Logic ---
if st.button("🔍 Predict Risk"):
    # Prepare input array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale the input using the saved scaler
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    
    # Display Result
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ HIGH RISK DETECTED!")
        st.write(f"The model predicts a **{probability[0][1]*100:.2f}%** probability of heart disease.")
        st.markdown("**Recommendation:** Please consult a cardiologist immediately.")
    else:
        st.success(f"✅ LOW RISK / HEALTHY")
        st.write(f"The model predicts a **{probability[0][0]*100:.2f}%** probability of being healthy.")
        st.markdown("**Recommendation:** Maintain a healthy lifestyle and regular checkups.")