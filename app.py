import streamlit as st
import joblib
import pandas as pd

model = joblib.load("XGmodel_top5.pkl")

st.set_page_config(page_title="🎉 Reunion Prediction App", layout="centered")

st.image("Bucknell.jpeg", use_column_width=True)

# Title
st.title("XGBoost Prediction App")
st.markdown("Predict whether someone will accept your invitation to the reunion using our top 5  features!")

# Graduation year as slider
grad_year = st.slider("What year did you graduate? 🎓", min_value=1950, max_value=2025, value=2019)
current_year = 2025 
reunion_years_out = current_year - grad_year

peer = st.selectbox("Did a friend refer you? 🫂", ["Yes", "No"])

volunteer = st.selectbox("Did you volunteer during your time at Bucknell ❤️", ["Yes", "No"])

greek = st.selectbox("Were you in Greek life 🏠", ["Yes", "No"])

engineering_bachelor = st.selectbox("Did you obtain a Bachelor's Degree in Engineering? 📏", ["Yes", "No"])

# Convert inputs to model format
input_data = pd.DataFrame([[
    reunion_years_out,
    1 if peer=="Yes" else 0,
    1 if volunteer=="Yes" else 0,
    1 if greek=="Yes" else 0,
    1 if engineering_bachelor=="Yes" else 0
]], columns=['Reunion_Years_Out','Peer','Volunteer','Greek?','Engineering_Bachelor'])

# Predict
if st.button("🎉 Predict"):
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    
    if pred[0] == 1:
        st.success(f"✅ Yes! Likely to accept your invitation ({prob*100:.1f}% confidence)")
    else:
        st.error(f"❌ No. Unlikely to accept your invitation ({(1-prob)*100:.1f}% confidence)")
