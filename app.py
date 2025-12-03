import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load("XGmodel_top5.pkl")

# Page config
st.set_page_config(page_title="🎉 Reunion Prediction App", layout="centered")

st.image("Bucknell.jpeg", caption="Bucknell University", use_column_width=True)

# Title
st.title("🎯 XGBoost Prediction App")
st.markdown("Predict whether someone will participate in the reunion using top 5 features!")

# Sidebar with instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
Select Yes or No for each feature.
For graduation year, choose your year of graduation.
Click Predict to see the result.
""")

# Input fields
st.header("Input Features")

# Graduation year as slider
grad_year = st.slider("What year did you graduate? 🎓", min_value=1950, max_value=2025, value=2019)
current_year = 2025  # or use datetime.datetime.now().year
reunion_years_out = current_year - grad_year  # convert to model feature

# Other features
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

# Predict button
if st.button("🎉 Predict"):
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    
    if pred[0] == 1:
        st.success(f"✅ Yes! Likely to participate ({prob*100:.1f}% confidence)")
    else:
        st.error(f"❌ No. Unlikely to participate ({(1-prob)*100:.1f}% confidence)")

st.markdown("---")