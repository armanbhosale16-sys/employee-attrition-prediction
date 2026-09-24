import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Employee Attrition Prediction System", layout="centered")

st.title("💼 Employee Attrition Prediction System")
st.write("Enter employee details to predict whether the employee will Stay or Leave the organization:")

# Load Model from 'model' folder
model_path = os.path.join('model', 'attrition_model.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_model.py' first!")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=18, max_value=65, value=30)
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=50000, value=5000)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
    job_satisfaction = st.selectbox("Job Satisfaction (1: Low - 4: High)", [1, 2, 3, 4], index=2)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

with col2:
    overtime = st.selectbox("Overtime", ["No", "Yes"])
    work_life_balance = st.selectbox("Work-Life Balance (1: Bad - 4: Best)", [1, 2, 3, 4], index=2)
    years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
    previous_experience = st.number_input("Previous Work Experience (Years)", min_value=0, max_value=30, value=5)

overtime_val = 1 if overtime == "Yes" else 0

input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'YearsAtCompany': [years_at_company],
    'JobSatisfaction': [job_satisfaction],
    'OverTime': [overtime_val],
    'JobLevel': [job_level],
    'WorkLifeBalance': [work_life_balance],
    'YearsSinceLastPromotion': [years_since_promotion],
    'PreviousWorkExperience': [previous_experience]
})

st.markdown("---")

if st.button("Predict Attrition", type="primary"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("🚨 **Prediction:** Employee is likely to **LEAVE**")
    else:
        st.success("✅ **Prediction:** Employee is likely to **STAY**")

    st.write(f"**Stay Probability:** {probability[0]*100:.2f}%")
    st.write(f"**Leave Probability:** {probability[1]*100:.2f}%")

with st.expander("Show Input Summary"):
    st.dataframe(input_data)

with st.expander("Interpretation & Key Insights"):
    st.markdown("""
    - **Key Factors:** High Overtime, Low Monthly Income, and Low Job Satisfaction contribute significantly to employee attrition.
    - **Confusion Matrix:** Measures the classification accuracy between predicted and actual classes.
    - **Limitations:** Performance depends on dataset size and balance between class labels.
    """)