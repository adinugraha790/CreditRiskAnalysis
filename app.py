import streamlit as st
import pickle
import pandas as pd
import sklearn

# Load the pre-trained LightGBM pipeline
with open("lightgbm_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Title and description
st.title("Credit Risk Prediction App")
st.write("Provide the following details to predict the credit risk.")

# Input fields
term = st.selectbox("Loan Term (months)", [36, 60])
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
sub_grade = st.selectbox("Loan Sub-Grade", [f"{grade}{i}" for i in range(1, 6)])
emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, step=1)
home_ownership = st.selectbox("Home Ownership Status", ["RENT", "OWN", "MORTGAGE", "OTHER"])
annual_inc = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0)
verification_status = st.selectbox("Income Verification Status", ["Verified", "Source Verified", "Not Verified"])
loan_status = st.selectbox("Loan Status", ["Fully Paid", "Charged Off", "Current", "Late"])
purpose = st.selectbox("Loan Purpose", ['debt_consolidation', 'credit_card', 'other', 'home_improvement', 'major_purchase', 'small_business', 'car'])
addr_state = st.selectbox("State", ['CA', 'other', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA', 'NC', 'MI', 'MA', 'MD', 'AZ', 'WA', 'CO', 'MN', 'MO', 'CT', 'IN', 'NV', 'TN', 'OR', 'WI', 'AL', 'SC', 'LA'])
dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, step=0.1)
delinq_2yrs = st.number_input("Delinquencies (last 2 years)", min_value=0, step=1)
inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0, step=1)
mths_since_last_delinq = st.number_input("Months Since Last Delinquency", min_value=0, step=1)
open_acc = st.number_input("Number of Open Credit Lines", min_value=0, step=1)
pub_rec = st.number_input("Number of Public Records", min_value=0, step=1)
revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, step=100.0)
revol_util = st.number_input("Revolving Utilization Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
total_acc = st.number_input("Total Number of Credit Lines", min_value=0, step=1)

# Predict button
if st.button("Predict Credit Risk"):
    # Prepare input data
    input_data = pd.DataFrame({
        "term": [term],
        "int_rate": [int_rate],
        "grade": [grade],
        "sub_grade": [sub_grade],
        "emp_length": [emp_length],
        "home_ownership": [home_ownership],
        "annual_inc": [annual_inc],
        "verification_status": [verification_status],
        "loan_status": [loan_status],
        "purpose": [purpose],
        "addr_state": [addr_state],
        "dti": [dti],
        "delinq_2yrs": [delinq_2yrs],
        "inq_last_6mths": [inq_last_6mths],
        "mths_since_last_delinq": [mths_since_last_delinq],
        "open_acc": [open_acc],
        "pub_rec": [pub_rec],
        "revol_bal": [revol_bal],
        "revol_util": [revol_util],
        "total_acc": [total_acc],
    })

    # Make prediction
    prediction = pipeline.predict(input_data)
    probability = pipeline.predict_proba(input_data)[:, 1]

    # Display results
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.write(f"### Predicted Risk: {risk}")
    st.write(f"### Risk Probability: {probability[0]:.2f}")
