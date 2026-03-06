import streamlit as st
import pandas as pd

st.set_page_config(page_title="EWS Dasboard", layout="wide")

st.sidebar.title("EWS Control Panel")
st.sidebar.markdown("Use this panel to filter the portfolio and manage agent resources.")

call_threshold = st.sidebar.slider("Phone Call Threshold (%)", min_value=0, max_value=100, value=60, step=5)
sms_threshold = st.sidebar.slider("SMS Threshold (%)", min_value=0, max_value=100, value=30, step=5)

st.sidebar.divider()
st.sidebar.info("Model powered byXGBoost. Feature tracking: 6-Month Behavioral History")

st.title("Collections Early Warning System")
st.markdown("Identify high-risk accounts and deploy targeted collections strategies before they roll into 60+ DPD.")

st.subheader("Portfolio Overview (This Month)")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Accounts", "5,000")
col2.metric("High Risk (Requires Call)", "432", "+12 from last week", delta_color="inverse")
col3.metric("Medium Risk (Requires SMS)", "950")
col4.metric("Est. Preventable Loss", "$246,880")

st.divider()

st.subheader("🔍 Individual Customer Lookup")
st.markdown("Enter a customer ID to run a real-time behavioral risk assessment.")

customer_id = st.text_input("Customer ID (e.g., CUST-8472):")

if st.button("Run Risk Assessment"):
    if customer_id:
        st.success(f"Fetching 6-month behavioral history for {customer_id}...")

        st.warning("Prediction Engine is currently offline. Please connect the XGBoost model.")
    else:
        st.error("Please enter a valid Customer ID.")