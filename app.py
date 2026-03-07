import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import plotly.express as px

st.set_page_config(page_title="EWS Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model/xgb_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/customer_database.csv")


try:
    model = load_model()
    df_db = load_data()
    assets_loaded = True
except FileNotFoundError:
    assets_loaded = False

st.sidebar.title("EWS Control Panel")
st.sidebar.markdown("Manage collection thresholds.")

call_threshold = st.sidebar.slider(
    "Phone Call Threshold (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5
)

sms_threshold = st.sidebar.slider(
    "SMS Threshold (%)",
    min_value=0,
    max_value=100,
    value=30,
    step=5
)
st.title("🚨 Credit Risk Early Warning System")
st.markdown(
    "Identify high-risk accounts and deploy targeted collections strategies."
)

if not assets_loaded:
    st.error(
        "Model or Data files missing! Ensure 'xgb_model.pkl' and "
        "'customer_database.csv' exist."
    )
    st.stop()

df = df_db.copy()

cols_to_drop = ['ID', 'target', 'PAY_RATIO_BIN', 'TREND_BIN']

X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

probabilities = model.predict_proba(X)[:, 1]

df["Predicted_Risk_Pct"] = probabilities * 100

conditions = [
    df["Predicted_Risk_Pct"] >= call_threshold,
    df["Predicted_Risk_Pct"] >= sms_threshold
]

choices = [
    "🚨 PHONE CALL",
    "📱 SMS"
]

df["Action_Tier"] = np.select(conditions, choices, default="✅ SAFE")

phone_count = (df["Action_Tier"] == "🚨 PHONE CALL").sum()
sms_count = (df["Action_Tier"] == "📱 SMS").sum()
safe_count = (df["Action_Tier"] == "✅ SAFE").sum()

st.subheader("📊 Portfolio Risk Summary")

col1, col2, col3 = st.columns(3)

col1.metric("🚨 Phone Call Required", phone_count)
col2.metric("📱 SMS Reminder", sms_count)
col3.metric("✅ Safe Accounts", safe_count)

st.subheader("🎯 High Risk Customer Action List")

action_df = df[df["Action_Tier"] != "✅ SAFE"]

action_df = action_df.sort_values(
    by="Predicted_Risk_Pct",
    ascending=False
)

st.dataframe(
    action_df[
        [
            "ID",
            "Predicted_Risk_Pct",
            "Action_Tier",
            "PAY_0",
            "MAX_DLQ"
        ]
    ],
    width='stretch'
)

st.subheader("📈 Risk Distribution")

tier_counts = df["Action_Tier"].value_counts().reset_index()

tier_counts.columns = ["Action_Tier", "Count"]

fig = px.bar(
    tier_counts,
    x="Action_Tier",
    y="Count",
    color="Action_Tier",
    title="Customer Segmentation by Risk Tier"
)

st.plotly_chart(fig, width='stretch')