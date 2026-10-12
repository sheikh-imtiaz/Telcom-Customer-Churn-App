"""Entry point for the Telecom Customer Churn Insights Streamlit application."""
import streamlit as st

st.set_page_config(page_title="Telecom Churn Insights", page_icon="📡", layout="wide")
st.title("📡 Telecom Customer Churn Insights")
st.caption("Customer retention analytics and machine learning predictions")
st.markdown("""
Explore telecom customer patterns, evaluate churn risk, and download prediction results.
Use the navigation sidebar to open **Data Explorer**, **Analytics Dashboard**,
**Churn Prediction**, or **Prediction History**.

### Project scope
This app packages the supplied cleaned churn dataset and two pretrained model artifacts:
**Gradient Boosting** and **Support Vector Machine**. Model predictions are estimates,
not guaranteed customer outcomes. The supplied repository does not include model training
code, reproducible validation results, or calibration evidence.

### Deployment note
Prediction history is stored only in the current browser session; it is not a shared
persistent database. Avoid entering real customer identifiers or sensitive information.
""")
st.info("Start with the Analytics Dashboard to understand the customer population.")
