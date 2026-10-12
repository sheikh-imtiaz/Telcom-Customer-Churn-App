"""Filter and export the cleaned telecom churn dataset."""
import streamlit as st
from churn_utils import load_telecom_churn_dataset

st.title("🗂️ Telecom Customer Data Explorer")
telecom_customers = load_telecom_churn_dataset()
feature_type = st.selectbox("Display columns", ["All features", "Numeric features", "Categorical features"])
if feature_type == "Numeric features":
    selected_data = telecom_customers.select_dtypes(include="number")
elif feature_type == "Categorical features":
    selected_data = telecom_customers.select_dtypes(exclude="number")
else:
    selected_data = telecom_customers
st.caption(f"{len(telecom_customers):,} records · {len(selected_data.columns)} visible columns")
st.dataframe(selected_data, use_container_width=True, hide_index=True)
st.download_button("Download visible dataset (CSV)", selected_data.to_csv(index=False).encode(), "telecom_customer_data.csv", "text/csv")
with st.expander("About the dataset"):
    st.write("The supplied cleaned dataset contains telecom account, service, billing, and churn fields. The Churn target indicates whether a customer left the service.")
