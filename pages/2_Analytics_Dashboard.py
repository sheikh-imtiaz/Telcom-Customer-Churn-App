"""Interactive exploratory analysis and customer retention metrics."""
import pandas as pd
import plotly.express as px
import streamlit as st
from churn_utils import load_telecom_churn_dataset

st.title("📊 Telecom Churn Analytics")
telecom_customers = load_telecom_churn_dataset()
contract_types = sorted(telecom_customers["Contract"].dropna().unique())
selected_contracts = st.multiselect("Contract type", contract_types, default=contract_types)
filtered_customers = telecom_customers[telecom_customers["Contract"].isin(selected_contracts)]
if filtered_customers.empty:
    st.warning("Select at least one contract type to view results.")
    st.stop()
churned_customers = filtered_customers["Churn"].astype(str).str.lower().isin(["yes", "1", "true"])
metric_columns = st.columns(4)
metric_columns[0].metric("Customer records", f"{len(filtered_customers):,}")
metric_columns[1].metric("Churn rate", f"{churned_customers.mean():.1%}")
metric_columns[2].metric("Average tenure", f"{filtered_customers['tenure'].mean():.1f} months")
metric_columns[3].metric("Average monthly charge", f"${filtered_customers['MonthlyCharges'].mean():,.2f}")
left_chart, right_chart = st.columns(2)
with left_chart:
    st.plotly_chart(px.histogram(filtered_customers, x="tenure", color="Churn", barmode="overlay", title="Customer tenure and churn"), use_container_width=True)
    st.plotly_chart(px.box(filtered_customers, x="Contract", y="MonthlyCharges", color="Churn", title="Monthly charges by contract"), use_container_width=True)
with right_chart:
    st.plotly_chart(px.histogram(filtered_customers, x="Churn", title="Churn outcome distribution"), use_container_width=True)
    contract_summary = filtered_customers.assign(_churn=churned_customers.values).groupby("Contract", as_index=False)["_churn"].mean()
    st.plotly_chart(px.bar(contract_summary, x="Contract", y="_churn", title="Churn rate by contract", labels={"_churn": "Churn rate"}), use_container_width=True)
st.caption("Charts describe the supplied dataset; they do not establish causal drivers of churn.")
