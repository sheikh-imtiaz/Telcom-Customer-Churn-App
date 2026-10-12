"""View predictions retained in the active Streamlit session."""
import pandas as pd
import streamlit as st

st.title("🕒 Prediction History")
recent_predictions = st.session_state.get("prediction_history", [])
if not recent_predictions:
    st.info("No predictions recorded in this session. Make a single-customer prediction first.")
else:
    prediction_history = pd.DataFrame(recent_predictions)
    st.dataframe(prediction_history, use_container_width=True, hide_index=True)
    st.download_button("Export session history", prediction_history.to_csv(index=False).encode(), "churn_prediction_history.csv", "text/csv")
    if st.button("Clear session history"):
        st.session_state["prediction_history"] = []
        st.rerun()
