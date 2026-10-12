"""Single-customer and batch inference using the provided pretrained models."""
import pandas as pd
import streamlit as st
from churn_utils import MODEL_PATHS, predict_customer_churn

st.title("🔮 Customer Churn Prediction")
selected_model = st.selectbox("Prediction model", list(MODEL_PATHS))
manual_tab, bulk_tab = st.tabs(["Single customer", "Batch CSV prediction"])

with manual_tab:
    with st.form("customer_churn_form"):
        st.subheader("Customer profile")
        left, right = st.columns(2)
        with left:
            customer_id = st.text_input("Customer reference (optional)", "DEMO-001")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"])
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=65.0)
            total_charges = st.number_input("Total charges", min_value=0.0, value=780.0)
        with right:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            internet_addon_options = ["No", "Yes", "No internet service"]
            online_security = st.selectbox("Online security", internet_addon_options)
            online_backup = st.selectbox("Online backup", internet_addon_options)
            device_protection = st.selectbox("Device protection", internet_addon_options)
            tech_support = st.selectbox("Tech support", internet_addon_options)
            streaming_tv = st.selectbox("Streaming TV", internet_addon_options)
            streaming_movies = st.selectbox("Streaming movies", internet_addon_options)
        submitted = st.form_submit_button("Predict customer churn", type="primary")
    if submitted:
        customer_record = pd.DataFrame([{
            "customerID": customer_id or "DEMO-001", "gender": gender,
            "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])
        try:
            churn_labels, churn_probabilities = predict_customer_churn(selected_model, customer_record)
            predicted_label = churn_labels.iloc[0]
            st.success(f"Predicted churn: {predicted_label}")
            if churn_probabilities is not None:
                st.metric("Estimated churn probability", f"{float(churn_probabilities[0]):.1%}")
            else:
                st.caption("This model does not expose a verifiable churn probability.")
            # Session-only history: avoids unreliable shared CSV writes on free hosts.
            st.session_state.setdefault("prediction_history", []).append({
                "Customer reference": customer_id, "Model": selected_model,
                "Prediction": predicted_label,
                "Churn probability": None if churn_probabilities is None else float(churn_probabilities[0]),
                "Timestamp (UTC)": pd.Timestamp.now(tz="UTC").isoformat(),
            })
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")
            st.info("The supplied legacy joblib files may require their original sklearn version or transformer classes. See README compatibility notes.")
with bulk_tab:
    uploaded_csv = st.file_uploader("Upload a CSV with the original customer feature columns", type="csv")
    if uploaded_csv is not None:
        uploaded_customers = pd.read_csv(uploaded_csv)
        st.write(f"Uploaded {len(uploaded_customers):,} customer records")
        if st.button("Predict uploaded customers"):
            try:
                predicted_labels, predicted_probabilities = predict_customer_churn(selected_model, uploaded_customers)
                predictions = uploaded_customers.copy()
                predictions["PredictedChurn"] = predicted_labels.to_numpy()
                if predicted_probabilities is not None:
                    predictions["ChurnProbability"] = predicted_probabilities
                st.dataframe(predictions.head(100), use_container_width=True)
                st.download_button("Download predictions", predictions.to_csv(index=False).encode(), "telecom_churn_predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Unable to score the uploaded data: {exc}")
