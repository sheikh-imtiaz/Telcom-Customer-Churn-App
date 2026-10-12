"""Shared dataset and model helpers for the telecom churn dashboard."""
from pathlib import Path
import joblib
import __main__
from sklearn.base import BaseEstimator, TransformerMixin

class TotalCharges_cleaner(BaseEstimator, TransformerMixin):
    """Compatibility shim for the original model artifact transformer."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        cleaned = X.copy()
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        return cleaned
    def get_feature_names_out(self, input_features=None):
        return input_features

class columnDropper(BaseEstimator, TransformerMixin):
    """Compatibility shim preserving the original serialized class name."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=["customerID"], errors="ignore")
    def get_feature_names_out(self, input_features=None):
        return [feature for feature in input_features if feature != "customerID"] if input_features is not None else None

# The supplied joblib files reference these classes under __main__.
__main__.TotalCharges_cleaner = TotalCharges_cleaner
__main__.columnDropper = columnDropper
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "telecom_churn_cleaned.csv"
MODEL_PATHS = {
    "Gradient Boosting": PROJECT_ROOT / "models" / "GradientBoosting.joblib",
    "Support Vector Machine": PROJECT_ROOT / "models" / "SupportVector.joblib",
}
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
CUSTOMER_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

@st.cache_data(show_spinner=False)
def load_telecom_churn_dataset():
    """Load bundled cleaned data once and cache it between page reruns."""
    churn_data = pd.read_csv(DATASET_PATH)
    churn_data["TotalCharges"] = pd.to_numeric(churn_data["TotalCharges"], errors="coerce")
    return churn_data

@st.cache_resource(show_spinner="Loading churn model...")
def load_churn_model(model_name):
    """Load a trusted local serialized pipeline and its target label encoder."""
    return joblib.load(MODEL_PATHS[model_name]), joblib.load(ENCODER_PATH)

def normalize_customer_features(customer_data):
    """Align uploaded and manually entered records to the training schema."""
    missing_features = sorted(set(CUSTOMER_FEATURES) - set(customer_data.columns))
    if missing_features:
        raise ValueError("Missing required features: " + ", ".join(missing_features))
    normalized = customer_data.loc[:, CUSTOMER_FEATURES].copy()
    for numeric_feature in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
        normalized[numeric_feature] = pd.to_numeric(normalized[numeric_feature], errors="coerce")
    if normalized[["tenure", "MonthlyCharges", "TotalCharges"]].isna().any().any():
        raise ValueError("Tenure and charge amounts must contain valid numbers.")
    return normalized

def predict_customer_churn(model_name, customer_data):
    """Return labels and optional churn probability; never fabricate probabilities."""
    churn_pipeline, churn_encoder = load_churn_model(model_name)
    normalized = normalize_customer_features(customer_data)
    # Legacy training pipelines may expect a customerID even though it is dropped internally.
    if "customerID" in customer_data.columns:
        normalized.insert(0, "customerID", customer_data["customerID"].astype(str).values)
    else:
        normalized.insert(0, "customerID", [f"APP-{i:05d}" for i in range(len(normalized))])
    raw_labels = churn_pipeline.predict(normalized)
    try:
        labels = churn_encoder.inverse_transform(raw_labels)
    except (ValueError, TypeError):
        labels = raw_labels
    labels = pd.Series(labels).astype(str).replace({"1": "Yes", "0": "No", "True": "Yes", "False": "No"})
    churn_probability = None
    if hasattr(churn_pipeline, "predict_proba"):
        probability_matrix = churn_pipeline.predict_proba(normalized)
        classes = list(getattr(churn_pipeline, "classes_", []))
        # The positive class may be numeric encoded 1 or the text label Yes.
        positive_index = next((index for index, value in enumerate(classes) if str(value).lower() in ("1", "yes", "true")), None)
        if positive_index is not None:
            churn_probability = probability_matrix[:, positive_index]
    return labels, churn_probability
