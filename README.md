# 📡 Telecom Customer Churn Insights

A portfolio-ready **Streamlit** dashboard for exploring telecommunications churn, viewing customer retention KPIs, and scoring customers with the supplied **Gradient Boosting** and **Support Vector Machine** models.

> **Status:** Refactored source code and deployment configuration. The bundled legacy model artifacts have not been proven compatible with a fresh hosted environment; validate predictions before presenting the deployment as production-ready.

## Features

- **Data Explorer:** browse the cleaned customer churn dataset, filter numeric/categorical columns, export CSV.
- **Analytics Dashboard:** churn rate, customer counts, average tenure, billing metrics, contract breakdowns and Plotly charts.
- **Churn Prediction:** single-customer form and batch CSV predictions using the existing serialized models.
- **Prediction History:** session-only history with CSV export. No shared database or persistent personal-data storage.
- **Portable paths:** no machine-specific Windows paths; files load relative to the project root.

## Project layout

```text
.
├── app.py
├── churn_utils.py
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Analytics_Dashboard.py
│   ├── 3_Churn_Prediction.py
│   └── 4_Prediction_History.py
├── data/
│   ├── telecom_churn_cleaned.csv
│   └── telecom_churn_sample.csv
├── models/
│   ├── GradientBoosting.joblib
│   ├── SupportVector.joblib
│   └── label_encoder.joblib
├── .streamlit/config.toml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed by Streamlit (typically `http://localhost:8501`).

## Deploy free with Streamlit Community Cloud (recommended)

1. Create a public GitHub repository, for example `telecom-customer-churn-streamlit`.
2. Upload the **contents** of this folder, keeping `app.py` at the repository root.
3. Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
4. Select **Create app → Deploy a public app from GitHub** (interface labels may change).
5. Choose the repository and branch, set **main file path** to `app.py`, then deploy.
6. Inspect build logs if dependency installation or model loading fails.

Streamlit Community Cloud offers free community hosting subject to platform resource and usage limits. The app can sleep when idle. This repository does not require API keys.

**Alternative:** [Hugging Face Spaces](https://huggingface.co/spaces) with a Docker Space using the included `Dockerfile`. Free hardware quotas and availability can change.

## Model artifact compatibility — important

The original project supplied three `joblib` artifacts but no training environment lockfile, model-training script, or guaranteed scikit-learn version. Serialized scikit-learn pipelines are often version-sensitive. The original prediction code defines custom transformers named `TotalCharges_cleaner` and `columnDropper` in `__main__`. This refactor includes compatibility shims registered under those names, and pins scikit-learn to the detected training version 1.4.2 with imbalanced-learn 0.12.3. These are best-effort compatibility measures and have not been verified with an end-to-end prediction run. The safest fix is to **retrain or re-export the pipelines** in a reproducible environment, with custom transformers imported from a dedicated module, and pin the tested dependency versions. Do not load untrusted joblib files.

The app intentionally reports inference errors rather than silently substituting a different model or claiming a successful prediction. A prediction probability is displayed only if the model exposes `predict_proba` and the positive class can be identified.

## Data and schema

The provided cleaned file contains **4,118 rows and 20 columns**, including the `Churn` target. The second provided CSV contains **3,000 rows and 21 columns**, including `customerID`. Prediction input requires the original telecom customer fields such as tenure, contract, payment method, internet service, monthly charges, and total charges. The app drops the target from prediction inputs.

The dashboard is descriptive; its patterns do not prove causality. There are no verified out-of-sample accuracy, recall, F1, ROC-AUC, or calibration metrics in this package. Add a held-out evaluation and documented training pipeline before making performance claims.

## Privacy and limitations

- Use synthetic or authorized data only; avoid exposing real customer IDs in a public repository.
- Prediction history exists only in the current Streamlit browser session and disappears when the session ends.
- Model predictions support exploration and do not replace human review.
- GitHub public repositories expose all committed datasets and model files; review licenses and data permissions before publishing.

## Future improvements

Reproducible model training and version pinning, model performance report, feature attribution, automated tests, authenticated persistent prediction history, and CI/CD.

## License

The uploaded source included an MIT-style LICENSE file; see `LICENSE` for its original terms and attribution.

## Live Demo Link
https://telcom-customer-churn-app-laibevauktt7jlzmih3su8.streamlit.app/
