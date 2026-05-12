import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DBT Fraud Detection Dashboard",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🏦 DBT Fraud Detection Dashboard")

st.markdown("""
Isolation Forest + XGBoost + SHAP + LIME
""")

# =========================================================
# LOAD MODELS
# =========================================================

xgb_model = joblib.load(
    "xgboost_fraud_model.pkl"
)

iso_model = joblib.load(
    "isolation_forest_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

feature_names = joblib.load(
    "feature_names.pkl"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙ Controls")

threshold = st.sidebar.slider(
    "Fraud Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📂 Upload CSV")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

# =========================================================
# MANUAL TRANSACTION
# =========================================================

st.sidebar.header("Manual Transaction")

step = st.sidebar.number_input(
    "Step",
    value=100
)

amount = st.sidebar.number_input(
    "Amount",
    value=5000.0
)

oldbalanceOrg = st.sidebar.number_input(
    "Old Balance Origin",
    value=10000.0
)

newbalanceOrig = st.sidebar.number_input(
    "New Balance Origin",
    value=5000.0
)

oldbalanceDest = st.sidebar.number_input(
    "Old Balance Destination",
    value=0.0
)

newbalanceDest = st.sidebar.number_input(
    "New Balance Destination",
    value=5000.0
)

nameOrig_txn_count = st.sidebar.number_input(
    "Origin Txn Count",
    value=2
)

nameDest_txn_count = st.sidebar.number_input(
    "Destination Txn Count",
    value=3
)

nameOrig_mean_amount = st.sidebar.number_input(
    "Origin Mean Amount",
    value=4000.0
)

nameDest_mean_amount = st.sidebar.number_input(
    "Destination Mean Amount",
    value=3500.0
)

hour = st.sidebar.number_input(
    "Hour",
    value=12
)

is_large_transaction = st.sidebar.selectbox(
    "Large Transaction",
    [0, 1]
)

type_PAYMENT = st.sidebar.selectbox(
    "PAYMENT",
    [0, 1]
)

type_TRANSFER = st.sidebar.selectbox(
    "TRANSFER",
    [0, 1]
)

type_CASH_OUT = st.sidebar.selectbox(
    "CASH_OUT",
    [0, 1]
)

# =========================================================
# MANUAL DATAFRAME
# =========================================================

manual_df = pd.DataFrame([{

    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "nameOrig_txn_count": nameOrig_txn_count,
    "nameDest_txn_count": nameDest_txn_count,
    "nameOrig_mean_amount": nameOrig_mean_amount,
    "nameDest_mean_amount": nameDest_mean_amount,
    "hour": hour,
    "is_large_transaction": is_large_transaction,
    "type_PAYMENT": type_PAYMENT,
    "type_TRANSFER": type_TRANSFER,
    "type_CASH_OUT": type_CASH_OUT

}])

# =========================================================
# DATA SOURCE
# =========================================================

if uploaded_file is not None:

    uploaded_df = pd.read_csv(
        uploaded_file
    )

else:

    uploaded_df = manual_df.copy()

# =========================================================
# FEATURE MATCHING
# =========================================================

for col in feature_names:

    if col not in uploaded_df.columns:

        uploaded_df[col] = 0

uploaded_df = uploaded_df[
    feature_names
]

# =========================================================
# SCALE
# =========================================================

scaled_data = scaler.transform(
    uploaded_df
)

# =========================================================
# ISOLATION FOREST
# =========================================================

iso_preds = iso_model.predict(
    scaled_data
)

iso_preds = np.where(
    iso_preds == -1,
    1,
    0
)

# =========================================================
# XGBOOST
# =========================================================

fraud_probs = xgb_model.predict_proba(
    scaled_data
)[:, 1]

fraud_preds = (
    fraud_probs >= threshold
).astype(int)

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = uploaded_df.copy()

results_df[
    "Fraud_Probability"
] = fraud_probs

results_df[
    "Fraud_Prediction"
] = fraud_preds

results_df[
    "Anomaly_Flag"
] = iso_preds

# =========================================================
# KPI DASHBOARD
# =========================================================

st.subheader("📊 KPI Dashboard")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Transactions",
    len(results_df)
)

k2.metric(
    "Frauds",
    int(
        results_df[
            "Fraud_Prediction"
        ].sum()
    )
)

k3.metric(
    "Anomalies",
    int(
        results_df[
            "Anomaly_Flag"
        ].sum()
    )
)

k4.metric(
    "Avg Fraud Risk",
    round(
        results_df[
            "Fraud_Probability"
        ].mean(),
        4
    )
)

# =========================================================
# RESULTS TABLE
# =========================================================

st.subheader("📋 Prediction Results")

st.dataframe(
    results_df
)

# =========================================================
# FRAUD TREND
# =========================================================

st.subheader("📈 Fraud Probability Trend")

line_fig = px.line(

    results_df,

    y="Fraud_Probability",

    markers=True,

    title="Fraud Probability Across Transactions"

)

st.plotly_chart(
    line_fig,
    use_container_width=True
)

# =========================================================
# FRAUD DISTRIBUTION
# =========================================================

st.subheader("🥧 Fraud Distribution")

fraud_count = int(
    results_df[
        "Fraud_Prediction"
    ].sum()
)

normal_count = int(
    len(results_df) - fraud_count
)

pie_df = pd.DataFrame({

    "Category": [
        "Normal",
        "Fraud"
    ],

    "Count": [
        normal_count,
        fraud_count
    ]

})

pie_fig = px.pie(

    pie_df,

    names="Category",

    values="Count",

    title="Fraud vs Normal"

)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)

# =========================================================
# THRESHOLD IMPACT
# =========================================================

st.subheader("🎚 Threshold Impact")

thresholds = np.arange(
    0.0,
    1.01,
    0.01
)

fraud_counts = []

for t in thresholds:

    temp_preds = (
        fraud_probs >= t
    ).astype(int)

    fraud_counts.append(
        int(temp_preds.sum())
    )

threshold_df = pd.DataFrame({

    "Threshold": thresholds,
    "Fraud_Count": fraud_counts

})

threshold_fig = px.line(

    threshold_df,

    x="Threshold",

    y="Fraud_Count",

    markers=True,

    title="Fraud Count vs Threshold"

)

threshold_fig.update_layout(

    xaxis_title="Threshold",
    yaxis_title="Predicted Fraud Count"

)

st.plotly_chart(
    threshold_fig,
    use_container_width=True
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("🔥 Feature Importance")

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": xgb_model.feature_importances_

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

importance_fig = px.bar(

    importance_df.head(10),

    x="Importance",

    y="Feature",

    orientation="h",

    title="Top Features"

)

st.plotly_chart(
    importance_fig,
    use_container_width=True
)

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

st.subheader("🧠 SHAP Explainability")

explainer = shap.TreeExplainer(
    xgb_model
)

sample_size = min(
    50,
    len(scaled_data)
)

sample_data = scaled_data[
    :sample_size
]

shap_values = explainer.shap_values(
    sample_data
)

fig_shap, ax = plt.subplots()

shap.summary_plot(

    shap_values,

    sample_data,

    feature_names=feature_names,

    show=False

)

st.pyplot(
    fig_shap
)

# =========================================================
# LIME WARNING
# =========================================================

st.warning(
    """
LIME explanations may be unstable for single transactions.
Use multiple transactions for better interpretability.
"""
)

# =========================================================
# LIME
# =========================================================

st.subheader("🔍 LIME Explainability")

lime_explainer = LimeTabularExplainer(

    training_data=scaled_data,

    feature_names=feature_names,

    class_names=[
        "Normal",
        "Fraud"
    ],

    mode="classification"

)

# =========================================================
# TRANSACTION SELECTION
# =========================================================

if len(results_df) == 1:

    selected_index = 0

    st.info(
        "Single transaction detected."
    )

else:

    selected_index = st.slider(

        "Select Transaction",

        min_value=0,

        max_value=len(results_df) - 1,

        value=0

    )

# =========================================================
# LIME EXPLANATION
# =========================================================

lime_exp = lime_explainer.explain_instance(

    scaled_data[selected_index],

    xgb_model.predict_proba,

    num_features=8

)

lime_fig = lime_exp.as_pyplot_figure()

st.pyplot(
    lime_fig
)

# =========================================================
# SPIDER GRAPH
# =========================================================

st.subheader("🕸 Transaction Spider Graph")

radar_features = [

    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"

]

radar_row = results_df.iloc[
    selected_index
]

radar_fig = go.Figure()

radar_fig.add_trace(

    go.Scatterpolar(

        r=[

            radar_row[x]
            for x in radar_features

        ],

        theta=radar_features,

        fill='toself',

        name='Transaction'

    )

)

radar_fig.update_layout(

    polar=dict(

        radialaxis=dict(
            visible=True
        )

    ),

    showlegend=False

)

st.plotly_chart(
    radar_fig,
    use_container_width=True
)