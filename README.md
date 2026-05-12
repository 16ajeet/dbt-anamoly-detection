# Installation step.
after cloning from github, open the project and run create virtual environment windows command (section 2.2) in terminal then install requirements by running command from section 3 followed by running section 4 run application command for running the project on ur local system.  

# for running the app
upload any csv file from assets folder and the app would show predictions alongwwith explanations

# ⚙ Installation

## 1. Clone Repository

```bash
git clone <repo-url>
cd dbt-fraud-project
```

---

## 2. Create Virtual Environment

### 2.1 Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶ Run Application

```bash
streamlit run app.py
```

---

# 🏦 DBT Fraud Detection Dashboard

A Machine Learning based fraud detection system built using **Isolation Forest**, **XGBoost**, **SHAP**, **LIME**, and **Streamlit** for detecting suspicious Direct Benefit Transfer (DBT) transactions.

---

# 📌 Project Overview

This project detects fraudulent financial transactions using a hybrid anomaly detection and supervised learning approach.

The system combines:

- **Isolation Forest** → detects anomalies
- **XGBoost** → predicts fraud probability
- **SHAP** → global explainability
- **LIME** → local explainability
- **Streamlit Dashboard** → interactive UI

The dashboard allows users to:

- Upload transaction CSV files
- Predict fraud in real-time
- Adjust fraud threshold dynamically
- Analyze fraud trends
- Visualize feature importance
- Understand model decisions using explainable AI

---

# 🚀 Features

## ✅ Fraud Detection

Predicts whether a transaction is:

- Normal
- Fraudulent

using trained XGBoost model probabilities.

---

## ✅ Isolation Forest Anomaly Detection

Detects abnormal transaction behavior independently from fraud labels.

Useful for:

- unknown attack patterns
- suspicious unseen behavior
- anomaly analysis

---

## ✅ Threshold Adjustment

Interactive fraud threshold slider.

Users can dynamically change prediction sensitivity.

Example:

- Lower threshold → more frauds detected
- Higher threshold → fewer but more confident fraud predictions

---

## ✅ CSV Upload Support

Users can upload custom transaction datasets.

Supported format:

```csv
.csv

````md
# 🏦 DBT Fraud Detection Dashboard

A Machine Learning based fraud detection system built using **Isolation Forest**, **XGBoost**, **SHAP**, **LIME**, and **Streamlit** for detecting suspicious Direct Benefit Transfer (DBT) transactions.

---

# 📌 Project Overview

This project detects fraudulent financial transactions using a hybrid anomaly detection and supervised learning approach.

The system combines:

- **Isolation Forest** → detects anomalies
- **XGBoost** → predicts fraud probability
- **SHAP** → global explainability
- **LIME** → local explainability
- **Streamlit Dashboard** → interactive UI

The dashboard allows users to:

- Upload transaction CSV files
- Predict fraud in real-time
- Adjust fraud threshold dynamically
- Analyze fraud trends
- Visualize feature importance
- Understand model decisions using explainable AI

---

# 🚀 Features

## ✅ Fraud Detection

Predicts whether a transaction is:

- Normal
- Fraudulent

using trained XGBoost model probabilities.

---

## ✅ Isolation Forest Anomaly Detection

Detects abnormal transaction behavior independently from fraud labels.

Useful for:

- unknown attack patterns
- suspicious unseen behavior
- anomaly analysis

---

## ✅ Threshold Adjustment

Interactive fraud threshold slider.

Users can dynamically change prediction sensitivity.

Example:

- Lower threshold → more frauds detected
- Higher threshold → fewer but more confident fraud predictions

---

## ✅ CSV Upload Support

Users can upload custom transaction datasets.

Supported format:

```csv
.csv
````

---

## ✅ Manual Transaction Prediction

Users can manually enter transaction values from sidebar controls.

Useful for:

* testing
* demonstrations
* real-time fraud checking

---

# 📊 Dashboard Visualizations

---

## 📈 Fraud Probability Trend

Line chart showing fraud probability across transactions.

Helps identify:

* spikes
* risky transactions
* suspicious patterns

---

## 🥧 Fraud Distribution Chart

Pie chart showing:

* fraud transactions
* normal transactions

---

## 📉 Threshold Impact Curve

Shows how fraud count changes when threshold changes.

Helps analyze:

* model sensitivity
* prediction behavior
* threshold optimization

---

## 🔥 Feature Importance Graph

Displays most important fraud detection features learned by XGBoost.

Examples:

* transaction amount
* balance changes
* transaction type
* account behavior

---

## 🕸 Transaction Spider Graph

Radar chart for transaction comparison.

Visualizes transaction characteristics across multiple dimensions.

---

# 🧠 Explainable AI (XAI)

---

## ✅ SHAP Explainability

SHAP explains:

* global feature impact
* model behavior
* feature contribution

Used for:

* interpretability
* debugging
* auditability

---

## ✅ LIME Explainability

LIME explains individual transaction predictions locally.

Shows:

* why a transaction was marked fraud
* which features influenced prediction

> Note:
> LIME explanations are more stable when multiple transactions are used.

---

# 🛠 Technologies Used

| Technology       | Purpose               |
| ---------------- | --------------------- |
| Python           | Core programming      |
| Pandas           | Data processing       |
| NumPy            | Numerical computation |
| Scikit-learn     | ML utilities          |
| XGBoost          | Fraud classification  |
| Isolation Forest | Anomaly detection     |
| SHAP             | Explainability        |
| LIME             | Local explainability  |
| Plotly           | Interactive charts    |
| Matplotlib       | Visualization         |
| Streamlit        | Dashboard UI          |
| Joblib           | Model serialization   |

---

# 📂 Project Structure

```bash
dbt-fraud-project/
│
├── app.py
├── xgboost_fraud_model.pkl
├── isolation_forest_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── requirements.txt
├── README.md
├── sample_csv/
│   ├── normal_transactions.csv
│   ├── fraud_transactions.csv
│   ├── mixed_transactions.csv
│   ├── realistic_transactions.csv
│   └── custom_transactions.csv
```

---


# 📁 Input Features

The model uses features such as:

* step
* amount
* oldbalanceOrg
* newbalanceOrig
* oldbalanceDest
* newbalanceDest
* hour
* transaction counts
* mean transaction amount
* transaction type
* engineered fraud indicators

---

# 🎯 ML Pipeline

## Data Processing

* cleaning
* encoding
* scaling
* feature engineering

---

## Models Used

### Isolation Forest

Used for anomaly detection.

### XGBoost

Used for fraud classification.

Optimized using:

* scale_pos_weight
* threshold tuning
* feature engineering

---

# 📌 Output

The system generates:

* fraud prediction
* fraud probability
* anomaly flag
* explainability graphs
* feature importance
* interactive visual analytics

---

# 🔐 Use Cases

* Banking fraud detection
* Digital payment monitoring
* DBT security systems
* Financial anomaly detection
* Real-time transaction analysis

---

# 👨‍💻 Author

* Ajeet Kumar
* Ashutosh Singh
* Shivam dahiphale
* Prateek D. Nagdeve
* Rajkumar Rajak

---

# 📜 License

This project is for educational and research purposes.

```
```
