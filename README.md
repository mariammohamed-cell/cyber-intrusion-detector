# CyberShield IDS — Traffic Intelligence Console

CyberShield IDS is an interactive Intrusion Detection System (IDS) web application built using Streamlit. It leverages a pre-trained Machine Learning model (Random Forest) to evaluate real-time network session metrics, score potential vulnerabilities, and expose analytical model insights.

---

## 🚀 System Architecture & Layout

The interface is structured dynamically into two dedicated control planes manageable from the sidebar:

### 1. Prediction Lab
* **Real-time Session Grading:** An input form accepting live network traffic vectors (such as packet size, protocol types, login attempts, encryption status, and IP reputation).
* **Automated Risk Profiling:** Instantly returns an administrative verdict (`Intrusion Detected` vs. `Normal Traffic`), an exact attack probability score ($0\%$ to $100\%$), and dynamic visual warning flags based on critical thresholds.

### 2. Model Insights
* **Cross-Model Performance Benchmarking:** A structured overview comparing the core Random Forest engine metrics against alternative baseline classifiers (SVM, Decision Trees, KNN, etc.).
* **Feature Importance Charting:** Renders an interactive Altair-backed bar chart pinpointing exactly which network markers heavily drive the model's risk decisions.
* **Dataset Consistency Audits:** Runs an automated matrix check on the default dataset to verify structural agreement, attack recall rates, and total false alert volumes.

---

## 🛠️ Prerequisites & Dependencies

To ensure clean processing pipelines (specifically log transformations and feature normalization components), install the following explicit package versions:

```bash
pip install numpy pandas streamlit altair joblib scikit-learn
