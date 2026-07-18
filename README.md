# CyberShield IDS — Traffic Intelligence Console

CyberShield IDS is an interactive Intrusion Detection System (IDS) web application built using Streamlit. It leverages a pre-trained Random Forest engine selected through rigorous baseline benchmarking to evaluate network session metrics and deliver real-time traffic intelligence.

---

## 🚀 Model Selection & Machine Learning Lifecycle

The core classification architecture is built on an empirical evaluation of various tabular modeling approaches. The application includes a comparison matrix derived during model selection to validate why the Random Forest model was chosen.

### 📊 Comparative Architecture Benchmark
During training, multiple algorithms were cross-evaluated on the attack vectors using identical evaluation splits. The metrics demonstrate clear functional differences across methods:

| Classifier Model | Inference Accuracy | Prediction Precision | Attack Recall Rate | Balanced F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **88.57%** | **100.00%** | **74.44%** | **85.35%** |
| Decision Tree | 88.42% | 100.00% | 74.09% | 85.12% |
| Support Vector Machine (SVM) | 86.79% | 95.32% | 74.09% | 83.38% |
| Naive Bayes | 81.29% | 88.87% | 66.47% | 76.06% |
| k-Nearest Neighbors (KNN) | 79.40% | 89.12% | 61.43% | 72.73% |
| Logistic Regression | 75.89% | 75.69% | 67.88% | 71.57% |

### 🔍 Why Random Forest Was Chosen
* **Zero False Alerts Protection:** The model achieved a perfect **100.00% Precision** score on the attack class during testing. In production environments, this prevents alert fatigue by ensuring that normal business traffic is never flagged as an intrusion.
* **Top-Tier F1-Score:** At **85.35%**, it provides the best mathematical balance between capturing real threats (Recall) and ensuring clean classifications (Precision).
* **Feature Interaction Handling:** The tree ensemble effectively handles multi-variable patterns—such as engineering a dynamic `failed_login_ratio` or applying logarithmic compression to skewed values like `session_duration`—without losing predictive stability.

---

## 🛠️ Prerequisites & Dependencies

To guarantee identical handling across the log-transformation steps and random forest estimators, align your workspace environment with these package dependencies:

```bash
pip install numpy pandas streamlit altair joblib scikit-learn
