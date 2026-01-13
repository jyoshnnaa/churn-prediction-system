# AI-Based Customer Churn Prediction System

## Project Overview
This project implements an **AI-based customer churn prediction system** for subscription-based businesses (e.g., telecom companies).  
The system predicts whether a customer is likely to churn and provides:

- **Churn probability**
- **Explainable AI insights** (feature contributions)
- **Business recommendations** to reduce churn

It is deployed as a **Streamlit web application** for professional usability.

---

## Features

- Predict if a customer is likely to churn
- Shows **churn probability**
- Visualizes **top features influencing prediction** using SHAP
- Generates **AI-driven business recommendations**
- Professional Streamlit web interface
- Ready for deployment and cross-browser compatible

---

## Dataset

- **IBM Watson Telco Customer Churn Dataset** (publicly available)
- Contains:
  - Customer demographics (gender, age, etc.)
  - Account info (tenure, monthly charges, contract type)
  - Service subscriptions
  - Churn labels (Yes/No)
- Features include both **numerical and categorical variables**

---

## Folder Structure
AI_Customer_Churn_project/
├── app/ # Streamlit web application
│ └── app.py
├── models/ # Trained models and scaler
│ ├── logistic_regression_final.pkl
│ └── scaler.pkl
├── data/ # Dataset files
│ └── telco_churn.csv
├── notebooks/ # Development notebooks
│ └── churn_system_development.ipynb
├── src/ # Optional preprocessing/predict modules
├── requirements.txt # Python dependencies
└── README.md # Project documentation

Results

The AI system successfully predicts churn and provides interpretability.

Key churn drivers identified include:

Contract type (Month-to-Month vs. Annual)

Tenure

Monthly charges

Enables targeted retention strategies for high-risk customers.

Future Scope

Integrate more ML models (Random Forest, XGBoost, Neural Networks)

Real-time churn prediction with streaming data

Automated retention campaigns based on predictions

Deploy in production environments for telecom, SaaS, and streaming services

References

IBM Watson Analytics, Telco Customer Churn Dataset

Scikit-learn Documentation

Géron, A. Hands-On Machine Learning with Scikit-Learn and TensorFlow

Molnar, C. Interpretable Machine Learning

Author

Jyoshna Maraboyina
Email: jyoshna.maraboyina24@gmail.com

LinkedIn/GitHub 
