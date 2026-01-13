import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Based Customer Churn Prediction",
    layout="wide"
)

st.title("AI-Based Customer Churn Prediction System")
st.caption(
    "Predict → Explain → Prioritize → Recommend"
)

# -------------------------------------------------
# Load Model Artifacts
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_final.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# -------------------------------------------------
# Sidebar – Customer Input
# -------------------------------------------------
st.sidebar.header("Customer Details")

def get_user_input():
    data = {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.sidebar.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        ),
        "InternetService": st.sidebar.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        ),
        "OnlineSecurity": st.sidebar.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        ),
        "OnlineBackup": st.sidebar.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        ),
        "DeviceProtection": st.sidebar.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        ),
        "TechSupport": st.sidebar.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        ),
        "StreamingTV": st.sidebar.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        ),
        "StreamingMovies": st.sidebar.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        ),
        "Contract": st.sidebar.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        ),
        "PaperlessBilling": st.sidebar.selectbox(
            "Paperless Billing", ["Yes", "No"]
        ),
        "PaymentMethod": st.sidebar.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
        ),
        "MonthlyCharges": st.sidebar.number_input(
            "Monthly Charges", 0.0, 200.0, 70.0
        ),
        "TotalCharges": st.sidebar.number_input(
            "Total Charges", 0.0, 10000.0, 1500.0
        ),
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# -------------------------------------------------
# Preprocessing (Feature-Safe)
# -------------------------------------------------
def preprocess_input(df):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df_encoded)
    return df_scaled

X_user = preprocess_input(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
churn_probability = model.predict_proba(X_user)[0][1]
churn_prediction = int(churn_probability >= 0.5)

# -------------------------------------------------
# Display Prediction
# -------------------------------------------------
st.subheader("Churn Prediction Result")

if churn_prediction == 1:
    st.error(
        f"High Risk of Churn\n\n"
        f"Churn Probability: **{churn_probability:.2%}**"
    )
else:
    st.success(
        f"Low Risk of Churn\n\n"
        f"Retention Probability: **{1 - churn_probability:.2%}**"
    )

# -------------------------------------------------
# Revenue Loss Prioritization
# -------------------------------------------------
st.subheader("Revenue Impact Analysis")

ASSUMED_REMAINING_MONTHS = 12
potential_loss = (
    churn_probability
    * input_df["MonthlyCharges"].values[0]
    * ASSUMED_REMAINING_MONTHS
)

st.metric(
    label="Estimated Revenue at Risk",
    value=f"₹{potential_loss:,.2f}"
)

# -------------------------------------------------
# Explainability – Top Drivers
# -------------------------------------------------
st.subheader("Top Churn Risk Drivers")

coef_series = pd.Series(
    model.coef_[0], index=feature_names
).sort_values(key=abs, ascending=False)

top_drivers = coef_series.head(5)

explain_df = pd.DataFrame({
    "Feature": top_drivers.index,
    "Impact Score": top_drivers.values
})

st.dataframe(explain_df, use_container_width=True)

# -------------------------------------------------
# Retention Action Recommendation (Dynamic)
# -------------------------------------------------
st.subheader("AI-Recommended Retention Action")

def retention_action_dynamic(prob, monthly_charge, tenure, contract, services):
    """
    Returns a customized retention recommendation based on multiple inputs.
    prob          : churn probability
    monthly_charge: user's monthly bill
    tenure        : months customer has been active
    contract      : type of contract
    services      : dict of additional service usage
    """
    action_list = []

    # High churn probability
    if prob > 0.8:
        action_list.append("Immediate retention call + onboarding support")
    elif prob > 0.6:
        action_list.append("Proactive customer support outreach")
    elif prob > 0.4:
        action_list.append("Engage via email/SMS campaigns")

    # Revenue-based action
    if monthly_charge > 100:
        action_list.append("Offer premium discount or loyalty pricing")
    elif monthly_charge > 50:
        action_list.append("Offer minor incentives or bundle services")

    # Tenure-based action
    if tenure < 6:
        action_list.append("Onboarding support and welcome offers")
    elif tenure > 36:
        action_list.append("VIP loyalty program or renewal bonus")

    # Contract type
    if contract == "Month-to-month":
        action_list.append("Encourage long-term contract for retention")

    # Services usage
    if services.get("StreamingTV") == "Yes" or services.get("StreamingMovies") == "Yes":
        action_list.append("Upsell entertainment add-ons")
    if services.get("OnlineSecurity") == "No":
        action_list.append("Offer security add-on for peace of mind")

    # Return combined unique recommendations
    return "\n• " + "\n• ".join(list(dict.fromkeys(action_list)))

services_dict = {
    "StreamingTV": input_df["StreamingTV"].values[0],
    "StreamingMovies": input_df["StreamingMovies"].values[0],
    "OnlineSecurity": input_df["OnlineSecurity"].values[0]
}

action_dynamic = retention_action_dynamic(
    churn_probability,
    input_df["MonthlyCharges"].values[0],
    input_df["tenure"].values[0],
    input_df["Contract"].values[0],
    services_dict
)

st.success(action_dynamic)


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "This system focuses on explainability, business impact, and decision support — ")
