
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    .main-title {
        font-size: 34px;
        font-weight: 750;
        color: #172554;
    }

    .subtitle {
        color: #64748B;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        color: #1E293B;
        font-size: 21px;
        font-weight: 700;
    }

    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 9px;
        border: none;
        height: 48px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "best_churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()

except FileNotFoundError:
    st.error("Model file not found! Please add best_churn_model.pkl.")
    st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Telecom Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict customer churn using '
    'Machine Learning and XGBoost.</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 5. CUSTOMER INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-title">Customer Information</div>',
    unsafe_allow_html=True
)

st.write("Enter customer details to predict the possibility of churn.")

with st.form("customer_form"):

    # PERSONAL INFORMATION

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

    with col2:
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col3:
        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )

    col1, col2 = st.columns(2)

    with col1:
        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )

    with col2:
        tenure = st.slider(
            "Tenure (Months)", 0, 72, 12
        )

    st.divider()

    # SERVICE INFORMATION

    st.subheader("Service Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col3:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    col1, col2 = st.columns(2)

    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    # CONTRACT AND BILLING

    st.subheader("Contract & Billing Information")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.divider()

    # BILLING DETAILS

    st.subheader("Billing Details")

    col1, col2 = st.columns(2)

    with col1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn"
    )


# ============================================================
# 6. PREDICTION
# ============================================================

if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    
    try:
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        # Identify the positive class (Churn = Yes)
        churn_index = next(
            (
                i for i, c in enumerate(classes)
                if str(c).strip().lower() in ["yes", "1", "true"]
            ),
            None
        )

        if churn_index is None:
            st.error(f"Unexpected model classes: {classes}")
            st.stop()

        churn_probability = probabilities[churn_index] * 100

        # Convert prediction into readable labels
        prediction_label = (
            "Yes"
            if str(prediction).strip().lower() in ["yes", "1", "true"]
            else "No"
        )

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.2f}%"
            )
            st.progress(float(churn_probability / 100))

        with col2:
            st.metric(
                "Prediction",
                prediction_label
            )

        if prediction_label == "Yes":
            st.error("Customer is predicted to churn.")
            st.write(
                "This customer may require attention "
                "from the customer retention team."
            )
        else:
            st.success("Customer is predicted to stay.")
            st.write(
                "The model predicts that this customer "
                "is likely to continue the subscription."
            )

        st.caption(
            "The churn probability is a model-estimated likelihood, "
            "not a guarantee of future customer behavior."
        )

        with st.expander("View Customer Profile"):
            st.dataframe(
                input_data.T.rename(
                    columns={0: "Customer Details"}
                ),
                use_container_width=True
            )

        with st.expander("Important Churn Factors"):
            st.write("""
            The model's overall feature importance analysis
            identified these influential factors:

            - Month-to-month contracts
            - Fiber optic internet service
            - No online security
            - Contract duration
            - No technical support
            - Electronic check payment method

            These are overall model-level findings and are
            not individual explanations for this prediction.
            """)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    
