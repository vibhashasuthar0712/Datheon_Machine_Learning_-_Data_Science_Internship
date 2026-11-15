
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="California Home Value",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. GLASSMORPHISM CSS
# ============================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, #dceafe 0%, transparent 35%),
        radial-gradient(circle at 90% 80%, #e8ddff 0%, transparent 35%),
        linear-gradient(135deg, #f7f9ff, #eef2ff);
    color: #1e293b;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.8);
}

/* Header */
.hero {
    padding: 32px;
    border-radius: 24px;
    background: rgba(255,255,255,0.58);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.85);
    box-shadow: 0 8px 32px rgba(31,38,135,0.08);
    margin-bottom: 28px;
}

.hero h1 {
    color: #172554;
    font-size: 35px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 0;
}

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.62);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 30px rgba(31,38,135,0.07);
    margin-bottom: 20px;
}

/* Section headings */
.section-heading {
    color: #1e3a8a;
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 18px;
}

/* Input labels */
label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Input fields */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid #dbe5f5 !important;
    border-radius: 11px !important;
}

/* Focus styling */
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.15);
}

/* Primary button */
div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    background: linear-gradient(100deg, #4f46e5, #7c3aed);
    box-shadow: 0 6px 18px rgba(99,102,241,0.25);
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(99,102,241,0.35);
}

/* Prediction result */
.result-card {
    background: linear-gradient(
        120deg,
        rgba(224,231,255,0.9),
        rgba(243,232,255,0.9)
    );
    border: 1px solid rgba(255,255,255,0.95);
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 35px rgba(79,70,229,0.10);
    margin-top: 24px;
}

.result-label {
    color: #475569;
    font-size: 16px;
    font-weight: 600;
}

.result-price {
    color: #4338ca;
    font-size: 38px;
    font-weight: 850;
    margin: 12px 0;
}

.result-note {
    color: #64748b;
    font-size: 13px;
}

/* Small text */
.muted {
    color: #64748b;
    font-size: 13px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 25px 0 5px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("best_california_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    data["rooms_per_household"] = (
        data["total_rooms"] /
        data["households"].replace(0, np.nan)
    )

    data["bedrooms_per_room"] = (
        data["total_bedrooms"] /
        data["total_rooms"].replace(0, np.nan)
    )

    data["population_per_household"] = (
        data["population"] /
        data["households"].replace(0, np.nan)
    )

    return data


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏡 HomeValue AI")

    st.caption("California Property Intelligence")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Price Prediction",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### ✨ Model Information")

    st.markdown("""
    **Dataset:** California Housing

    **Approach:** Regression

    **Models:** OLS, Ridge & Lasso

    **Selection:** Best evaluated model
    """)

    st.divider()

    st.markdown(
        '<p class="muted">Datheon ML Internship · Task 2</p>',
        unsafe_allow_html=True
    )


# ============================================================
# 6. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>California Home Value Predictor 🏡</h1>
    <p>
        Discover estimated property values using machine learning.
        Enter the property details below to generate your prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 7. PRICE PREDICTION PAGE
# ============================================================

if page == "Price Prediction":

    st.markdown(
        '<div class="section-heading">📍 Property Information</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # LOCATION DETAILS
    # --------------------------------------------------------

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### 📍 Location Details")

    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-113.0,
            value=-119.0,
            step=0.1,
            help="Geographical longitude of the property."
        )

    with col2:
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=43.0,
            value=36.0,
            step=0.1,
            help="Geographical latitude of the property."
        )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN"
        ]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # HOUSE DETAILS
    # --------------------------------------------------------

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### 🏠 House Details")

    col3, col4 = st.columns(2)

    with col3:

        housing_median_age = st.number_input(
            "Housing Median Age",
            min_value=1.0,
            max_value=52.0,
            value=20.0,
            step=1.0
        )

        total_rooms = st.number_input(
            "Total Rooms",
            min_value=1.0,
            value=2000.0,
            step=100.0
        )

        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=1.0,
            value=400.0,
            step=10.0
        )

    with col4:

        population = st.number_input(
            "Population",
            min_value=1.0,
            value=1000.0,
            step=50.0
        )

        households = st.number_input(
            "Households",
            min_value=1.0,
            value=350.0,
            step=10.0
        )

        median_income = st.number_input(
            "Median Income",
            min_value=0.0,
            max_value=20.0,
            value=3.5,
            step=0.1,
            help="Median income expressed in tens of thousands of dollars."
        )

    st.markdown(
        '<p class="muted">Enter the median income in the dataset scale. '
        'For example, 3.5 represents approximately $35,000.</p>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.write("")

    predict = st.button(
        "✨ Generate House Price Prediction"
    )

    if predict:

        # Build input data
        input_data = pd.DataFrame({

            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity]

        })

        # Apply the same feature engineering
        input_data = create_features(input_data)

        try:
            with st.spinner("Analyzing property details..."):
                prediction = model.predict(input_data)[0]

            st.markdown("### 🎉 Your Prediction Result")

            st.metric(
                label="Estimated Median House Value",
                value=f"${prediction:,.2f}"
            )

            st.success("Your house price prediction is ready!")

            st.caption(
                "This is a machine learning based estimated value, "
                "not a guaranteed market sale price."
            )

            st.markdown("### 📋 Property Summary")

            summary1, summary2, summary3 = st.columns(3)

            summary1.metric("Median Income", f"{median_income:.2f}")
            summary2.metric("Total Rooms", f"{total_rooms:,.0f}")
            summary3.metric("Total Bedrooms", f"{total_bedrooms:,.0f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# 8. ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    st.markdown("## 📊 About This Project")

    st.markdown("""
    <div class="glass-card">

    ### California Housing Price Prediction

    This project uses machine learning regression techniques
    to estimate median house values based on California housing
    dataset features.

    **Project workflow:**

    - Exploratory Data Analysis
    - Missing Value Imputation
    - Feature Engineering
    - Multicollinearity Analysis using VIF
    - Numerical Feature Scaling
    - Categorical Feature Encoding
    - Regression Model Training
    - Hyperparameter Tuning
    - Model Evaluation and Comparison

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Models Evaluated")

    model_col1, model_col2, model_col3 = st.columns(3)

    model_col1.markdown("""
    <div class="glass-card">
        <h3>OLS</h3>
        <p class="muted">Ordinary Least Squares</p>
    </div>
    """, unsafe_allow_html=True)

    model_col2.markdown("""
    <div class="glass-card">
        <h3>Ridge</h3>
        <p class="muted">L2 Regularization</p>
    </div>
    """, unsafe_allow_html=True)

    model_col3.markdown("""
    <div class="glass-card">
        <h3>Lasso</h3>
        <p class="muted">L1 Regularization</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "The application uses the saved best-performing model "
        "selected during notebook evaluation."
    )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.markdown(
    "**HomeValue AI** · California Housing Price Prediction"
)

st.caption(
    "Developed as part of Datheon Machine Learning Internship"
)