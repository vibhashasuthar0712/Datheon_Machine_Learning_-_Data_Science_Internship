
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Classification AI",
    page_icon="🌸",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef7ff);
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #26345c;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #65708b;
    font-size: 16px;
    margin-bottom: 30px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    background-color: #536dfe;
    color: white;
    border: none;
    padding: 12px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #3549c9;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("iris.csv")

df = load_data()

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

X = df[features]
y = df["species"]

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

@st.cache_resource
def train_models(X_train_scaled, y_train):
    logistic_model = LogisticRegression(max_iter=1000)
    tree_model = DecisionTreeClassifier(random_state=42)

    logistic_model.fit(X_train_scaled, y_train)
    tree_model.fit(X_train, y_train)

    return logistic_model, tree_model

logistic_model, tree_model = train_models(
    X_train_scaled,
    y_train
)

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

logistic_predictions = logistic_model.predict(X_test_scaled)
tree_predictions = tree_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌸 Iris Classification AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Iris Flower Species Prediction</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌸 Navigation")

page = st.sidebar.radio(
    "Choose Section",
    ["Flower Prediction", "Model Performance", "Dataset Overview"]
)

# --------------------------------------------------
# PAGE 1: FLOWER PREDICTION
# --------------------------------------------------

if page == "Flower Prediction":

    st.header("🔍 Predict Iris Flower Species")

    st.write(
        "Enter the flower measurements below to predict its species."
    )

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.1,
            max_value=10.0,
            value=5.1,
            step=0.1
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.1,
            max_value=10.0,
            value=3.5,
            step=0.1
        )

    with col2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.1,
            max_value=10.0,
            value=1.4,
            step=0.1
        )

        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=10.0,
            value=0.2,
            step=0.1
        )

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        ["Logistic Regression", "Decision Tree"]
    )

    if st.button("Predict Flower Species"):

        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=features
        )

        if selected_model == "Logistic Regression":

            input_scaled = scaler.transform(input_data)

            prediction = logistic_model.predict(
                input_scaled
            )[0]

            probabilities = logistic_model.predict_proba(
                input_scaled
            )[0]

        else:

            prediction = tree_model.predict(
                input_data
            )[0]

            probabilities = tree_model.predict_proba(
                input_data
            )[0]

        st.markdown("---")
        st.subheader("🎉 Prediction Result")

        st.success(
            f"Predicted Iris Species: {prediction.capitalize()}"
        )

        st.write("### Prediction Confidence")

        probability_df = pd.DataFrame({
            "Species": tree_model.classes_,
            "Confidence": probabilities
        })

        for _, row in probability_df.iterrows():

            st.write(
                f"{row['Species'].capitalize()}: "
                f"{row['Confidence']:.2%}"
            )

            st.progress(float(row["Confidence"]))

        st.caption(
            "Prediction generated using the selected machine learning model."
        )

# --------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("📊 Model Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Logistic Regression Accuracy",
            f"{logistic_accuracy:.2%}"
        )

    with col2:
        st.metric(
            "Decision Tree Accuracy",
            f"{tree_accuracy:.2%}"
        )

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree"],
        "Accuracy": [
            logistic_accuracy,
            tree_accuracy
        ]
    })

    st.subheader("Accuracy Comparison")

    st.bar_chart(
        comparison.set_index("Model")
    )

    st.subheader("Classification Report")

    report_model = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree"]
    )

    if report_model == "Logistic Regression":
        predictions = logistic_predictions
    else:
        predictions = tree_predictions

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

# --------------------------------------------------
# PAGE 3: DATASET OVERVIEW
# --------------------------------------------------

elif page == "Dataset Overview":

    st.header("📁 Iris Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Total Features", len(features))

    with col3:
        st.metric("Flower Species", df["species"].nunique())

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Species Distribution")

    st.bar_chart(
        df["species"].value_counts()
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "**Iris Classification AI** · Multiclass Classification"
)

st.caption(
    "Developed as part of Datheon Machine Learning Internship"
)
