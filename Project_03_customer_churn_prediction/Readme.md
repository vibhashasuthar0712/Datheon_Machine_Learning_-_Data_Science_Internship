# Telecom Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn using machine learning techniques on the IBM Telco Customer Churn dataset.

Customer churn refers to customers discontinuing their subscription or services. Predicting potential churn helps businesses identify customers who may leave and take proactive retention measures.

The objective of this project is to build and evaluate ensemble machine learning models, specifically **Random Forest and XGBoost**, to predict customer churn and identify the key factors influencing customer retention.

This project was completed as **Task 3 of the Datheon Machine Learning Internship**.

## Objectives

- Analyze telecom customer data to understand churn patterns.
- Perform data cleaning and preprocessing.
- Convert `TotalCharges` into a numerical format and handle missing values.
- Encode categorical variables using One-Hot Encoding.
- Handle class imbalance using class-weighting techniques.
- Train Random Forest and XGBoost classification models.
- Evaluate model performance using 5-fold stratified cross-validation.
- Compare models using classification metrics and ROC-AUC.
- Identify the top 10 influential features.
- Derive business insights to support customer retention strategies.

## Dataset

**Dataset:** IBM Telco Customer Churn

**Source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The dataset contains customer information related to:

- Customer demographics
- Subscription and contract details
- Internet and additional services
- Payment methods
- Monthly and total charges
- Customer tenure
- Churn status

### Dataset Information

| Description | Value |
|---|---:|
| Total Records | 7,043 |
| Original Features | 21 |
| Features After Removing Customer ID | 20 |
| Input Features | 19 |
| Target Variable | Churn |
| Training Data | 80% |
| Testing Data | 20% |

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Project Workflow

### 1. Data Loading and Understanding

- Loaded the IBM Telco Customer Churn dataset.
- Examined dataset dimensions, data types, and statistical summaries.
- Analyzed numerical and categorical variables.
- Checked missing values and duplicate records.

### 2. Data Cleaning and Preprocessing

- Removed the unnecessary `customerID` column.
- Converted `TotalCharges` from object to numerical format using `pd.to_numeric()`.
- Identified 11 missing values generated during numerical conversion.
- Replaced these missing values with 0, corresponding to customers with zero tenure.
- Verified that no missing values or duplicate records remained.

### 3. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand customer churn behavior.

Key areas investigated:

- Overall churn distribution.
- Churn across different contract types.
- Internet service and payment method patterns.
- Online security and technical support subscriptions.
- Paperless billing and senior citizen status.
- Relationship between customer tenure, monthly charges, total charges, and churn.

The analysis showed an overall customer churn rate of approximately **26.54%**.

### 4. Feature Preparation

- Separated independent variables (X) and the target variable (y).
- Used `Churn` as the target variable.
- Identified numerical and categorical features.
- Used One-Hot Encoding to transform categorical features into numerical representations.
- Applied a stratified 80:20 train-test split to preserve class distribution.

### 5. Handling Class Imbalance

The dataset contains more non-churn customers than churn customers.

To address this imbalance, class-weighting techniques were applied:

- **Random Forest:** Used `class_weight='balanced'`.
- **XGBoost:** Used `scale_pos_weight` based on the ratio of non-churn to churn customers in the training data.

This approach helps the models pay more attention to the minority churn class.

### 6. Machine Learning Models

Two ensemble classification models were trained and compared.

**Random Forest Classifier**

- Uses multiple decision trees.
- Combines predictions from individual trees.
- Supports class weighting to address class imbalance.
- Helps capture complex relationships between customer features.

**XGBoost Classifier**

- Uses gradient boosting to build trees sequentially.
- Learns from previous prediction errors.
- Supports class weighting through `scale_pos_weight`.
- Captures complex patterns and feature interactions.

### 7. Model Evaluation and Cross-Validation

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

Additionally, **5-fold Stratified Cross-Validation** was performed using ROC-AUC as the scoring metric.

Stratified cross-validation maintains the class distribution across folds and provides a more reliable estimate of model performance.

### 8. Feature Importance Analysis

Feature importance was extracted from both trained models to identify the variables contributing most to churn predictions.

The top 10 influential features from the XGBoost model were:

| Rank | Feature |
|---:|---|
| 1 | Contract - Month-to-month |
| 2 | Internet Service - Fiber optic |
| 3 | Online Security - No |
| 4 | Contract - Two year |
| 5 | Contract - One year |
| 6 | Internet Service - DSL |
| 7 | Streaming Movies - Yes |
| 8 | Tech Support - No |
| 9 | Payment Method - Electronic check |
| 10 | Internet Service - No |

These features help highlight important customer characteristics associated with churn predictions.

## Model Performance

### Cross-Validation Results

5-fold Stratified Cross-Validation using ROC-AUC:

| Model | Mean CV ROC-AUC |
|---|---:|
| Random Forest | 82.14% |
| XGBoost | 83.88% |

### Test Set Evaluation

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 78.57% | 76.58% |
| Precision | 62.86% | 54.20% |
| Recall | 47.06% | 75.94% |
| F1-Score | 53.82% | 63.25% |
| ROC-AUC | 82.24% | 83.95% |

### Best Performing Model

**XGBoost Classifier**

XGBoost achieved the highest test ROC-AUC score of **83.95%**, exceeding the internship target of 80%.

It also achieved a higher recall and F1-score compared to Random Forest.

Although Random Forest achieved higher accuracy and precision, XGBoost identified a larger proportion of actual churn customers.

Therefore, XGBoost was selected as the preferred model for this churn prediction task, particularly because identifying potential churn customers is an important business objective.

## Business Insights

The exploratory analysis revealed several useful patterns:

### 1. Contract Type

- Month-to-month customers had a churn rate of approximately 42.71%.
- One-year contract customers had a churn rate of approximately 11.27%.
- Two-year contract customers had a churn rate of approximately 2.83%.

**Insight:** Customers with shorter contract commitments may require stronger retention strategies.

### 2. Internet Service

- Fiber optic customers had a churn rate of approximately 41.89%.
- DSL customers had a churn rate of approximately 18.96%.
- Customers without internet service had a churn rate of approximately 7.40%.

**Insight:** The fiber optic customer segment deserves further investigation to understand service experience, pricing, and customer expectations.

### 3. Payment Method

- Electronic check customers had a churn rate of approximately 45.29%.
- Bank transfer (automatic) customers had a churn rate of approximately 16.71%.
- Credit card (automatic) customers had a churn rate of approximately 15.24%.

**Insight:** Electronic check users represent a higher-churn segment and may benefit from targeted engagement.

### 4. Customer Tenure

- Average tenure of non-churn customers: approximately 37.57 months.
- Average tenure of churn customers: approximately 17.98 months.

**Insight:** Early-tenure customers may need additional onboarding and engagement support.

### 5. Additional Services

Customers without Online Security and Tech Support showed higher churn rates than customers subscribed to these services.

**Insight:** Businesses can investigate whether service bundles and additional support options improve customer retention.

*Note: These findings represent observed associations in the dataset and do not establish direct causation.*

## Key Results

- Successfully processed 7,043 customer records.
- Removed unnecessary identifiers and handled missing values.
- Implemented a preprocessing pipeline using numerical imputation and One-Hot Encoding.
- Addressed class imbalance using class-weighting techniques.
- Trained and compared Random Forest and XGBoost classifiers.
- Performed 5-fold Stratified Cross-Validation.
- Achieved a test ROC-AUC score of 83.95% with XGBoost.
- Identified the top 10 influential features.
- Derived actionable business insights for customer retention.

## Project Structure

```text
Project_03_Telecom_Customer_Churn/
│
├── telecom_churn_prediction.ipynb
├── README.md
└── requirements.txt
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd Project_03_Telecom_Customer_Churn
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

Open `telecom_churn_prediction.ipynb` using Jupyter Notebook or JupyterLab and execute the cells sequentially.

The dataset is loaded directly from the provided IBM GitHub dataset URL.

<img width="698" height="501" alt="Screenshot (541)" src="https://github.com/user-attachments/assets/e316f14c-e7cb-4027-8480-ec7f676dde6e" />

<img width="997" height="410" alt="Screenshot (544)" src="https://github.com/user-attachments/assets/10b25bc2-541c-4253-89db-fcffccd7eba7" />



## Conclusion

This project demonstrates an end-to-end machine learning workflow for telecom customer churn prediction, including data cleaning, exploratory data analysis, preprocessing, class imbalance handling, ensemble model training, cross-validation, evaluation, and feature importance analysis.

Among the two evaluated models, XGBoost achieved the best ROC-AUC score of 83.95% on the test set.

The project also highlights how machine learning can help businesses identify customers at risk of leaving and develop more informed customer retention strategies.

---

**Internship:** Datheon – Machine Learning Internship  
**Task:** Task 3 – Telecom Customer Churn Prediction  
**Domain:** Machine Learning | Classification | Ensemble Learning
