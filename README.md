# Machine Learning Internship

## 📌 About This Repository

This repository contains all the tasks and projects completed as part of my **Machine Learning Internship at Datheon**.

Throughout this internship, I worked on practical machine learning problems to strengthen my understanding of data preprocessing, exploratory data analysis, feature engineering, model development, hyperparameter tuning, evaluation, and implementation using Python and its libraries.

The repository includes four projects covering different machine learning concepts, including classification, regression, ensemble learning, and predictive modeling.

## 🛠️ Technologies & Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Statsmodels
- XGBoost
- Jupyter Notebook

## 📚 Internship Projects

### Project 1: Iris Multiclass Classification

**Objective:** Build and evaluate supervised machine learning models to classify Iris flowers into three different species.

**Dataset:** Iris Dataset

**Key Tasks Performed:**

- Loaded and explored the Iris dataset using Pandas.
- Performed Exploratory Data Analysis (EDA) using Seaborn.
- Split the dataset into training and testing sets (80:20).
- Applied feature scaling using StandardScaler.
- Trained Logistic Regression and Decision Tree classifiers.
- Evaluated models using Accuracy, Precision, Recall, and F1-Score.
- Visualized confusion matrices and decision boundaries.

**Algorithms Used:**

- Logistic Regression
- Decision Tree Classifier

**Model Performance:**

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| Decision Tree | 93.33% |

**Key Learning:** Gained practical experience in multiclass classification, feature scaling, model training, performance evaluation, and data visualization.

---

### Project 2: California Housing Price Prediction

**Objective:** Predict median house values using the California Housing dataset by applying feature engineering, multicollinearity analysis, and regularized regression techniques.

**Dataset:** California Housing Dataset (20,640 records, 10 columns)

**Key Tasks Performed:**

- Loaded and explored the California Housing dataset.
- Performed Exploratory Data Analysis (EDA) and correlation heatmap visualization.
- Handled missing values using median imputation.
- Created interaction features:
  - Rooms per Household
  - Bedrooms per Room
  - Population per Household
- Applied One-Hot Encoding to categorical features.
- Analyzed and reduced multicollinearity using Variance Inflation Factor (VIF).
- Standardized numerical features using StandardScaler.
- Trained and compared OLS, Ridge, and Lasso regression models.
- Applied GridSearchCV for Ridge and Lasso hyperparameter tuning.
- Evaluated models using RMSE, MAE, and R² Score.
- Visualized actual vs predicted values and residual distributions.

**Algorithms Used:**

- Ordinary Least Squares (OLS) Regression
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)

**Model Performance:**

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| OLS | 70308.83 | 50594.98 | 0.6228 |
| Ridge | 70315.07 | 50600.56 | 0.6227 |
| Lasso | 70337.16 | 50587.86 | 0.6225 |

**Best Model:** OLS Regression (based on the lowest RMSE and highest R² Score).

**Key Learning:** Gained practical experience in regression modeling, interaction feature engineering, multicollinearity analysis, regularization techniques, hyperparameter optimization, and residual analysis.

---

### Project 3: Telecom Customer Churn Prediction

**Objective:** Build ensemble machine learning models to predict customer churn using telecom subscription records and identify important factors influencing customer retention.

**Dataset:** IBM Telco Customer Churn Dataset (7,043 records)

**Key Tasks Performed:**

- Loaded and explored the telecom customer churn dataset.
- Performed data cleaning and Exploratory Data Analysis (EDA).
- Removed unnecessary customer identifiers.
- Converted `TotalCharges` into numerical format and handled missing values.
- Applied One-Hot Encoding to categorical features.
- Built preprocessing pipelines using Scikit-learn.
- Addressed class imbalance using class-weighting techniques.
- Trained and compared Random Forest and XGBoost classifiers.
- Applied 5-fold Stratified Cross-Validation using ROC-AUC.
- Evaluated models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- Visualized ROC curves and confusion matrices.
- Extracted and analyzed the top 10 influential features.
- Derived business insights to support customer retention strategies.

**Algorithms Used:**

- Random Forest Classifier
- XGBoost Classifier

**Model Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 78.57% | 62.86% | 47.06% | 53.82% | 82.24% |
| XGBoost | 76.58% | 54.20% | 75.94% | 63.25% | 83.95% |

**Cross-Validation Results:**

| Model | Mean CV ROC-AUC |
|---|---:|
| Random Forest | 82.14% |
| XGBoost | 83.88% |

**Best Model:** XGBoost Classifier (based on the highest test ROC-AUC and F1-Score).

<img width="698" height="501" alt="Screenshot (541)" src="https://github.com/user-attachments/assets/5cb534cf-4066-47dd-a8fc-433816ad653d" />

<img width="997" height="410" alt="Screenshot (544)" src="https://github.com/user-attachments/assets/1774bb50-50e2-4495-a4f9-0d2ab5ae0ec1" />

<img width="909" height="544" alt="Screenshot (543)" src="https://github.com/user-attachments/assets/09d7c4d2-7268-41a4-8497-cd40093ee5f4" />

<img width="901" height="533" alt="Screenshot (542)" src="https://github.com/user-attachments/assets/5568f231-a368-4155-b925-09f58f5764c8" />

**Key Learning:** Gained practical experience in ensemble learning, imbalanced classification, preprocessing pipelines, stratified cross-validation, ROC-AUC evaluation, feature importance analysis, and business-oriented machine learning insights.

---

x

All internship projects have been completed.

## 📂 Repository Structure

```text
Machine-Learning-Internship/
│
├── Project_01_Iris_Classification/
│   ├── Iris_Multiclass_Benchmark.ipynb
│   ├── multiclass_benchmark.py
│   ├── ml_project/
│   │   └── iris.csv
│   └── README.md
│
├── Project_02_California_Housing/
│   ├── California_Housing_Price_Prediction.ipynb
│   ├── housing.csv
│   └── README.md
│
├── Project_03_Telecom_Customer_Churn/
│   ├── telecom_churn_prediction.ipynb
│   ├── requirements.txt
│   └── README.md
│
└── README.md
```

## 🎯 Key Skills Developed

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Scaling and Encoding
- Regression and Classification
- Ensemble Machine Learning
- Handling Imbalanced Datasets
- Cross-Validation
- Hyperparameter Tuning
- Model Evaluation and Comparison
- Feature Importance Analysis
- Data Visualization
- Business Insight Generation

## 👩‍💻 Author

**Vibhasha Suthar**

- GitHub: [github.com/vibhashasuthar0712](https://github.com/vibhashasuthar0712)
- LinkedIn: [linkedin.com/in/vibhasha-suthar-280b51351](https://www.linkedin.com/in/vibhasha-suthar-280b51351/)

---

**Internship:** Datheon – Machine Learning Internship  

*This repository documents my practical learning journey and hands-on experience with machine learning concepts and real-world datasets.*
