# Machine Learning Internship

## 📌 About This Repository

This repository contains the tasks and projects completed as part of my **Machine Learning Internship**.

Throughout this internship, I am working on practical machine learning problems to strengthen my understanding of data preprocessing, feature engineering, model development, hyperparameter tuning, evaluation, and implementation using Python and its libraries.

## 🛠️ Technologies & Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Statsmodels
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

### Project 3: Coming Soon

### Project 4: Coming Soon

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
├── Project_03/
├── Project_04/
│
└── README.md
```

## 👩‍💻 Author

**Vibhasha Suthar**

- GitHub: https://github.com/vibhashasuthar0712
- LinkedIn: https://www.linkedin.com/in/vibhasha-suthar-280b51351/

---

*This repository will be continuously updated as I progress through the remaining internship projects.*
