# California Housing Price Prediction

## Project Overview

This project focuses on predicting median house values using the California Housing dataset through machine learning regression techniques.

The main objective is to build and compare different regression models while applying data preprocessing, feature engineering, missing value imputation, multicollinearity analysis, and hyperparameter tuning.

Three regression techniques — **Ordinary Least Squares (OLS), Ridge Regression, and Lasso Regression** — are implemented and evaluated to identify the best-performing model.

## Objectives

- Explore and understand the California Housing dataset.
- Handle missing values using appropriate imputation techniques.
- Perform feature engineering to create meaningful interaction features.
- Analyze correlations and handle multicollinearity using VIF.
- Encode categorical variables and standardize numerical features.
- Train OLS, Ridge, and Lasso regression models.
- Perform hyperparameter tuning using GridSearchCV.
- Compare models using RMSE, MAE, and R² scores.
- Visualize actual vs. predicted values and residuals.

## Dataset

The project uses the California Housing dataset.

**Dataset Information:**
- Total Records: 20,640
- Total Original Columns: 10
- Target Variable: `median_house_value`

**Features:**
- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- `ocean_proximity`

**Target Variable:**
- `median_house_value`

## Technologies & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Jupyter Notebook

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the California Housing dataset using Pandas.
- Examined dataset dimensions, data types, and statistical summaries.
- Checked missing values and explored feature distributions.
- Analyzed relationships between numerical features and the target variable using correlation analysis and heatmaps.

### 2. Handling Missing Values

- Identified 207 missing values in the `total_bedrooms` column.
- Used median imputation to handle missing values.
- Verified that no missing values remained after imputation.

### 3. Feature Engineering

Created additional features to capture useful relationships between existing variables:

- `rooms_per_household`: Average number of rooms per household.
- `bedrooms_per_room`: Proportion of bedrooms relative to total rooms.
- `population_per_household`: Average population per household.

These features help represent household characteristics more meaningfully.

### 4. Multicollinearity Analysis

- Used Variance Inflation Factor (VIF) to identify multicollinearity among numerical features.
- Iteratively removed features with high VIF values.
- Retained features with VIF values below the selected threshold of 10.

This step helped reduce redundancy among input variables.

### 5. Data Preprocessing

Applied different preprocessing techniques to numerical and categorical features.

**Numerical Features:**
- Missing value imputation.
- Standardization using StandardScaler.

**Categorical Features:**
- Missing value imputation.
- One-Hot Encoding using OneHotEncoder.

Used ColumnTransformer and Pipeline to organize preprocessing and model training.

### 6. Model Building

Implemented three regression algorithms:

**Ordinary Least Squares (OLS)**
- Used as the baseline linear regression model.
- Estimates relationships between independent variables and house prices.

**Ridge Regression (L2 Regularization)**
- Applies L2 regularization to reduce the impact of large coefficients.
- Helps manage multicollinearity and control model complexity.

**Lasso Regression (L1 Regularization)**
- Applies L1 regularization.
- Can reduce some coefficients to zero, helping with feature selection.

### 7. Hyperparameter Tuning

- Used GridSearchCV to find suitable regularization parameters.
- Tuned Ridge and Lasso regression models.
- Selected the best hyperparameters based on cross-validation RMSE.

**Best Parameters:**
- Ridge Alpha: 1
- Lasso Alpha: 0.001

### 8. Model Evaluation

Evaluated all three models using the following regression metrics:

- **RMSE (Root Mean Squared Error):** Measures the square root of average squared prediction errors.
- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **R² Score:** Measures how much variation in the target variable is explained by the model.

## Model Performance

The models were evaluated on the test dataset.

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| OLS | 70,308.83 | 50,594.98 | 0.6228 |
| Ridge | 70,315.07 | 50,600.56 | 0.6227 |
| Lasso | 70,337.16 | 50,587.86 | 0.6225 |

**Best Results:**
- Lowest RMSE: OLS Regression
- Lowest MAE: Lasso Regression
- Highest R² Score: OLS Regression

OLS achieved the best overall performance based on RMSE and R² score, while Lasso achieved the lowest MAE.

## Data Visualization

Created visualizations to analyze model performance and prediction errors:

- Correlation heatmap.
- VIF analysis.
- Actual vs. Predicted values plots.
- Residual distribution plots.
- Residual scatter plots.

These visualizations helped evaluate how closely the predictions matched actual house values and understand the distribution of prediction errors.

## Key Learnings

Through this project, I gained practical experience in:

- Regression-based machine learning.
- Exploratory Data Analysis (EDA).
- Missing value imputation.
- Feature engineering and interaction features.
- Multicollinearity analysis using VIF.
- Categorical encoding and feature scaling.
- OLS, Ridge, and Lasso regression.
- Hyperparameter tuning using GridSearchCV.
- Model evaluation using RMSE, MAE, and R².
- Residual analysis and model comparison.

<img width="986" height="297" alt="Screenshot (545)" src="https://github.com/user-attachments/assets/63a0d42e-c11f-4541-9dd3-689230b85a7b" />


## Conclusion

This project demonstrates a complete regression workflow, starting from data exploration and preprocessing to model training, hyperparameter tuning, and evaluation.

Among the three models, OLS Regression achieved the lowest RMSE and highest R² score, making it the best-performing model overall on the test dataset.

The results show that the selected features explain approximately **62.28% of the variation** in median house values on the test dataset.

## Author

**Vibhasha Suthar**

- GitHub: https://github.com/vibhashasuthar0712
- LinkedIn: https://www.linkedin.com/in/vibhasha-suthar-280b51351/
