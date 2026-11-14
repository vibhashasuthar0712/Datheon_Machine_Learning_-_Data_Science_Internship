# Iris Multiclass Classification

## Project Overview

This project focuses on building and evaluating machine learning models for multiclass classification using the Iris dataset.

The objective is to classify iris flowers into three species — **Setosa, Versicolor, and Virginica** — based on their sepal and petal measurements.

Two classification algorithms, Logistic Regression and Decision Tree, are implemented and compared to understand their performance on the dataset.

## Objectives

- Understand and explore the Iris dataset through Exploratory Data Analysis (EDA).
- Prepare the dataset for machine learning.
- Split the data into training and testing sets.
- Apply feature scaling using StandardScaler.
- Train and evaluate multiple classification models.
- Compare model performance using evaluation metrics.
- Visualize classification results using plots and confusion matrices.

## Dataset

The Iris dataset contains 150 observations and 5 columns.

**Features:**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Target Variable:**
- Species

**Target Classes:**
- Setosa
- Versicolor
- Virginica

## Technologies & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

### 1. Data Loading and Exploration
- Loaded the Iris dataset using Pandas.
- Checked dataset dimensions, column information, and statistical summaries.
- Explored feature distributions and relationships using visualizations.

### 2. Data Preprocessing
- Separated independent features and the target variable.
- Divided the dataset into training and testing sets.
- Used an 80:20 split with stratified sampling.
- Applied StandardScaler for feature scaling.

### 3. Model Building

Implemented the following machine learning algorithms:

**Logistic Regression**
- Used for multiclass classification.
- Learns relationships between input features and flower species.

**Decision Tree Classifier**
- Uses decision rules based on feature values.
- Provides an interpretable classification approach.

### 4. Model Evaluation

Evaluated both models using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 5. Data Visualization

Created visualizations to understand model performance and dataset patterns:

- Pairplot for feature relationships.
- Confusion matrix heatmaps.
- Decision boundary visualization using two selected features.

## Model Performance

Both models achieved the same accuracy on the test dataset.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| Decision Tree | 93.33% |

**Test Dataset:** 30 samples

Both models correctly classified 28 out of 30 test samples.

## Key Learnings

Through this project, I gained practical experience in:

- Exploratory Data Analysis (EDA).
- Data preprocessing and feature scaling.
- Stratified train-test splitting.
- Implementing multiclass classification algorithms.
- Evaluating models using multiple performance metrics.
- Visualizing classification results.
- Comparing machine learning algorithms.

<img width="998" height="422" alt="Screenshot (546)" src="https://github.com/user-attachments/assets/7de99cf8-edbd-4549-a943-dfc9951283bb" />
<img width="997" height="419" alt="Screenshot (547)" src="https://github.com/user-attachments/assets/53b752c6-3ca2-431b-b05b-b678bba81426" />


## Conclusion

Both Logistic Regression and Decision Tree achieved an accuracy of **93.33%** on the test dataset.

This project provided hands-on experience with the complete machine learning classification workflow, from data exploration and preprocessing to model training, evaluation, and visualization.

## Author

**Vibhasha Suthar**

- GitHub: https://github.com/vibhashasuthar0712
- LinkedIn: https://www.linkedin.com/in/vibhasha-suthar-280b51351/
