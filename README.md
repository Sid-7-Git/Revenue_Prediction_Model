# Revenue Prediction using Machine Learning

## Overview
This notebook demonstrates a machine learning workflow to predict revenue based on various financial factors. It covers data loading, exploratory data analysis (EDA), data preprocessing, and training multiple regression models including XGBoost, LightGBM, and a Linear Regression pipeline.

## Dataset
The project uses `Data.csv`, a simulated dataset containing the following columns:
*   `Marketing_Spend`: Amount spent on marketing.
*   `R&D_Spend`: Research and Development expenses.
*   `Administration_Costs`: General administration overheads.
*   `Number_of_Employees`: Total number of employees.
*   `Region`: Categorical variable indicating the operational region (e.g., North America, Asia, Europe).
*   `Revenue`: The target variable, representing the total revenue generated.

## Notebook Structure and Explanation

### 1. Data Loading and Initial Exploration (Cell 9GHiveZ1jPUv)
This section loads the `Data.csv` file into a Pandas DataFrame. It then performs initial checks to understand the dataset's structure, identify data types, non-null values, and summarizes descriptive statistics. It also checks for missing values and potential date/time columns.

### 2. Categorical Feature Encoding (Cell e9a33cba)
The `Region` column, being a categorical feature, is converted into numerical format using one-hot encoding with `pd.get_dummies()`. This creates new binary columns for each unique region, making the data suitable for machine learning algorithms.

### 3. Exploratory Data Analysis (EDA) - Revenue Distribution (Cells a42bcf6a, db7204ca, cfd3abbc, 31603a20, 6dc01681, ec3ad74f)
Visualizations such as histograms, KDE plots, box plots, and violin plots are used to understand the distribution of the `Revenue` target variable. These plots help in identifying the central tendency, spread, and any potential outliers in the revenue data.

### 4. Correlation Analysis (Cells 1bc73c8a, 5edf0d61)
A correlation matrix is calculated and visualized using a heatmap to show the linear relationships between all numerical features and the `Revenue` target. This helps in understanding which features are strongly correlated with revenue and with each other.

### 5. R-squared Values for Individual Factors (Cells 4a17c15d, c6ca1bf5)
The R-squared value for each individual feature's relationship with `Revenue` is calculated and visualized. This metric quantifies the proportion of variance in the `Revenue` that can be explained by each factor alone.

### 6. Data Preparation for Machine Learning (Cells 3bec6a01, 110ba2e6)
The dataset is split into training and testing sets (80% training, 20% testing) using `train_test_split` from `sklearn.model_selection`. This step is crucial for evaluating the model's ability to generalize to unseen data. Features (`X`) and the target variable (`y`) are defined.

### 7. XGBoost Regressor Model (Cells eadec739, f9cb0373)
An XGBoost Regressor model is initialized and trained on the training data. After training, predictions are made on the test set, and the model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) score. A scatter plot visualizes actual vs. predicted revenue.

### 8. LightGBM Regressor Model (Cells 896b04fb, b6b28f27)
A LightGBM Regressor model, known for its speed and efficiency, is trained and evaluated similarly to XGBoost. Its performance metrics (MSE, R2) are presented, along with a visualization of actual vs. predicted revenue.

### 9. Linear Regression Model with Preprocessing Pipeline (Cells c4d8619b, 18c28aba)
A more comprehensive Linear Regression approach is implemented using a Scikit-learn `Pipeline`. This pipeline integrates:
*   **`StandardScaler`**: For numerical features, to standardize them.
*   **`OneHotEncoder`**: For categorical features (though `Region` was already encoded, this demonstrates a robust pipeline for handling mixed data types).
*   **`LinearRegression`**: The final regression model.
The model is trained, evaluated, and includes a function for predicting revenue for new user inputs.

## Usage
1.  Upload the `Data.csv` file to your Colab environment.
2.  Run all cells in the notebook sequentially.
3.  Observe the data exploration, model training, and evaluation results.
4.  Modify the `sample_input` dictionary in the Linear Regression section (Cell 18c28aba) to predict revenue for different scenarios.

## Dependencies
The following Python libraries are required:
*   `pandas`
*   `numpy`
*   `scikit-learn` (sklearn)
*   `matplotlib`
*   `seaborn`
*   `xgboost`
*   `lightgbm`
