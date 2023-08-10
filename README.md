# Data Analysis and Machine Learning Notebooks
This repository contains two Jupyter notebooks focusing on data analysis and machine learning tasks using Python. The notebooks are meant to demonstrate how to analyze datasets, preprocess the data, and train various machine learning models for different types of predictions.

## Asteroid Hazard Prediction
The first notebook, named asteroid_hazard_prediction.ipynb, analyzes data related to Near-Earth Objects (NEOs) and attempts to predict whether an asteroid is hazardous or not based on several features.

### Contents
* Data loading and initial exploration
* Data preprocessing and visualization
* Feature analysis and selection
* Data splitting using StratifiedShuffleSplit
* Model training using Logistic Regression, K-Nearest Neighbors, Random Forest, and XGBoost
* Hyperparameter tuning using GridSearchCV
* Model evaluation using cross-validation and recall score

## Health Insurance Cost Prediction
The second notebook, named insurance_cost_prediction.ipynb, focuses on predicting health insurance costs for individuals based on various factors.

### Contents
* Data loading and initial exploration
* Data preprocessing, encoding categorical features
* Data visualization and correlation analysis
* Data splitting using train_test_split
* Model training using Linear Regression, K-Nearest Neighbors Regressor, Random Forest Regressor, and XGBoost Regressor
* Hyperparameter tuning using GridSearchCV
* Model evaluation using cross-validation and negative root mean squared error (neg_RMSE)
* Confidence interval calculation for model performance

## Requirements
The notebooks are self-contained and designed to run in a Jupyter environment. The following libraries are used in the notebooks:
* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* xgboost
* scipy

You can install these libraries using the following command:

```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy
```
