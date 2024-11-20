# Sales Prediction Model Using LightGBM and Optuna

## Project Overview
This repository contains a sales prediction model built using **LightGBM**, a powerful gradient boosting algorithm, and **Optuna** for hyperparameter optimization. The model predicts sales for various stores using features such as store information, transaction data, holidays, and oil prices.

Key visualizations:
1. **Actual vs. Predicted Plot**: Displays the alignment of model predictions with actual sales values.
2. **Residual Histogram**: Plots residuals to assess model performance and bias.

---

## Requirements
To run this project, ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- optuna
- lightgbm


pip install pandas numpy matplotlib seaborn scikit-learn optuna lightgbm
## How to Run
Step 1: Load and Preprocess Data
Run the script sales_prediction_model.py to:

Load the datasets.
Clean and preprocess the data (handle missing values, encode categorical features, etc.).
Merge datasets into a single DataFrame for model training.
Step 2: Train the Model
The script trains the model using LightGBM and optimizes hyperparameters with Optuna. The best hyperparameters are automatically selected to maximize model performance.

Step 3: Evaluate Model Performance
The model's performance is evaluated on a validation set using the following metrics:

R² (R-squared): Proportion of variance in the target explained by the model.
RMSE (Root Mean Squared Error): Average magnitude of prediction errors.
MAE (Mean Absolute Error): Average absolute difference between predicted and actual values.
RMSLE (Root Mean Squared Logarithmic Error): Measures log-scaled prediction error.
Step 4: Visualize Model Performance
The script generates two key plots:

## Actual vs. Predicted Plot:

A scatter plot comparing actual and predicted sales values.
The red dashed line represents perfect predictions.
Residual Histogram:
Shows the distribution of residuals (errors) to assess bias and variance.
A well-performing model should have residuals centered around zero.
Step 5: Predict on Test Data
Use the trained model to predict sales on the test dataset.
Save predictions to submission.csv with columns:
id: Unique identifier for each store.
sales: Predicted sales.

## Results
Model Evaluation Metrics
R²: 0.87 – Explains 87% of the variance in sales.
RMSE: 499.09 – Average deviation of predictions by 499 sales units.
MAE: 150.65 – Average error of approximately 150.65 units.
RMSLE: 0.15 – Indicates reasonable log-scale prediction error.

## Graphs

Actual vs. Predicted Plot:
Most points cluster near the red dashed line, showing accurate predictions.
Some deviations are observed for high sales values (outliers).
Residual Histogram:
Residuals are centered around zero with some skewness, indicating good overall performance but room for improvement in handling outliers.
