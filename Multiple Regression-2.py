"""
Description: This script performs multiple regression analysis on a dataset
to predict the dependent variable 'speed' based on various independent variables.
It includes data loading, visualization, model fitting, multicollinearity analysis,
and different regression techniques (linear, ridge, lasso).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as stm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv(r"C:\Users\satis\Downloads\data_problem_4.csv")

# Define the independent and dependent variable
target_variable = "speed"
independent_variables = ['flow', 'occupancy', 'speed_down', 'flow_ratio']

# Create scatter plots for visualization
pd.plotting.scatter_matrix(data[independent_variables], figsize=(10, 10))
plt.show()

# Fit the initial regression model
x_matrix = stm.add_constant(data[independent_variables])
y_matrix = data[target_variable]
model = stm.OLS(y_matrix, x_matrix)
results = model.fit()

# Analyze multicollinearity using the correlation matrix
correlation_matrix = x_matrix.corr()
print("Correlation Matrix: ")
print(correlation_matrix)

# Remove variables causing multicollinearity
independent_variables.remove('occupancy')
independent_variables.remove('flow_ratio')

# Refit the regression model with the updated independent variables
x_matrix = stm.add_constant(data[independent_variables])
model = stm.OLS(y_matrix, x_matrix)
results = model.fit()

# Regression Analysis
# Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(x_matrix, y_matrix)
linear_regression_coefficients = linear_regression.coef_
linear_regression_predictions = linear_regression.predict(x_matrix)
linear_regression_summary = {
    'Coefficients': linear_regression_coefficients,
    'R2 Score': linear_regression.score(x_matrix, y_matrix),
    'Mean Squared Error (MSE)': mean_squared_error(y_matrix, linear_regression_predictions),
    'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_matrix, linear_regression_predictions))}
print(linear_regression_summary)

# Ridge Regression
ridge_regression = Ridge(alpha=1.0)
ridge_regression.fit(x_matrix, y_matrix)
ridge_regression_coefficients = ridge_regression.coef_
ridge_regression_predictions = ridge_regression.predict(x_matrix)
ridge_regression_summary = {
    'Coefficients': ridge_regression_coefficients,
    'R2 Score': ridge_regression.score(x_matrix, y_matrix),
    'Mean Squared Error (MSE)': mean_squared_error(y_matrix, ridge_regression_predictions),
    'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_matrix, ridge_regression_predictions))}
print(ridge_regression_summary)

# Lasso Regression
lasso_regression = Lasso(alpha=1.0)
lasso_regression.fit(x_matrix, y_matrix)
lasso_regression_coefficients = lasso_regression.coef_
lasso_regression_predictions = lasso_regression.predict(x_matrix)
lasso_regression_summary = {
    'Coefficients': lasso_regression_coefficients,
    'R2 Score': lasso_regression.score(x_matrix, y_matrix),
    'Mean Squared Error (MSE)': mean_squared_error(y_matrix, lasso_regression_predictions),
    'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_matrix, lasso_regression_predictions))}
print(lasso_regression_summary)
