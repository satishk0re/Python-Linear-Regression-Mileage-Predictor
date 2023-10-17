"""
Description: This script performs multiple regression analysis on a dataset
to predict the dependent variable 'mpg' based on various independent variables.
It includes data loading, model fitting, significance tests, confidence
intervals, residual analysis, and result visualization.
"""

import numpy as np
import pandas as pd
import statsmodels.api as stm
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv(r"C:\Users\satis\Downloads\data_problem_3.csv")

# Define the independent and dependent variable
target_variable = "mpg"
independent_variables = ["cid", "rhp", "etw", "cmp", "axle", "n/v"]

# Fit the multiple regression model
x_matrix = stm.add_constant(data[independent_variables])
y_vector = data[target_variable]
model = stm.OLS(y_vector, x_matrix)
initial_results = model.fit()

# Calculate and print summary statistics
summary = initial_results.summary()
print(summary)

# Estimate sigma^2 and standard error of the regression coefficients
sigma_squared = initial_results.mse_resid
standard_errors = initial_results.bse
print("Standard Error:")
print(standard_errors)

# Perform significance tests
t_values = initial_results.tvalues[1:]
p_values = initial_results.pvalues[1:]
alpha = .05
critical_value = alpha / 2  # Divide alpha by 2 for two-tailed test

# Iterate through independent variables and test their significance
for i in range(len(t_values)):
    variable = x_matrix.columns[i + 1]
    t_value = t_values[i]
    p_value = p_values[i]

    # Check if the p-value is significant
    if np.abs(p_value) > critical_value:
        result = "Reject the null hypothesis."
    else:
        result = "Failed to reject the null hypothesis."

    print(f"Independent Variable: {variable}")
    print(f"t-value: {t_value}")
    print(f"p_value: {p_value}")
    print(result)
    print()

# Calculate 99% confidence intervals on regression coefficients
confidence_intervals = initial_results.conf_int(alpha=0.01)
print("Confidence intervals of the regression coefficients are :\n")
print(confidence_intervals)

# Plot residuals versus predicted values and each independent variable
predicted_values = initial_results.fittedvalues
residuals = initial_results.resid

plt.scatter(predicted_values, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals versus Predicted Values")
plt.show()

# Create residual plots for each independent variable
for i, var in enumerate(independent_variables):
    plt.figure()
    plt.scatter(data[var], residuals)
    plt.xlabel(var)
    plt.ylabel('Residuals')
    plt.title(f"Residuals vs {var}")

plt.show()

# Calculate and display confidence intervals for the mean response
prediction_intervals_mean = initial_results.get_prediction(x_matrix).conf_int(alpha=0.05)
prediction_intervals_mean_lower = prediction_intervals_mean[:, 0]
prediction_intervals_mean_upper = prediction_intervals_mean[:, 1]

plt.scatter(data[target_variable], prediction_intervals_mean_lower, label='Lower Limit')
plt.scatter(data[target_variable], prediction_intervals_mean_upper, label='Upper Limit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Confidence Limits for Mean Response')
plt.legend()
plt.show()
