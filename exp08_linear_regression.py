"""
Experiment 08: Linear Regression Algorithm
Description: Write a program to implement Linear Regression (LR) algorithm in python.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class ScratchLinearRegression:
    """Simple and Multiple Linear Regression using Ordinary Least Squares."""
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add bias column (vector of 1s)
        X_design = np.c_[np.ones(X.shape[0]), X]
        # Beta = (X^T * X)^(-1) * X^T * y
        beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        return self.intercept + np.dot(X, self.coefficients)


def main():
    print("=" * 60)
    print("EXPERIMENT 8: LINEAR REGRESSION ALGORITHM")
    print("=" * 60)

    # Dataset: Years of Experience vs Salary
    np.random.seed(42)
    years_experience = np.random.uniform(1.0, 12.0, 50)
    # Salary = 30000 + 8500 * Experience + noise
    salary = 30000 + 8500 * years_experience + np.random.normal(0, 3000, 50)

    df = pd.DataFrame({'YearsExperience': years_experience, 'Salary': salary})
    print("Sample Data (First 5 records):")
    print(df.head().round(2).to_string(index=False))
    print("-" * 60)

    X = df[['YearsExperience']].values
    y = df['Salary'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Scratch Implementation
    scratch_lr = ScratchLinearRegression()
    scratch_lr.fit(X_train, y_train)
    y_pred_scratch = scratch_lr.predict(X_test)

    # 2. Scikit-learn Implementation
    sklearn_lr = LinearRegression()
    sklearn_lr.fit(X_train, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred_sklearn)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_sklearn)
    r2 = r2_score(y_test, y_pred_sklearn)

    print("Model Parameters:")
    print(f"Scratch:   Slope = {scratch_lr.coefficients[0]:.2f}, Intercept = {scratch_lr.intercept:.2f}")
    print(f"Sklearn:   Slope = {sklearn_lr.coef_[0]:.2f}, Intercept = {sklearn_lr.intercept_:.2f}")
    print("-" * 60)
    print("Performance Metrics:")
    print(f"Mean Absolute Error (MAE)      : ${mae:.2f}")
    print(f"Mean Squared Error (MSE)       : {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE) : ${rmse:.2f}")
    print(f"R-squared Score (R2)           : {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    print("-" * 60)

    # Predict for new experience values
    new_exp = np.array([[3.5], [6.0], [10.0]])
    predictions = sklearn_lr.predict(new_exp)
    for exp, pred in zip(new_exp, predictions):
        print(f"Experience: {exp[0]} years -> Predicted Salary: ${pred:,.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
