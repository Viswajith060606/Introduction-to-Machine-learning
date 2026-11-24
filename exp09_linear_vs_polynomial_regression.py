"""
Experiment 09: Compare Linear and Polynomial Regression
Description: Compare Linear and Polynomial Regression using Python.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("EXPERIMENT 9: COMPARE LINEAR AND POLYNOMIAL REGRESSION")
    print("=" * 60)

    # Non-linear relationship: Temperature vs Chemical Yield / Production Rate
    # True relationship: y = 20 + 3*x - 0.2*x^2 + noise
    np.random.seed(42)
    x = np.linspace(0, 20, 60)
    y = 20 + 3.5 * x - 0.22 * (x ** 2) + np.random.normal(0, 2.5, len(x))

    X = x.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. Simple Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    # 2. Polynomial Regression Degree 2
    poly2 = PolynomialFeatures(degree=2)
    X_train_poly2 = poly2.fit_transform(X_train)
    X_test_poly2 = poly2.transform(X_test)

    poly2_model = LinearRegression()
    poly2_model.fit(X_train_poly2, y_train)
    y_pred_poly2 = poly2_model.predict(X_test_poly2)

    # 3. Polynomial Regression Degree 3
    poly3 = PolynomialFeatures(degree=3)
    X_train_poly3 = poly3.fit_transform(X_train)
    X_test_poly3 = poly3.transform(X_test)

    poly3_model = LinearRegression()
    poly3_model.fit(X_train_poly3, y_train)
    y_pred_poly3 = poly3_model.predict(X_test_poly3)

    # Comparison metrics
    models = {
        'Linear Regression': (y_pred_linear),
        'Polynomial Regression (Degree 2)': (y_pred_poly2),
        'Polynomial Regression (Degree 3)': (y_pred_poly3)
    }

    results = []
    for name, pred in models.items():
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        results.append({
            'Model': name,
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2 Score': round(r2, 4)
        })

    results_df = pd.DataFrame(results)
    print("Performance Comparison Table:")
    print(results_df.to_string(index=False))
    print("-" * 60)

    print("Model Equations / Learned Parameters:")
    print(f"Linear:     y = {linear_model.intercept_:.2f} + ({linear_model.coef_[0]:.2f})*x")
    print(f"Poly Deg 2: y = {poly2_model.intercept_:.2f} + ({poly2_model.coef_[1]:.2f})*x + ({poly2_model.coef_[2]:.2f})*x^2")
    print("-" * 60)
    print("Conclusion:")
    print("Polynomial Regression captures the curved, non-linear pattern in the data,")
    print("achieving significantly lower error (MSE) and a much higher R2 Score than Linear Regression.")
    print("=" * 60)


if __name__ == '__main__':
    main()
