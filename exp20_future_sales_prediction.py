"""
Experiment 20: Future Sales Prediction
Description: Implement Future Sales Prediction using a suitable machine learning algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def generate_sales_data(n_samples=250):
    np.random.seed(42)
    # Advertising budgets in thousands of dollars
    tv_budget = np.random.uniform(10.0, 300.0, n_samples)
    radio_budget = np.random.uniform(5.0, 50.0, n_samples)
    newspaper_budget = np.random.uniform(0.0, 100.0, n_samples)

    # True relationship with synergistic marketing effects + random noise
    sales = (
        4.5
        + 0.054 * tv_budget
        + 0.115 * radio_budget
        + 0.003 * newspaper_budget
        + 0.0008 * (tv_budget * radio_budget)  # Synergy between TV and Radio
        + np.random.normal(0, 1.2, n_samples)
    )

    df = pd.DataFrame({
        'TV': tv_budget.round(2),
        'Radio': radio_budget.round(2),
        'Newspaper': newspaper_budget.round(2),
        'Sales_Units_k': sales.round(2)
    })
    return df


def main():
    print("=" * 60)
    print("EXPERIMENT 20: FUTURE SALES PREDICTION")
    print("=" * 60)

    df = generate_sales_data(250)
    print("Sample Advertising Spend and Sales Records (in $ thousands):")
    print(df.head().to_string(index=False))
    print("-" * 60)

    # Correlation Matrix
    print("Correlation Analysis with Sales:")
    corr = df.corr()['Sales_Units_k'].sort_values(ascending=False)
    print(corr.round(4).to_string())
    print("-" * 60)

    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales_Units_k']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1: Multiple Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Model 2: Random Forest Regressor (Captures non-linear ad synergy)
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("Model Evaluation Comparison:")
    print(f"Multiple Linear Regression -> R2 Score: {r2_score(y_test, y_pred_lr):.4f} | MAE: {mean_absolute_error(y_test, y_pred_lr):.3f}k | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.3f}k")
    print(f"Random Forest Regressor    -> R2 Score: {r2_score(y_test, y_pred_rf):.4f} | MAE: {mean_absolute_error(y_test, y_pred_rf):.3f}k | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}k")
    print("-" * 60)

    print("Linear Regression Coefficients:")
    for feature, coef in zip(X.columns, lr.coef_):
        print(f"  {feature:<10} : {coef:.4f}")
    print(f"  Intercept  : {lr.intercept_:.4f}")
    print("-" * 60)

    # Predicting future sales for upcoming campaign budgets
    upcoming_campaigns = pd.DataFrame([
        {'TV': 150.0, 'Radio': 25.0, 'Newspaper': 15.0},
        {'TV': 280.0, 'Radio': 45.0, 'Newspaper': 30.0},
        {'TV': 40.0,  'Radio': 10.0, 'Newspaper': 50.0}
    ])

    pred_sales_lr = lr.predict(upcoming_campaigns)
    pred_sales_rf = rf.predict(upcoming_campaigns)

    print("Forecasted Sales for Next Marketing Campaigns:")
    for i in range(len(upcoming_campaigns)):
        row = upcoming_campaigns.iloc[i]
        print(f"Campaign {i+1}: TV=${row['TV']}k, Radio=${row['Radio']}k, Newspaper=${row['Newspaper']}k")
        print(f"  -> Predicted Sales (Linear Regression): {pred_sales_lr[i]:.2f}k units")
        print(f"  -> Predicted Sales (Random Forest):     {pred_sales_rf[i]:.2f}k units\n")

    print("=" * 60)


if __name__ == '__main__':
    main()
