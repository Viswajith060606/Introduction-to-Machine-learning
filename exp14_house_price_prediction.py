"""
Experiment 14: House Price Prediction
Description: Implement House price Prediction using appropriate machine learning algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def generate_housing_data(n_samples=400):
    np.random.seed(42)
    sqft = np.random.uniform(600, 4500, n_samples)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.35, 0.2, 0.05])
    bathrooms = np.clip(bedrooms - np.random.choice([0, 1], n_samples, p=[0.7, 0.3]), 1, 5)
    age = np.random.uniform(1, 40, n_samples)
    garage_cars = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.45, 0.15])
    distance_to_city_km = np.random.uniform(2, 35, n_samples)

    # Base price calculation in thousands of dollars
    price = (
        120
        + 0.18 * sqft
        + 25 * bedrooms
        + 18 * bathrooms
        + 15 * garage_cars
        - 2.5 * distance_to_city_km
        - 1.8 * age
        + np.random.normal(0, 25, n_samples)
    )

    df = pd.DataFrame({
        'SquareFeet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'AgeYears': age,
        'GarageCars': garage_cars,
        'DistanceCityCenterKm': distance_to_city_km,
        'Price_Thousands': price
    })
    return df


def main():
    print("=" * 60)
    print("EXPERIMENT 14: HOUSE PRICE PREDICTION")
    print("=" * 60)

    df = generate_housing_data(400)
    print("Sample Housing Dataset:")
    print(df.head().round(2).to_string(index=False))
    print("-" * 60)

    X = df.drop('Price_Thousands', axis=1)
    y = df['Price_Thousands']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: Ridge Regularized Linear Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)

    # Model 2: Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("Model Evaluation Comparison:")
    print(f"Ridge Regression     -> R2 Score: {r2_score(y_test, y_pred_ridge):.4f} | MAE: ${mean_absolute_error(y_test, y_pred_ridge) * 1000:,.2f} | RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_ridge)) * 1000:,.2f}")
    print(f"Random Forest Model  -> R2 Score: {r2_score(y_test, y_pred_rf):.4f} | MAE: ${mean_absolute_error(y_test, y_pred_rf) * 1000:,.2f} | RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_rf)) * 1000:,.2f}")
    print("-" * 60)

    # Predict price for a sample house
    sample_house = pd.DataFrame([{
        'SquareFeet': 2400,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'AgeYears': 8,
        'GarageCars': 2,
        'DistanceCityCenterKm': 12
    }])

    pred_price_k = rf_model.predict(sample_house)[0]
    print("Prediction for Sample House:")
    for k, v in sample_house.iloc[0].items():
        print(f"  {k}: {v}")
    print(f"\nEstimated Market Value: ${pred_price_k * 1000:,.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
