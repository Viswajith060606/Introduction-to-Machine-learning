"""
Experiment 13: Car Price Prediction Model
Description: Implement the Car Price Prediction Model using Python.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def generate_car_dataset(n_samples=300):
    np.random.seed(42)
    current_year = 2026
    year = np.random.randint(2010, 2024, n_samples)
    car_age = current_year - year
    present_price = np.random.uniform(3.0, 25.0, n_samples)  # in Lakhs
    kms_driven = np.random.randint(5000, 150000, n_samples)
    fuel_type = np.random.choice(['Petrol', 'Diesel', 'CNG'], n_samples, p=[0.6, 0.35, 0.05])
    seller_type = np.random.choice(['Dealer', 'Individual'], n_samples, p=[0.7, 0.3])
    transmission = np.random.choice(['Manual', 'Automatic'], n_samples, p=[0.8, 0.2])
    owner = np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.12, 0.03])

    # Realistic Selling Price calculation
    depreciation = (car_age * 0.45) + (kms_driven / 40000.0) + (owner * 0.8)
    fuel_bonus = np.where(fuel_type == 'Diesel', 1.2, np.where(fuel_type == 'CNG', -0.5, 0.0))
    trans_bonus = np.where(transmission == 'Automatic', 1.5, 0.0)
    selling_price = present_price * 0.85 - depreciation + fuel_bonus + trans_bonus + np.random.normal(0, 0.6, n_samples)
    selling_price = np.clip(selling_price, 0.5, present_price * 0.95)

    df = pd.DataFrame({
        'Car_Age': car_age,
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission,
        'Owner': owner,
        'Selling_Price': selling_price
    })
    return df


def main():
    print("=" * 60)
    print("EXPERIMENT 13: CAR PRICE PREDICTION MODEL")
    print("=" * 60)

    df = generate_car_dataset(300)
    print("Sample Car Dataset:")
    print(df.head().round(2).to_string(index=False))
    print("-" * 60)

    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

    X = df_encoded.drop('Selling_Price', axis=1)
    y = df_encoded['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("Model Performance Comparison:")
    print(f"Linear Regression    -> R2 Score: {r2_score(y_test, y_pred_lr):.4f} | MAE: {mean_absolute_error(y_test, y_pred_lr):.3f} Lakhs | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.3f}")
    print(f"Random Forest Model  -> R2 Score: {r2_score(y_test, y_pred_rf):.4f} | MAE: {mean_absolute_error(y_test, y_pred_rf):.3f} Lakhs | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}")
    print("-" * 60)

    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top Feature Importances (Random Forest):")
    print(importances.round(4).to_string())

    # Predict price for a sample queried car
    sample_car = pd.DataFrame([{
        'Car_Age': 5,
        'Present_Price': 9.85,
        'Kms_Driven': 35000,
        'Owner': 0,
        'Fuel_Type_Diesel': 1,
        'Fuel_Type_Petrol': 0,
        'Seller_Type_Individual': 0,
        'Transmission_Manual': 1
    }])

    # Ensure matching columns
    sample_car = sample_car.reindex(columns=X.columns, fill_value=0)
    predicted_val = rf.predict(sample_car)[0]

    print("-" * 60)
    print("Prediction for Sample Used Car (Age: 5 yrs, Showroom Price: 9.85 Lakhs, 35k km, Diesel, Manual):")
    print(f"Estimated Resale Price: {predicted_val:.2f} Lakhs INR")
    print("=" * 60)


if __name__ == '__main__':
    main()
