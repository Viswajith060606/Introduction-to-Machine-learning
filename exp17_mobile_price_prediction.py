"""
Experiment 17: Mobile Price Prediction
Description: Implement Mobile Price Prediction using appropriate machine learning algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def generate_mobile_dataset(n_samples=600):
    np.random.seed(42)
    battery_power = np.random.randint(500, 5000, n_samples)
    blue = np.random.choice([0, 1], n_samples)
    clock_speed = np.random.uniform(0.5, 3.0, n_samples)
    dual_sim = np.random.choice([0, 1], n_samples)
    fc = np.random.randint(0, 20, n_samples)  # Front camera MP
    four_g = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    int_memory = np.random.choice([16, 32, 64, 128, 256], n_samples)
    mobile_wt = np.random.randint(80, 220, n_samples)
    n_cores = np.random.choice([2, 4, 6, 8], n_samples)
    pc = np.random.randint(5, 64, n_samples)  # Primary camera MP
    px_height = np.random.randint(200, 1960, n_samples)
    px_width = np.random.randint(500, 2560, n_samples)
    ram = np.random.randint(512, 12000, n_samples)  # RAM in MB

    # Realistic price score: RAM and Battery are dominant factors in mobile pricing
    score = (
        (ram / 1024.0) * 15.0
        + (battery_power / 500.0) * 3.5
        + (px_height * px_width / 1e6) * 4.0
        + (int_memory / 32.0) * 2.0
        + (pc / 10.0) * 1.5
        + (n_cores * 1.2)
        + np.random.normal(0, 5, n_samples)
    )

    # 4 Price Ranges: 0 (Low), 1 (Medium), 2 (High), 3 (Very High / Flagship)
    q25, q50, q75 = np.percentile(score, [25, 50, 75])
    price_range = np.where(score < q25, 0, np.where(score < q50, 1, np.where(score < q75, 2, 3)))

    df = pd.DataFrame({
        'battery_power': battery_power,
        'bluetooth': blue,
        'clock_speed': clock_speed.round(2),
        'dual_sim': dual_sim,
        'fc_camera_mp': fc,
        'four_g': four_g,
        'int_memory_gb': int_memory,
        'weight_grams': mobile_wt,
        'n_cores': n_cores,
        'pc_camera_mp': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram_mb': ram,
        'price_range': price_range
    })
    return df


def main():
    print("=" * 60)
    print("EXPERIMENT 17: MOBILE PRICE PREDICTION")
    print("=" * 60)

    df = generate_mobile_dataset(600)
    print("Sample Mobile Specification Data:")
    print(df.head().to_string(index=False))
    print("-" * 60)

    X = df.drop('price_range', axis=1)
    y = df['price_range']
    price_labels = ['Low Cost (0)', 'Medium Cost (1)', 'High Cost (2)', 'Flagship (3)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)

    y_pred = rf_classifier.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"Mobile Price Classification Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=price_labels))
    print("-" * 60)

    # Feature Importance
    importances = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top Feature Importances for Mobile Pricing:")
    print(importances.head(5).round(4).to_string())
    print("-" * 60)

    # Predict price range for a new phone
    new_phone = pd.DataFrame([{
        'battery_power': 4500,
        'bluetooth': 1,
        'clock_speed': 2.8,
        'dual_sim': 1,
        'fc_camera_mp': 16,
        'four_g': 1,
        'int_memory_gb': 128,
        'weight_grams': 175,
        'n_cores': 8,
        'pc_camera_mp': 48,
        'px_height': 1080,
        'px_width': 2400,
        'ram_mb': 8192
    }])

    new_phone_scaled = scaler.transform(new_phone)
    predicted_range = rf_classifier.predict(new_phone_scaled)[0]
    print(f"Tested Specs: 8GB RAM, 4500mAh Battery, 48MP Camera, Octa-Core 2.8GHz")
    print(f"Predicted Price Category: {price_labels[predicted_range]}")
    print("=" * 60)


if __name__ == '__main__':
    main()
