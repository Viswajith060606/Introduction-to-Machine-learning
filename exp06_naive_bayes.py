"""
Experiment 06: Naive Bayes Algorithm
Description: Write a program to implement Naïve Bayes algorithm in python
and to display the results using confusion matrix and accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    print("=" * 60)
    print("EXPERIMENT 6: NAIVE BAYES ALGORITHM")
    print("=" * 60)

    # Synthetic clinical / diagnostic dataset
    np.random.seed(42)
    n_samples = 200
    glucose = np.random.normal(120, 30, n_samples)
    blood_pressure = np.random.normal(75, 15, n_samples)
    insulin = np.random.normal(80, 40, n_samples)
    bmi = np.random.normal(28, 6, n_samples)
    
    # Target label: Diabetic (1) or Not (0) based on features with noise
    prob = 1 / (1 + np.exp(-(0.03 * glucose + 0.02 * blood_pressure + 0.04 * bmi - 7)))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'Insulin': insulin,
        'BMI': bmi,
        'Diabetes': target
    })

    print("Sample Data (First 5 rows):")
    print(df.head().to_string(index=False))
    print("-" * 60)

    X = df[['Glucose', 'BloodPressure', 'Insulin', 'BMI']]
    y = df['Diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Initialize and train Gaussian Naive Bayes Classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = nb_classifier.predict(X_test)

    # Performance Evaluation
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print("                 Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {cm[0][0]:<15}      {cm[0][1]:<15}")
    print(f"Actual Positive        {cm[1][0]:<15}      {cm[1][1]:<15}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))
    print("=" * 60)


if __name__ == '__main__':
    main()
