"""
Experiment 12: Iris Flower Classification using KNN
Description: Implement Iris Flower Classification using KNN.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    print("=" * 60)
    print("EXPERIMENT 12: IRIS FLOWER CLASSIFICATION USING KNN")
    print("=" * 60)

    # Load standard Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df['Species'] = [target_names[i] for i in y]

    print("Iris Dataset Overview (First 5 samples):")
    print(df.head().to_string(index=False))
    print(f"\nClass Distribution:\n{df['Species'].value_counts().to_string()}")
    print("-" * 60)

    # Train-test split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate multiple values of K
    print("Evaluating K values from 1 to 11:")
    k_scores = {}
    for k in range(1, 12, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)
        k_scores[k] = acc
        print(f"  K = {k:2d} -> Test Accuracy: {acc * 100:.2f}%")

    optimal_k = max(k_scores, key=k_scores.get)
    print(f"\nSelected Optimal K = {optimal_k}")
    print("-" * 60)

    # Final Model Training with Optimal K
    final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
    final_knn.fit(X_train_scaled, y_train)
    y_pred = final_knn.predict(X_test_scaled)

    print("Final Model Evaluation:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Prediction on unseen sample flower
    sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Classic Iris-setosa values
    sample_flower_scaled = scaler.transform(sample_flower)
    prediction = final_knn.predict(sample_flower_scaled)[0]
    probabilities = final_knn.predict_proba(sample_flower_scaled)[0]

    print("-" * 60)
    print(f"Testing Unseen Sample: {sample_flower[0]} (Sepal Length, Sepal Width, Petal Length, Petal Width)")
    print(f"Predicted Species: {target_names[prediction]}")
    for name, prob in zip(target_names, probabilities):
        print(f"  Confidence ({name}): {prob * 100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
