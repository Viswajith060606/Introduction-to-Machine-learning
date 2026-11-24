"""
Experiment 18: Perceptron based Iris Classification
Description: Implement Perceptron based IRIS classification.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ScratchPerceptron:
    """Single Layer Perceptron with Step Activation Function."""
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0.0 else 0
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                if update != 0.0:
                    errors += 1
            if errors == 0:
                print(f"Scratch Perceptron perfectly converged at epoch {epoch + 1}!")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)


def main():
    print("=" * 60)
    print("EXPERIMENT 18: PERCEPTRON BASED IRIS CLASSIFICATION")
    print("=" * 60)

    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Part 1: Binary Classification (Setosa vs Non-Setosa) demonstrating Scratch Perceptron
    print("Part 1: Binary Perceptron (Setosa [1] vs Others [0])")
    y_binary = np.where(y == 0, 1, 0)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train_b_s = scaler.fit_transform(X_train_b)
    X_test_b_s = scaler.transform(X_test_b)

    scratch_p = ScratchPerceptron(learning_rate=0.05, n_iters=50)
    scratch_p.fit(X_train_b_s, y_train_b)
    pred_b = scratch_p.predict(X_test_b_s)
    acc_b = accuracy_score(y_test_b, pred_b)

    print(f"Scratch Binary Perceptron Test Accuracy: {acc_b * 100:.2f}%\n")
    print("-" * 60)

    # Part 2: Multi-Class Perceptron on all 3 Iris Species using Scikit-Learn
    print("Part 2: Multi-Class Perceptron (Setosa, Versicolor, Virginica)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    sklearn_perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    sklearn_perceptron.fit(X_train_s, y_train)

    y_pred = sklearn_perceptron.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    print(f"Multi-Class Perceptron Test Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("=" * 60)


if __name__ == '__main__':
    main()
