"""
Experiment 07: Logistic Regression Algorithm
Description: Write a program to implement Logistic Regression (LR) algorithm in python.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class ScratchLogisticRegression:
    """Logistic Regression implemented from scratch with Gradient Descent."""
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Gradient calculation
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Weight update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


def main():
    print("=" * 60)
    print("EXPERIMENT 7: LOGISTIC REGRESSION ALGORITHM")
    print("=" * 60)

    # Generate synthetic admission/pass dataset
    np.random.seed(42)
    hours_studied = np.random.uniform(1, 10, 120)
    previous_score = np.random.uniform(40, 100, 120)
    # Higher study hours and previous scores increase probability of passing
    z = -8.0 + 0.8 * hours_studied + 0.08 * previous_score
    prob = 1 / (1 + np.exp(-z))
    passed = (prob > 0.5).astype(int)

    X = np.column_stack((hours_studied, previous_score))
    y = passed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Custom Scratch Implementation
    scratch_lr = ScratchLogisticRegression(lr=0.1, n_iters=1500)
    scratch_lr.fit(X_train_scaled, y_train)
    y_pred_scratch = scratch_lr.predict(X_test_scaled)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)

    # 2. Scikit-learn Implementation
    sklearn_lr = LogisticRegression()
    sklearn_lr.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test_scaled)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print(f"Scratch Logistic Regression Accuracy: {acc_scratch * 100:.2f}%")
    print(f"Scikit-learn Logistic Regression Accuracy: {acc_sklearn * 100:.2f}%")
    print("-" * 60)

    print("Confusion Matrix (Scikit-Learn Model):")
    print(confusion_matrix(y_test, y_pred_sklearn))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sklearn, target_names=['Fail', 'Pass']))

    # Predict for a new student
    new_student = np.array([[7.5, 82.0]])
    new_student_scaled = scaler.transform(new_student)
    pred_prob = sklearn_lr.predict_proba(new_student_scaled)[0][1]
    pred_class = sklearn_lr.predict(new_student_scaled)[0]
    print("-" * 60)
    print(f"Sample Student (Hours: 7.5, Prev Score: 82.0):")
    print(f"Pass Probability: {pred_prob:.4f} -> Prediction: {'Pass' if pred_class == 1 else 'Fail'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
