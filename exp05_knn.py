"""
Experiment 05: K-Nearest Neighbours (K-NN)
Description: Write a program for Implementation of K-Nearest Neighbours (K-NN) in Python.
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification


class KNNScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        # Compute Euclidean distance between x and all training points
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        # Extract labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def main():
    print("=" * 60)
    print("EXPERIMENT 5: K-NEAREST NEIGHBOURS (K-NN) IMPLEMENTATION")
    print("=" * 60)

    # Generate synthetic 2D classification dataset for demonstration
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=3, n_redundant=1,
        n_classes=3, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset Details:")
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Testing samples : {X_test.shape[0]}")
    print("-" * 60)

    for k in [3, 5, 7]:
        knn = KNNScratch(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"K = {k}: Test Accuracy = {acc * 100:.2f}%")

    # Evaluate best model (K=3)
    best_knn = KNNScratch(k=3)
    best_knn.fit(X_train, y_train)
    y_pred = best_knn.predict(X_test)

    print("-" * 60)
    print("Classification Report for K=3:")
    print(classification_report(y_test, y_pred))

    # Test single query prediction
    sample_query = X_test[0:1]
    pred = best_knn.predict(sample_query)[0]
    print(f"Single Query Sample: {sample_query[0].round(3)}")
    print(f"Actual Label: {y_test[0]} | Predicted Label: {pred}")
    print("=" * 60)


if __name__ == '__main__':
    main()
