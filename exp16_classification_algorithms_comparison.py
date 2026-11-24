"""
Experiment 16: Compare Different Classification Algorithms
Description: Compare different types Classification Algorithms and evaluate their performance.
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    print("=" * 60)
    print("EXPERIMENT 16: COMPARISON OF CLASSIFICATION ALGORITHMS")
    print("=" * 60)

    # Generate multi-class synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classification models to benchmark
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Gaussian Naive Bayes': GaussianNB()
    }

    results = []

    print(f"{'Algorithm':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Train Time (ms)':<15}")
    print("-" * 90)

    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = (time.time() - start_time) * 1000  # milliseconds

        y_pred = clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Algorithm': name,
            'Accuracy (%)': round(acc * 100, 2),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Train Time (ms)': round(train_time, 2)
        })

        print(f"{name:<25} | {acc*100:6.2f}%    | {prec:8.4f}   | {rec:8.4f}   | {f1:8.4f}   | {train_time:10.2f} ms")

    results_df = pd.DataFrame(results).sort_values(by='Accuracy (%)', ascending=False)
    print("-" * 90)
    print("\nLeaderboard Ranked by Accuracy:")
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]['Algorithm']
    best_acc = results_df.iloc[0]['Accuracy (%)']
    print(f"\nTop Performing Classifier: {best_model} ({best_acc}% Accuracy)")
    print("=" * 60)


if __name__ == '__main__':
    main()
