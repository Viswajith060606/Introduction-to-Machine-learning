"""
Experiment 15: Iris Flower Classification using Naive Bayes
Description: Implement Iris Flower Classification using Naive Bayes classifier.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    print("=" * 60)
    print("EXPERIMENT 15: IRIS FLOWER CLASSIFICATION USING NAIVE BAYES")
    print("=" * 60)

    # Load standard Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print("Dataset Information:")
    print(f"Features: {feature_names}")
    print(f"Target Classes: {list(target_names)}")
    print(f"Total Samples: {len(X)}")
    print("-" * 60)

    # Split dataset into 75% training and 25% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Instantiate Gaussian Naive Bayes model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Predictions
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Gaussian Naive Bayes Model Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {name}" for name in target_names],
                             columns=[f"Pred {name}" for name in target_names])
    print(cm_df.to_string())

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("-" * 60)

    # Classifying new unseen iris test instances
    test_flowers = np.array([
        [5.0, 3.4, 1.5, 0.2],   # Expected: Setosa
        [6.2, 2.9, 4.3, 1.3],   # Expected: Versicolor
        [7.3, 3.0, 6.3, 1.8]    # Expected: Virginica
    ])

    predictions = gnb.predict(test_flowers)
    probabilities = gnb.predict_proba(test_flowers)

    print("Predictions on New Test Flower Samples:")
    for i, flower in enumerate(test_flowers):
        pred_class = target_names[predictions[i]]
        print(f"\nSample {i+1} Measurements: {flower}")
        print(f"  Predicted Species: {pred_class}")
        for class_idx, class_name in enumerate(target_names):
            print(f"    Confidence ({class_name}): {probabilities[i][class_idx] * 100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
