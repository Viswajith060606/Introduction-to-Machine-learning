"""
Experiment 03: Decision Tree based ID3 Algorithm
Description: Demonstrate the working of the decision tree based ID3 algorithm.
Use an appropriate data set for building the decision tree and apply this knowledge
to classify a new sample.
"""

import math
import pandas as pd


def entropy(data, target_attribute):
    total = len(data)
    if total == 0:
        return 0
    counts = data[target_attribute].value_counts()
    ent = 0.0
    for count in counts:
        p = count / total
        ent -= p * math.log2(p)
    return ent


def information_gain(data, split_attribute, target_attribute):
    total = len(data)
    total_entropy = entropy(data, target_attribute)
    values = data[split_attribute].unique()
    weighted_entropy = 0.0

    for val in values:
        subset = data[data[split_attribute] == val]
        weighted_entropy += (len(subset) / total) * entropy(subset, target_attribute)

    return total_entropy - weighted_entropy


def id3(data, original_data, features, target_attribute):
    # Base Case 1: Pure node
    unique_classes = data[target_attribute].unique()
    if len(unique_classes) == 1:
        return unique_classes[0]

    # Base Case 2: No data in subset
    if len(data) == 0:
        return original_data[target_attribute].mode()[0]

    # Base Case 3: No remaining features to split
    if len(features) == 0:
        return data[target_attribute].mode()[0]

    # Find the feature with the maximum information gain
    gains = [information_gain(data, f, target_attribute) for f in features]
    best_feature_idx = gains.index(max(gains))
    best_feature = features[best_feature_idx]

    # Build the tree dictionary
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, original_data, remaining_features, target_attribute)
        tree[best_feature][value] = subtree

    return tree


def classify(tree, sample, default_label='Yes'):
    """Classifies a single dictionary sample using the ID3 decision tree."""
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    feature_val = sample.get(feature)
    if feature_val in tree[feature]:
        return classify(tree[feature][feature_val], sample, default_label)
    else:
        return default_label


def print_tree(tree, indent=""):
    """Pretty prints the decision tree."""
    if not isinstance(tree, dict):
        print(f" -> Label: {tree}")
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}[{feature} == {val}]", end="")
            if isinstance(subtree, dict):
                print()
                print_tree(subtree, indent + "    ")
            else:
                print(f" -> {subtree}")


def main():
    print("=" * 60)
    print("EXPERIMENT 3: ID3 DECISION TREE ALGORITHM")
    print("=" * 60)

    # Classic PlayTennis dataset
    dataset = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }

    df = pd.DataFrame(dataset)
    print("Training Dataset (Play Tennis):")
    print(df.to_string(index=False))
    print("-" * 60)

    features = [col for col in df.columns if col != 'PlayTennis']
    tree = id3(df, df, features, 'PlayTennis')

    print("\nLearned Decision Tree Structure:")
    print_tree(tree)
    print("-" * 60)

    # Classifying new samples
    test_samples = [
        {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'},
        {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Wind': 'Weak'},
        {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Weak'}
    ]

    print("\nClassifying New Test Samples:")
    for i, sample in enumerate(test_samples, 1):
        prediction = classify(tree, sample)
        print(f"Sample {i}: {sample}")
        print(f"Prediction: PlayTennis = {prediction}\n")
    print("=" * 60)


if __name__ == '__main__':
    main()
