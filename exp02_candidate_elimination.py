"""
Experiment 02: Candidate-Elimination Algorithm
Description: For a given set of training data examples stored in a .CSV file,
implement and demonstrate the Candidate-Elimination algorithm in python to output
a description of the set of all hypotheses consistent with the training examples.
"""

import os
import pandas as pd
import numpy as np


def candidate_elimination(csv_path):
    df = pd.read_csv(csv_path)
    print("Loaded Training Dataset from CSV:")
    print(df.to_string(index=False))
    print("-" * 60)

    concepts = df.iloc[:, :-1].values
    targets = df.iloc[:, -1].values
    num_attributes = concepts.shape[1]

    # Initialize Specific and General Boundaries
    specific_h = ['0'] * num_attributes
    general_h = [['?' for _ in range(num_attributes)]]

    # Find the first positive instance to initialize specific_h
    for i, target in enumerate(targets):
        if str(target).strip().lower() in ['yes', '1', 'positive', 'true']:
            specific_h = list(concepts[i])
            break

    print(f"Initial S0: {specific_h}")
    print(f"Initial G0: {general_h}\n")

    for i, instance in enumerate(concepts):
        target = targets[i]
        is_positive = str(target).strip().lower() in ['yes', '1', 'positive', 'true']

        if is_positive:
            # Positive Example: Generalize S
            for j in range(num_attributes):
                if instance[j] != specific_h[j]:
                    specific_h[j] = '?'
            
            # Remove any hypothesis from G that is inconsistent with the positive instance
            general_h = [
                g for g in general_h
                if all(g[k] == '?' or g[k] == instance[k] for k in range(num_attributes))
            ]
        else:
            # Negative Example: Specialize G
            new_general_h = []
            for g in general_h:
                # If g is inconsistent with negative instance (i.e. g predicts positive), specialize it
                if all(g[k] == '?' or g[k] == instance[k] for k in range(num_attributes)):
                    for j in range(num_attributes):
                        if g[j] == '?' and specific_h[j] != instance[j] and specific_h[j] != '?':
                            specialized = g.copy()
                            specialized[j] = specific_h[j]
                            new_general_h.append(specialized)
                else:
                    new_general_h.append(g)

            # Eliminate duplicates and hypotheses not more general than some hypothesis in S
            general_h = []
            for h in new_general_h:
                if h not in general_h:
                    general_h.append(h)

        print(f"Step {i+1} (Instance: {list(instance)}, Label: {target}):")
        print(f"  S{i+1}: {specific_h}")
        print(f"  G{i+1}: {general_h}\n")

    return specific_h, general_h


def create_sample_csv(csv_path="enjoy_sport.csv"):
    if not os.path.exists(csv_path):
        data = {
            'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
            'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
            'Humidity': ['Normal', 'High', 'High', 'High'],
            'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
            'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
            'Forecast': ['Same', 'Same', 'Change', 'Change'],
            'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)
        print(f"Created sample dataset: {csv_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 2: CANDIDATE-ELIMINATION ALGORITHM")
    print("=" * 60)
    csv_file = "enjoy_sport.csv"
    create_sample_csv(csv_file)
    s_final, g_final = candidate_elimination(csv_file)
    print("=" * 60)
    print("Final Version Space:")
    print(f"Most Specific Boundary (S): {s_final}")
    print(f"Most General Boundary  (G): {g_final}")
    print("=" * 60)


if __name__ == '__main__':
    main()
