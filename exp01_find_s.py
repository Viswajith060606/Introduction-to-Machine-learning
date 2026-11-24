"""
Experiment 01: FIND-S Algorithm
Description: Implement and demonstrate the FIND-S algorithm for finding the most
specific hypothesis based on a given set of training data samples.
"""

import pandas as pd
import numpy as np


def find_s_algorithm(concepts, targets):
    """
    Find-S algorithm to find the maximally specific hypothesis.
    concepts: 2D array of attribute values for training instances.
    targets: 1D array of target concept labels ('Yes' / 'No' or 1 / 0).
    """
    num_attributes = concepts.shape[1]
    
    # Initialize most specific hypothesis with '0' (or None)
    specific_h = ['0'] * num_attributes
    print("Initial Specific Hypothesis: ", specific_h)

    # Find the first positive example to initialize the specific hypothesis
    for i, target in enumerate(targets):
        if str(target).strip().lower() in ['yes', '1', 'positive', 'true']:
            specific_h = list(concepts[i])
            print(f"Initialized with first positive example (Instance {i+1}): {specific_h}")
            break

    # Update hypothesis for each subsequent positive example
    for i, instance in enumerate(concepts):
        target = targets[i]
        if str(target).strip().lower() in ['yes', '1', 'positive', 'true']:
            for j in range(num_attributes):
                if instance[j] != specific_h[j]:
                    specific_h[j] = '?'
            print(f"Instance {i+1} [Positive] -> Updated Specific Hypothesis: {specific_h}")
        else:
            print(f"Instance {i+1} [Negative] -> Ignored by FIND-S")

    return specific_h


def main():
    print("=" * 60)
    print("EXPERIMENT 1: FIND-S ALGORITHM")
    print("=" * 60)

    # Standard EnjoySport training dataset
    data = {
        'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
        'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
        'Humidity': ['Normal', 'High', 'High', 'High'],
        'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
        'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
        'Forecast': ['Same', 'Same', 'Change', 'Change'],
        'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
    }
    
    df = pd.DataFrame(data)
    print("\nTraining Dataset:")
    print(df.to_string(index=True))
    print("-" * 60)

    attributes = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values

    final_h = find_s_algorithm(attributes, target)
    print("-" * 60)
    print(f"\nFinal Most Specific Hypothesis: {final_h}")
    print("=" * 60)


if __name__ == '__main__':
    main()
