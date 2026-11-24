"""
Experiment 10: Expectation & Maximization (EM) Algorithm
Description: Write a Python Program to Implement Expectation & Maximization Algorithm.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


class GMM_EM_1D:
    """Expectation-Maximization for 1D Gaussian Mixture Model."""
    def __init__(self, k=2, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        X = np.asarray(X, dtype=float).flatten()
        n = len(X)

        # Initialize parameters
        np.random.seed(42)
        random_indices = np.random.choice(n, self.k, replace=False)
        self.mu = X[random_indices]
        self.sigma = np.ones(self.k) * np.std(X)
        self.weights = np.ones(self.k) / self.k

        print(f"Initial Means: {self.mu.round(3)}")
        print(f"Initial Standard Deviations: {self.sigma.round(3)}")
        print(f"Initial Mixture Weights: {self.weights.round(3)}\n")

        for iteration in range(1, self.max_iters + 1):
            # --- E-Step: Calculate responsibilities (posterior probabilities) ---
            responsibilities = np.zeros((n, self.k))
            for j in range(self.k):
                responsibilities[:, j] = self.weights[j] * norm.pdf(X, loc=self.mu[j], scale=self.sigma[j])

            # Normalize responsibilities across components
            sum_resp = responsibilities.sum(axis=1, keepdims=True) + 1e-10
            responsibilities /= sum_resp

            # --- M-Step: Update parameters using responsibilities ---
            effective_n = responsibilities.sum(axis=0)
            new_mu = np.sum(responsibilities * X[:, np.newaxis], axis=0) / effective_n
            new_sigma = np.sqrt(
                np.sum(responsibilities * ((X[:, np.newaxis] - new_mu) ** 2), axis=0) / effective_n
            )
            new_weights = effective_n / n

            # Check convergence
            change = np.max(np.abs(new_mu - self.mu))
            self.mu, self.sigma, self.weights = new_mu, new_sigma, new_weights

            if change < self.tol:
                print(f"EM converged at iteration {iteration}!")
                break

        return self


def main():
    print("=" * 60)
    print("EXPERIMENT 10: EXPECTATION-MAXIMIZATION (EM) ALGORITHM")
    print("=" * 60)

    # Generate synthetic bimodal data from two distinct normal distributions
    np.random.seed(42)
    cluster1 = np.random.normal(loc=2.0, scale=0.8, size=100)
    cluster2 = np.random.normal(loc=7.5, scale=1.2, size=150)
    X = np.concatenate([cluster1, cluster2])

    print("Dataset Summary:")
    print(f"Total samples: {len(X)} (Generated from Normal(2.0, 0.8) and Normal(7.5, 1.2))\n")

    # 1. Run Scratch EM Algorithm
    print("--- Running Custom EM Algorithm ---")
    em = GMM_EM_1D(k=2, max_iters=100)
    em.fit(X)

    print("\nCustom EM Final Results:")
    for k in range(em.k):
        print(f"Component {k+1}: Mean = {em.mu[k]:.4f}, Std Dev = {em.sigma[k]:.4f}, Weight = {em.weights[k]:.4f}")
    print("-" * 60)

    # 2. Run Scikit-learn GaussianMixture for verification
    print("--- Scikit-Learn GaussianMixture Results ---")
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X.reshape(-1, 1))

    for k in range(2):
        mean = gmm.means_[k][0]
        std = np.sqrt(gmm.covariances_[k][0][0])
        weight = gmm.weights_[k]
        print(f"Component {k+1}: Mean = {mean:.4f}, Std Dev = {std:.4f}, Weight = {weight:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
