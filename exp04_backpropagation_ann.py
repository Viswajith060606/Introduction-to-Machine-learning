"""
Experiment 04: Artificial Neural Network with Backpropagation
Description: Build an Artificial Neural Network by implementing the
Backpropagation algorithm and test the same using appropriate data sets.
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Initialize weights and biases with random values
        np.random.seed(42)
        self.W1 = np.random.uniform(-1, 1, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.uniform(-1, 1, (self.hidden_dim, self.output_dim))
        self.b2 = np.zeros((1, self.output_dim))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1.0 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        # Compute error and delta for output layer
        error = y - output
        d_output = error * self.sigmoid_derivative(output)

        # Compute error and delta for hidden layer
        error_hidden = np.dot(d_output, self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 += np.dot(self.a1.T, d_output) * self.lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
        self.W1 += np.dot(X.T, d_hidden) * self.lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=5000, log_every=1000):
        print(f"Training Neural Network for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean(np.square(y - output))
            if epoch % log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:5d}/{epochs} - Mean Squared Error Loss: {loss:.6f}")


def main():
    print("=" * 60)
    print("EXPERIMENT 4: ARTIFICIAL NEURAL NETWORK WITH BACKPROPAGATION")
    print("=" * 60)

    # Classic non-linearly separable XOR problem
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    print("Training Dataset (XOR Truth Table):")
    for i in range(len(X)):
        print(f"Inputs: {X[i]} -> Target: {y[i][0]}")
    print("-" * 60)

    # Build and train ANN
    ann = NeuralNetwork(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5)
    ann.train(X, y, epochs=5000, log_every=1000)

    # Testing and evaluation
    print("-" * 60)
    print("Testing the Trained ANN:")
    predictions = ann.forward(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} | Target: {int(y[i][0])} | Predicted Prob: {predictions[i][0]:.4f} | Final Class: {int(round(predictions[i][0]))}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
