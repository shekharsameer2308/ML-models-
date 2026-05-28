# Multi-Layer Perceptron (MLP)

## Overview
A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network (ANN). It consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer.

## Type
- Deep Learning
- Classification and Regression

## How it Works
Each node (except in the input layer) is a neuron that uses a nonlinear activation function (like ReLU, Sigmoid, or Tanh). Data flows in one direction from input to output. The network learns by updating the weights of the connections between neurons using a technique called backpropagation, which minimizes the error between the predicted output and the actual output using gradient descent.

## Pros and Cons
**Pros:**
- Can model complex, non-linear relationships.
- Very flexible architecture.
- Performs well on a wide variety of tasks.

**Cons:**
- Requires large amounts of data to train effectively without overfitting.
- Computationally expensive to train.
- Considered a "black box" as it's hard to interpret the internal representations.

## Common Use Cases
- Tabular data classification and regression.
- Basic pattern recognition.
- Approximation of complex mathematical functions.

## Code Example
```python
from sklearn.neural_network import MLPClassifier
import numpy as np

# Sample data (XOR problem)
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, random_state=1)
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[0, 1], [1, 1]]))
print(f"Predictions: {predictions}")
```
