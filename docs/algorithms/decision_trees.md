# Decision Trees

## Overview
A Decision Tree is a flowchart-like tree structure where an internal node represents a feature (or attribute), the branch represents a decision rule, and each leaf node represents the outcome.

## Type
- Supervised Learning
- Classification and Regression

## How it Works
The algorithm splits the data into subsets based on the value of input features. This splitting process is repeated recursively in a top-down manner. The splits are chosen to maximize "Information Gain" or minimize "Gini Impurity" (for classification) or minimize variance (for regression) at each step, creating a set of rules that lead to a prediction.

## Pros and Cons
**Pros:**
- Simple to understand and visualize.
- Requires little data preparation (no need for normalization/scaling).
- Can handle both numerical and categorical data.

**Cons:**
- Highly prone to overfitting, creating overly complex trees.
- Unstable; a small change in data can lead to a completely different tree.

## Common Use Cases
- Credit scoring models.
- Medical diagnosis.
- Customer segmentation.

## Code Example
```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Sample data
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([0, 1, 1, 0]) # XOR-like data

# Model initialization and training
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[0.5, 0.5]]))
print(f"Prediction: {predictions[0]}")
```
