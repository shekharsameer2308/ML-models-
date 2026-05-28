# Random Forest

## Overview
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## Type
- Supervised Learning
- Classification and Regression

## How it Works
It builds multiple decision trees by training each one on a random subset of the training data (bootstrap aggregating or bagging). Furthermore, when splitting a node, it only considers a random subset of the features. This randomness helps make the model robust and prevents the overfitting that plagues individual decision trees.

## Pros and Cons
**Pros:**
- Reduces overfitting compared to individual decision trees.
- High accuracy and robustness.
- Can handle missing values and maintain accuracy.
- Provides feature importance.

**Cons:**
- Less interpretable than a single decision tree.
- Can be slow to train on large datasets due to multiple trees.
- Requires more memory.

## Common Use Cases
- Fraud detection in banking.
- Recommender systems.
- Feature selection.

## Code Example
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([0, 1, 1, 0])

# Model initialization and training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[0.5, 0.5]]))
print(f"Prediction: {predictions[0]}")
```
