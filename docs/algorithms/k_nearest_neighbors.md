# K-Nearest Neighbors (KNN)

## Overview
K-Nearest Neighbors is a simple, instance-based learning algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).

## Type
- Supervised Learning
- Classification and Regression

## How it Works
For a given data point, the algorithm finds the 'K' closest data points (neighbors) in the training dataset. For classification, it assigns the class that is most common among its K nearest neighbors (majority voting). For regression, it takes the average of the values of its K nearest neighbors. The distance can be measured using Euclidean, Manhattan, or Minkowski distance.

## Pros and Cons
**Pros:**
- Very simple and intuitive.
- No training phase (lazy learner).
- Can naturally handle multi-class cases.

**Cons:**
- Computationally expensive at test time, as it calculates the distance to all training instances.
- Sensitive to irrelevant features and the scale of the data (requires feature scaling).
- Performance degrades with high-dimensional data (curse of dimensionality).

## Common Use Cases
- Recommendation systems (basic collaborative filtering).
- Concept search.
- Pattern recognition.

## Code Example
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample data
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[1.1]]))
print(f"Prediction: {predictions[0]}")
```
