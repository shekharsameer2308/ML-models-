# Support Vector Machines (SVM)

## Overview
Support Vector Machines is a powerful algorithm that finds a hyperplane in an N-dimensional space (N — the number of features) that distinctly classifies the data points.

## Type
- Supervised Learning
- Classification and Regression (SVR)

## How it Works
SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable (using the "kernel trick"). It finds the hyperplane that has the maximum margin, which is the maximum distance between data points of both classes. The data points that are closest to the hyperplane are called Support Vectors.

## Pros and Cons
**Pros:**
- Effective in high dimensional spaces.
- Memory efficient since it uses a subset of training points in the decision function.
- Versatile through the use of different Kernel functions.

**Cons:**
- Not suitable for large datasets as the training time is high.
- Less effective on noisier datasets with overlapping classes.

## Common Use Cases
- Face detection.
- Text and hypertext categorization.
- Bioinformatics (e.g., protein classification).

## Code Example
```python
from sklearn.svm import SVC
import numpy as np

# Sample data
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

# Model initialization and training
model = SVC(kernel='linear')
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[-0.8, -1]]))
print(f"Prediction: {predictions[0]}")
```
