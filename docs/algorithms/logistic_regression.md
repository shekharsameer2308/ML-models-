# Logistic Regression

## Overview
Despite its name, Logistic Regression is used for classification problems, not regression. It is a predictive analysis algorithm based on the concept of probability. It uses a logistic function (sigmoid) to model a binary dependent variable.

## Type
- Supervised Learning
- Classification (Binary or Multi-class)

## How it Works
It calculates the weighted sum of inputs and passes them through a sigmoid activation function `f(x) = 1 / (1 + e^-x)`, which maps any real value into a value between 0 and 1. This output can be interpreted as the probability of the input belonging to a particular class. A threshold (typically 0.5) is used to classify the output.

## Pros and Cons
**Pros:**
- Highly interpretable and fast to train.
- Outputs calibrated probabilities.
- Less prone to overfitting in low dimensional datasets.

**Cons:**
- Assumes a linear decision boundary between classes.
- Cannot solve non-linear problems without feature engineering.

## Common Use Cases
- Spam detection in emails.
- Disease prediction (e.g., probability of diabetes).
- Customer churn prediction.

## Code Example
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data
X = np.array([[1, 2], [2, 3], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = LogisticRegression()
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[3, 4]]))
probabilities = model.predict_proba(np.array([[3, 4]]))
print(f"Prediction: {predictions[0]}, Probabilities: {probabilities[0]}")
```
