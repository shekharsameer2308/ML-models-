# Linear Regression

## Overview
Linear regression is a linear model that assumes a linear relationship between the input variables (X) and the single output variable (y). More specifically, that y can be calculated from a linear combination of the input variables.

## Type
- Supervised Learning
- Regression

## How it Works
The algorithm finds the best-fitting straight line through the data points. The line is represented by the equation `y = wx + b`, where `w` is the weight (slope) and `b` is the bias (y-intercept). It optimizes these parameters by minimizing the mean squared error (MSE) between the predicted values and the actual target values using optimization algorithms like Ordinary Least Squares or Gradient Descent.

## Pros and Cons
**Pros:**
- Simple to implement and easy to interpret.
- Fast training time.
- Works well when the relationship is genuinely linear.

**Cons:**
- Prone to underfitting if the data is complex and non-linear.
- Sensitive to outliers.

## Common Use Cases
- Predicting house prices based on size, location, etc.
- Forecasting sales revenue.
- Trend analysis in finance.

## Code Example
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Model initialization and training
model = LinearRegression()
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[5]]))
print(f"Prediction for input 5: {predictions[0]}")
```
