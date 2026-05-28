# Gradient Boosting

## Overview
Gradient Boosting is an ensemble technique that builds models sequentially, with each new model attempting to correct the errors of the previous ones. It's often implemented using decision trees (Gradient Boosted Trees). Popular implementations include XGBoost, LightGBM, and CatBoost.

## Type
- Supervised Learning
- Classification and Regression

## How it Works
Unlike Random Forest which builds trees independently, Gradient Boosting builds trees sequentially. It trains a weak learner (like a shallow decision tree), evaluates its errors (residuals), and then trains the next weak learner specifically to predict and correct those residuals. The final prediction is the weighted sum of the predictions from all the models. It uses gradient descent to minimize the loss function.

## Pros and Cons
**Pros:**
- Often provides state-of-the-art results on tabular data.
- Flexible (can optimize on different loss functions).
- Handles missing data well (depending on the implementation).

**Cons:**
- Prone to overfitting if not tuned properly (e.g., learning rate, number of trees).
- Computationally expensive and slower to train than Random Forests.
- Harder to tune due to many hyperparameters.

## Common Use Cases
- Kaggle competitions (tabular data).
- Search ranking.
- Click-through rate prediction.

## Code Example
```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Sample data
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([0, 1, 1, 0])

# Model initialization and training
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X, y)

# Prediction
predictions = model.predict(np.array([[0.5, 0.5]]))
print(f"Prediction: {predictions[0]}")
```
