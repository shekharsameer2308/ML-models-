# Principal Component Analysis (PCA)

## Overview
Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

## Type
- Unsupervised Learning
- Dimensionality Reduction

## How it Works
PCA identifies the hyperplanes that are closest to the data points and projects the data onto them. It does this by calculating the eigenvectors and eigenvalues of the covariance matrix of the data. The first principal component captures the maximum variance, the second captures the remaining maximum variance orthogonal to the first, and so on.

## Pros and Cons
**Pros:**
- Reduces dimensionality and thus speeds up other machine learning algorithms.
- Helps alleviate the curse of dimensionality and overfitting.
- Useful for data visualization (projecting high-dimensional data to 2D or 3D).

**Cons:**
- Features become less interpretable (principal components are linear combinations of original features).
- Assumes linear relationships between features.
- Can be affected by outliers.

## Common Use Cases
- Data visualization of high-dimensional datasets.
- Preprocessing step to compress data before applying other algorithms.
- Noise filtering.

## Code Example
```python
from sklearn.decomposition import PCA
import numpy as np

# Sample high-dimensional data (e.g., 3D)
X = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]])

# Model initialization to reduce to 1 dimension
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```
