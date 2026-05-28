# K-Means Clustering

## Overview
K-Means is a centroid-based clustering algorithm that partitions data into K distinct, non-overlapping clusters.

## Type
- Unsupervised Learning
- Clustering

## How it Works
The algorithm starts by randomly initializing K cluster centroids. Then, it iterates through two steps:
1. **Assignment step:** Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.
2. **Update step:** The centroids are recomputed as the center of mass (mean) of all data points assigned to that cluster.
The algorithm converges when the assignments no longer change.

## Pros and Cons
**Pros:**
- Simple, easy to understand and implement.
- Scales well to large datasets.
- Guarantees convergence (though to a local minimum).

**Cons:**
- You must specify the number of clusters (K) beforehand.
- Sensitive to initial centroid placement.
- Assumes clusters are spherical and of similar size, struggling with complex shapes or varying densities.
- Sensitive to outliers.

## Common Use Cases
- Customer segmentation based on purchase history.
- Document clustering.
- Image segmentation.

## Code Example
```python
from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Model initialization and training
model = KMeans(n_clusters=2, random_state=0, n_init='auto')
model.fit(X)

# Prediction (cluster assignment)
predictions = model.predict(np.array([[0, 0], [12, 3]]))
print(f"Cluster assignments: {predictions}")
```
