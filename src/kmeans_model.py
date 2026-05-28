import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('../data/kmeans_data.csv')
X = df[['Income', 'SpendingScore']]
model = KMeans(n_clusters=2, random_state=0)
model.fit(X)
print("Labels:", model.labels_)
