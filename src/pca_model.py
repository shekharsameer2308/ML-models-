import pandas as pd
from sklearn.decomposition import PCA
df = pd.read_csv('../data/pca_data.csv')
X = df[['F1', 'F2', 'F3']]
pca = PCA(n_components=2)
pca.fit(X)
print("Explained Variance:", pca.explained_variance_ratio_)
