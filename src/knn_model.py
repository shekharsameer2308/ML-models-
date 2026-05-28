import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../data/knn_data.csv')
X, y = df[['BillLength', 'BillDepth']], df['Species']
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print("Accuracy:", model.score(X, y))
