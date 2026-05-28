import pandas as pd
from sklearn.neural_network import MLPClassifier
df = pd.read_csv('../data/mlp_data.csv')
X, y = df[['F1', 'F2']], df['Label']
model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000)
model.fit(X, y)
print("Accuracy:", model.score(X, y))
