import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('../data/decision_tree_data.csv')
X, y = df[['SepalLength', 'SepalWidth']], df['Species']
model = DecisionTreeClassifier()
model.fit(X, y)
print("Accuracy:", model.score(X, y))
