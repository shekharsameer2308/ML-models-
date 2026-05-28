import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('../data/logistic_regression_data.csv')
X, y = df[['Feature1', 'Feature2']], df['Target']
model = LogisticRegression()
model.fit(X, y)
print("Accuracy:", model.score(X, y))
