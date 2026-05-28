import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../data/random_forest_data.csv')
X, y = df[['CapShape', 'CapColor']], df['Poisonous']
model = RandomForestClassifier()
model.fit(X, y)
print("Accuracy:", model.score(X, y))
