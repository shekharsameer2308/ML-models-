import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv('../data/gradient_boosting_data.csv')
X, y = df[['Rooms', 'Area']], df['Price']
model = GradientBoostingRegressor()
model.fit(X, y)
print("R2 Score:", model.score(X, y))
