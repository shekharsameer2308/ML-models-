import pandas as pd
from sklearn.svm import SVC
df = pd.read_csv('../data/svm_data.csv')
X, y = df[['Radius', 'Texture']], df['Malignant']
model = SVC(kernel='linear')
model.fit(X, y)
print("Accuracy:", model.score(X, y))
