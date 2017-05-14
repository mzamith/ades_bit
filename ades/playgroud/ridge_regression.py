from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


boston = pd.read_csv('boston.csv')
X = boston.drop('MV', axis=1).values
y = boston['MV'].values

print(boston.info())
print(boston.head())
print(boston.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ridge = linear_model.Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print(ridge.score(X_test, y_test))
