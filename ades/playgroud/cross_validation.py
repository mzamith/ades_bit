from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import pandas as pd
import numpy as np

boston = pd.read_csv('boston.csv')
print(boston.head())

X = boston.drop('MV', axis=1).values
y = boston['MV'].values

reg = linear_model.LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5)

print(cv_results)
print(np.mean(cv_results))