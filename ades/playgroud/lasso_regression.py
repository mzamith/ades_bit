from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


boston = pd.read_csv('boston.csv')
X = boston.drop('MV', axis=1).values
names = boston.drop('MV', axis=1).columns
y = boston['MV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lasso = linear_model.Lasso(alpha=0.1, normalize=True)
#lasso.fit(X_train, y_train)

lasso_coef = lasso.fit(X, y).coef_
plt.plot(range(len(names)), lasso_coef)
plt.xticks(range(len(names)), names, rotation=60)
plt.ylabel('Coefficients')
plt.show()

lasso_pred = lasso.predict(X_test)

print(lasso.score(X_test, y_test))
