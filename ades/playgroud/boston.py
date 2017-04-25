from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


boston = pd.read_csv('boston.csv')
print(boston.head())

X = boston.drop('MV', axis=1).values
y = boston['MV'].values

X_rooms = X[:, 5]

y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

print(reg_all.score(X_test, y_test))



