import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('winequality-red.csv')

#print(df.shape)
#print(df.info())
#print(df.describe())

X = df.drop('quality', axis=1)
y = df.quality

y.loc[y < 5] = 1
y.loc[y >= 5] = 0

steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

parameter = {'knn__n_neighbors': np.arange(1, 50)}
cv = GridSearchCV(pipeline, parameter)
cv.fit(X_train, y_train)
y_pred_cv = cv.predict(X_test)

X_scaled = scale(X)

print("----SCALED----")
print(np.mean(X), np.std(X))
print(np.mean(X_scaled), np.std(X_scaled))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("----WITH CROSS VALIDATION----")
print(cv.best_params_)
print (cv.score(X_test, y_test))
print (classification_report(y_test, y_pred_cv))
