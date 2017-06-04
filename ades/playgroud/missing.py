import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('indians.csv')
print(df.shape)

#dropping missing values
df.insulin.replace(0, np.nan, inplace=True)
df.triceps.replace(0, np.nan, inplace=True)
df.bmi.replace(0, np.nan, inplace=True)

df = df.dropna()
print(df.shape)
print(df.info())

#Imputing missing data
df = pd.read_csv('indians.csv')

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(df)
X = imp.transform(df)

#Doint it all at once
df = pd.read_csv('indians.csv')
X = df.drop('diabetes', axis=1).values
y = df.diabetes.values

df.insulin.replace(0, np.nan, inplace=True)
df.triceps.replace(0, np.nan, inplace=True)
df.bmi.replace(0, np.nan, inplace=True)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
logreg = LinearRegression()
steps = [('imputation', imp), ('linear_regression', logreg)]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print(pipeline.score(X_test, y_test))