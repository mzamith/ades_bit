import data
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

print("Getting data...")
X = data.get_data()
y = data.get_labels()

lasso = DecisionTreeRegressor()
lb = LabelEncoder()
X = X.apply(lb.fit_transform)

print (X.info())
print (X.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Fitting model...")
lasso.fit(X_train, y_train)

print("Making prediction...")
y_pred = lasso.predict(X_test)

df = pd.DataFrame(data=y_pred, columns=["ypred"])
df["ytest"] = y_test.tolist()

print("MEAN " + str(np.mean(y)))

print(df.head())
print(df.info())
print(df.describe())

print(lasso.score(X_test, y_test))

print ("Variance Score: " + str(metrics.explained_variance_score(y_test, y_pred)))
print ("MAE: " + str(metrics.mean_absolute_error(y_test, y_pred)))
print ("MSW: " + str(metrics.mean_squared_error(y_test, y_pred)))
print ("R2 SCORE: " + str(metrics.r2_score(y_test, y_pred)))




