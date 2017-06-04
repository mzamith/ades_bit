import data
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

print("Getting data...")
X = data.get_data()
y = data.get_labels()

lasso = Lasso(alpha=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Fitting model...")
lasso.fit(X_train, y_train)

print("Making prediction...")
y_pred = lasso.predict(X_test)

df = pd.DataFrame(data=y_pred, columns=["ypred"])
df["ytest"] = pd.Series(y_test)

print("MEAN " + str(np.mean(y)))

print(df.head())
print(df.info())
print(df.describe())

print(lasso.score(X_test, y_test))
