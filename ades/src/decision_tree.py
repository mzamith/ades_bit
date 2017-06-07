import data
import pandas as pd
import numpy as np
import eda
import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns

print("Getting data...")
dt = data.get_data(label="processed_no_encoding")
X = dt[0]
y = dt[1]

tree = DecisionTreeRegressor()
X = X.apply(LabelEncoder().fit_transform)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Fitting model...")
# tree.fit(X_train, y_train)
tree.fit(X, y)


print("Making prediction...")
# y_pred = tree.predict(X)
scores = cross_val_score(tree, X, y, cv=5, scoring='r2')

# df = pd.DataFrame(data=y_pred, columns=["ypred"])
# df["ytest"] = y_test.tolist()

# print(df.head())
# print(df.info())
# print(df.describe())

# metrics.print_metrics(y_test, y_pred)
# sns.distplot(y_test)
# sns.distplot(y_pred)
# sns.plt.show()

print (scores.mean() * -1.0)
print (scores.std())


