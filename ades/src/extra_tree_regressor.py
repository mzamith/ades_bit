import model
import data
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import LabelEncoder

df = data.get_data("processed_no_encoding")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(ExtraTreeRegressor(), X, y, export=True)
b = model.predict(tree, X_test, y_test)

"""
Time spent on fitting model: 0.36 minutes.
****************************************

Making prediction...
Variance Score: 0.97486010329
MAE: 0.00920371787561
MSE: 0.0109773529804
R2 SCORE: 0.974860087699
Time spent on making prediction: 0.01 minutes.
****************************************
"""
