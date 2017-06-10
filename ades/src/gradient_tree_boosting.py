import model
import data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

df = data.get_data("processed_no_encoding")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(GradientBoostingRegressor(), X, y, export=True)
b = model.predict(tree, X_test, y_test)

"""
Time spent on fitting model: 6.61 minutes.
****************************************

Making prediction...
Variance Score: 0.953521629508
MAE: 0.0313627836122
MSE: 0.0202948010303
R2 SCORE: 0.953521625936
Time spent on making prediction: 0.03 minutes.
****************************************
"""