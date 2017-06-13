import model
import data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

df = data.get_data("categorical_new_imp")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(GradientBoostingRegressor(n_estimators=150), X, y, export=True)
b = model.predict(tree, X_test, y_test)

"""
n_estimators = 150

Time spent on fitting model: 6.82 minutes.
********************************************

Making prediction...
Variance Score: 0.826744303435
MAE: 0.065950936033
MSE: 0.0756521994694
RMSE: 0.275049449135
R2 SCORE: 0.826744237577
Time spent on making prediction: 0.04 minutes.
********************************************

"""