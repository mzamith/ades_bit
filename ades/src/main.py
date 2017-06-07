import model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

tree, X_test, y_test = model.fit(RandomForestRegressor(), resampling="normal")
b = model.predict(tree, X_test, y_test)