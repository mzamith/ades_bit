import model
import data
from sklearn.linear_model import Lasso

X = data.import_feature("no_categorical_new_new")
y = data.import_feature("labels")

tree, X_test, y_test = model.fit(Lasso(alpha=0.005), X, y, export=True)
b = model.predict(tree, X_test, y_test)
