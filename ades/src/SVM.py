import model
import data
from sklearn.svm import SVR

X = data.import_feature("no_categorical_new_new_100")
y = data.import_feature("labels")

tree, X_test, y_test = model.fit(SVR(), X, y, export=True)
b = model.predict(tree, X_test, y_test)
