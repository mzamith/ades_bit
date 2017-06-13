import model
import data
from sklearn.linear_model import Lasso

X = data.import_feature("no_categorical_new_imp")
y = data.import_feature("labels")


tree, X_test, y_test = model.fit(Lasso(alpha=0.005), X, y, export=True)
b = model.predict(tree, X_test, y_test)

"""
Time spent on fitting model: 1.81 minutes.
********************************************

Making prediction...
Variance Score: 0.759241732399
MAE: 0.0751772554499
MSE: 0.10512722049
RMSE: 0.324233280973
R2 SCORE: 0.759241676181
Time spent on making prediction: 0.09 minutes.
********************************************
"""