import model
import data
from sklearn.linear_model import LinearRegression

X = data.import_feature("no_categorical_new_imp")
y = data.import_feature("labels")


tree, X_test, y_test = model.fit(LinearRegression(), X, y, export=True)
b = model.predict(tree, X_test, y_test)

"""
Time spent on fitting model: 1.58 minutes.
********************************************

Making prediction...
Variance Score: 0.76056363692
MAE: 0.0820052909146
MSE: 0.10455001251
RMSE: 0.323341943629
R2 SCORE: 0.760563575735
Time spent on making prediction: 0.08 minutes.
********************************************
"""