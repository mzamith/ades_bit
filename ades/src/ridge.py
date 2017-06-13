import model
import data
from sklearn.linear_model import Ridge

X = data.import_feature("no_categorical_new_imp")
y = data.import_feature("labels")


tree, X_test, y_test = model.fit(Ridge(alpha=0.005), X, y, export=True)
b = model.predict(tree, X_test, y_test)

"""
Time spent on fitting model: 1.3 minutes.
********************************************

Making prediction...
Variance Score: 0.760563636936
MAE: 0.0820052908752
MSE: 0.104550012503
RMSE: 0.323341943618
R2 SCORE: 0.760563575752
Time spent on making prediction: 0.08 minutes.
********************************************
"""