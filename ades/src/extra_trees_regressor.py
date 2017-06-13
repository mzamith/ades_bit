import model
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
import data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("")
print("********************************************")
print ("Getting data...")
df = data.get_data("categorical_new_imp")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

for m in xrange(6, 16):
# print m
    tree, X_test, y_test = model.fit(ExtraTreesRegressor(n_estimators=50, max_features=m, random_state=42, n_jobs=-1), X, y, export=False)
    b = model.predict(tree, X_test, y_test)

# model.apply_cross_validation(X, y, ExtraTreesRegressor(n_estimators=50, random_state=42), folds=10)


"""
********************************************
Getting data...
1

********************************************
Fitting model...
Time spent on fitting model: 1.04 minutes.
********************************************

Making prediction...
Variance Score: 0.86150108996
MAE: 0.0527868885594
MSE: 0.0604767497898
RMSE: 0.24592021021
R2 SCORE: 0.861498469745
Time spent on making prediction: 0.43 minutes.
********************************************

2

********************************************
Fitting model...
Time spent on fitting model: 1.58 minutes.
********************************************

Making prediction...
Variance Score: 0.867646127646
MAE: 0.0520649512215
MSE: 0.0577928733384
RMSE: 0.240401483644
R2 SCORE: 0.867644980542
Time spent on making prediction: 0.56 minutes.
********************************************

3

********************************************
Fitting model...
Time spent on fitting model: 2.03 minutes.
********************************************

Making prediction...
Variance Score: 0.872213831474
MAE: 0.0517294677309
MSE: 0.0557980651772
RMSE: 0.236216140806
R2 SCORE: 0.872213413598
Time spent on making prediction: 0.42 minutes.
********************************************

4

********************************************
Fitting model...
Time spent on fitting model: 2.03 minutes.
********************************************

Making prediction...
Variance Score: 0.872520857552
MAE: 0.0515561649868
MSE: 0.0556639016301
RMSE: 0.235931985178
R2 SCORE: 0.872520669802
Time spent on making prediction: 0.4 minutes.
********************************************

5

********************************************
Fitting model...
Time spent on fitting model: 2.64 minutes.
********************************************

Making prediction...
Variance Score: 0.87394587922
MAE: 0.051432188126
MSE: 0.0550415908904
RMSE: 0.23460944331
R2 SCORE: 0.873945861963
Time spent on making prediction: 0.52 minutes.
********************************************

"""



