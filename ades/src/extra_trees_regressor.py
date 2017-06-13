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

tree, X_test, y_test = model.fit(ExtraTreesRegressor(n_estimators=100, max_features=5, random_state=42, n_jobs=-1), X, y, export=False)
b = model.predict(tree, X_test, y_test)

# model.apply_cross_validation(X, y, ExtraTreesRegressor(n_estimators=50, random_state=42), folds=10)


"""
BEST MODEL

100 estimators:
********************************************
Fitting model...
Time spent on fitting model: 3.66 minutes.
********************************************

Making prediction...
Variance Score: 0.875179749082
MAE: 0.0509412975854
MSE: 0.0545028188164
RMSE: 0.233458387762
R2 SCORE: 0.87517973708
Time spent on making prediction: 1.6 minutes.
********************************************


Process finished with exit code 0


OTHER TESTS, WITH 50 ESTIMATORS


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

6

********************************************
Fitting model...
Time spent on fitting model: 2.53 minutes.
********************************************

Making prediction...
Variance Score: 0.872437274821
MAE: 0.051218679835
MSE: 0.0557003425776
RMSE: 0.236009200197
R2 SCORE: 0.872437214143
Time spent on making prediction: 0.41 minutes.
********************************************

7

********************************************
Fitting model...
Time spent on fitting model: 3.12 minutes.
********************************************

Making prediction...
Variance Score: 0.8735560066
MAE: 0.0511547374178
MSE: 0.0552119070546
RMSE: 0.234972141018
R2 SCORE: 0.873555810424
Time spent on making prediction: 0.43 minutes.
********************************************

8

********************************************
Fitting model...
Time spent on fitting model: 3.19 minutes.
********************************************

Making prediction...
Variance Score: 0.873417232879
MAE: 0.0511097215943
MSE: 0.0552726108963
RMSE: 0.235101277955
R2 SCORE: 0.873416788816
Time spent on making prediction: 0.47 minutes.
********************************************

9

********************************************
Fitting model...
Time spent on fitting model: 3.37 minutes.
********************************************

Making prediction...
Variance Score: 0.872679078006
MAE: 0.0513872440314
MSE: 0.055595132053
RMSE: 0.235786199878
R2 SCORE: 0.872678163247
Time spent on making prediction: 0.35 minutes.
********************************************

10

********************************************
Fitting model...
Time spent on fitting model: 3.47 minutes.
********************************************

Making prediction...
Variance Score: 0.871275599652
MAE: 0.0514191664966
MSE: 0.0562080963898
RMSE: 0.237082467487
R2 SCORE: 0.871274375859
Time spent on making prediction: 0.36 minutes.
********************************************

11

********************************************
Fitting model...
Time spent on fitting model: 3.57 minutes.
********************************************

Making prediction...
Variance Score: 0.871107961812
MAE: 0.0514422526153
MSE: 0.0562814193249
RMSE: 0.237237053019
R2 SCORE: 0.871106454488
Time spent on making prediction: 0.34 minutes.
********************************************

12

********************************************
Fitting model...
Time spent on fitting model: 3.83 minutes.
********************************************

Making prediction...
Variance Score: 0.870221173016
MAE: 0.0515213847837
MSE: 0.0566689353623
RMSE: 0.238052379451
R2 SCORE: 0.870218980139
Time spent on making prediction: 0.36 minutes.
********************************************

13

********************************************
Fitting model...
Time spent on fitting model: 3.85 minutes.
********************************************

Making prediction...
Variance Score: 0.870590145617
MAE: 0.0515591277317
MSE: 0.0565079193284
RMSE: 0.237713944329
R2 SCORE: 0.870587732877
Time spent on making prediction: 0.33 minutes.
********************************************

14

********************************************
Fitting model...
Time spent on fitting model: 4.22 minutes.
********************************************

Making prediction...
Variance Score: 0.870489072462
MAE: 0.0516471403874
MSE: 0.0565522727172
RMSE: 0.237807217547
R2 SCORE: 0.870486156449
Time spent on making prediction: 0.35 minutes.
********************************************

15

********************************************
Fitting model...
Time spent on fitting model: 4.9 minutes.
********************************************

Making prediction...
Variance Score: 0.869528557366
MAE: 0.0517715991236
MSE: 0.0569719317161
RMSE: 0.238687937936
R2 SCORE: 0.869525069523
Time spent on making prediction: 0.42 minutes.
********************************************

"""



