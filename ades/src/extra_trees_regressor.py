import model
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
import data


print("")
print("********************************************")
print ("Getting data...")
df = data.get_data("categorical_new")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(ExtraTreesRegressor(n_estimators=50, random_state=42), X, y, export=False)
b = model.predict(tree, X_test, y_test)

# model.apply_cross_validation(X, y, ExtraTreesRegressor(n_estimators=50, random_state=42), folds=10)

"""
n_estimators = 10
Variance Score: 0.98899408094
MAE: 0.00610595273265
MSE: 0.00491614346015
R2 SCORE: 0.988993795715

n_estimators = 50
Variance Score: 0.99053421176
MAE: 0.00546669770124
MSE: 0.00422819095212
R2 SCORE: 0.990533975717
"""


"""
Results with CV

Cross Validation in progress...
Time spent on calculating cross validation scores: 120.42 minutes.
r2: 0.982920525643
Standard Dev.: 0.0117557336957
"""

"""
Variance Score: 0.991126999906
MAE: 0.00500302369331
MSE: 0.00387441230134
R2 SCORE: 0.991126969712
Time spent on making prediction: 0.33 minutes.


Results with new features.
"""