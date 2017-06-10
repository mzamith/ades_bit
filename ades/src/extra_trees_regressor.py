import model
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
import data


print("")
print("********************************************")
print ("Getting data...")
df = data.get_data("processed_no_encoding")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(ExtraTreesRegressor(n_estimators=50), X, y, export=False)
b = model.predict(tree, X_test, y_test)

# model.apply_cross_validation(X, y, ExtraTreesRegressor(n_estimators=50), folds=10)

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