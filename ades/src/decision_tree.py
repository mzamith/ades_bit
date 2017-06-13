import model
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import data


print("")
print("********************************************")
print ("Getting data...")
df = data.get_data("categorical_new_new")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(DecisionTreeRegressor(random_state=42), X, y, export=False)
b = model.predict(tree, X_test, y_test)

# model.apply_cross_validation(X, y, ExtraTreesRegressor(n_estimators=50, random_state=42), folds=10)

"""
Variance Score: 0.769549902641
MAE: 0.0663817334165
MSE: 0.100627394658
R2 SCORE: 0.769547004523
"""