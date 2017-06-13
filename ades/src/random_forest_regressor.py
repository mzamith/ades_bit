import model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import data


print("")
print("********************************************")
print ("Getting data...")
df = data.get_data("categorical_new_imp")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(RandomForestRegressor(), X, y, export=True)
b = model.predict(tree, X_test, y_test)

"""
Time spent on fitting model: 4.78 minutes.
********************************************

Making prediction...
Variance Score: 0.858464866083
MAE: 0.052965225233
MSE: 0.0618039749905
RMSE: 0.248604052643
R2 SCORE: 0.858458909551
Time spent on making prediction: 0.13 minutes.
********************************************
"""