import model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import data


print("")
print("********************************************")
print ("Getting data...")
df = data.get_data("processed_no_encoding")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

tree, X_test, y_test = model.fit(RandomForestRegressor(), X, y, resampling="normal", export=True)
b = model.predict(tree, X_test, y_test)

"""
Variance Score: 0.985952220531
MAE: 0.00688990670877
MSE: 0.00627474991772
R2 SCORE: 0.985952163522
"""