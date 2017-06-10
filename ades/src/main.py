import model
import data
from sklearn.neural_network import MLPRegressor

# df = data.get_data("processed_no_encoding")

# X = df[0]
# y = df[1]

X = data.import_feature("processed_shrinked_30")
print("Imported features")
print("")
y = data.import_feature("labels")
print("Imported labels")
print("")

# print("assign labels")
# df["labels"] = y.values

# print("create sample")
# X = df.sample(n=df.shape[0]//4)
# X.to_hdf("/Users/mzamith/Desktop/MESW/ADS/ades_bit/ades/src/features/shrinked_sample.h5", 'table', append=True)

# print("retrieve labels")
# y = X["labels"]

# print("drop labels")
# X.drop("labels", axis=1, inplace=True)

# print (X.info())

activation = 'relu'
solver = ['lbfgs', 'sgd', 'adam']

tree, X_test, y_test = model.fit(MLPRegressor(), X, y, resampling="normal", export=True)
b = model.predict(tree, X_test, y_test)

"""
Starting for solver lbfgs
Fitting model...
Time spent on fitting model: 40.01 minutes.
****************************************

Making prediction...
Variance Score: 0.980465396924
MAE: 0.0266214234425
MSE: 0.00872609831717
R2 SCORE: 0.980464113493
Time spent on making prediction: 0.06 minutes.
****************************************

Starting for solver sgd
Fitting model...
Time spent on fitting model: 1.7 minutes.
****************************************

Making prediction...
Variance Score: 0.973845862484
MAE: 0.0335222668908
MSE: 0.0116854699427
R2 SCORE: 0.973838707028
Time spent on making prediction: 0.04 minutes.
****************************************

Starting for solver adam
Fitting model...
Time spent on fitting model: 2.78 minutes.
****************************************

Making prediction...
Variance Score: 0.958130998713
MAE: 0.0312351057801
MSE: 0.0187512157458
R2 SCORE: 0.958019998245
Time spent on making prediction: 0.08 minutes.
****************************************

"""