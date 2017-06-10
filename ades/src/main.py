import model
import data
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import LabelEncoder

df = data.get_data("processed_no_encoding")

X = df[0]
y = df[1]

X = X.apply(LabelEncoder().fit_transform)

# X = data.import_feature("processed_shrinked_30")
# print("Imported features")
# print("")
# y = data.import_feature("labels")
# print("Imported labels")
# print("")

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

# activation = 'relu'
# solver = ['lbfgs', 'sgd', 'adam']

tree, X_test, y_test = model.fit(ExtraTreeRegressor(), X, y, export=True)
b = model.predict(tree, X_test, y_test)
