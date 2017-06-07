import preprocessing
import data

a = data.import_feature("processed_dataset_categorical_2")

print( a[a.columns[1]].dtype == "int64")

print (a.head())
print (a.info())
print (a.describe())