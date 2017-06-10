import preprocessing
import data
from time import time
import numpy as np

alphas = np.logspace(-4, -0.5, 5)
print alphas
print len(alphas)


labels = data.import_feature("labels")

a = time()
X = data.import_feature("processed_shrinked")
print (str((time() - a) / 60.0))

print (X.head())
print (X.info())
print (X.describe())
