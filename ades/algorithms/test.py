import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print np.unique(iris_y)


