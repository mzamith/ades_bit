#KNEIGHBORS

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# sadas
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])

print knn