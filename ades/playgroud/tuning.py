from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from time import time


param_grid = {'n_neighbors': np.arange(1, 50)}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)

# Load the digits dataset: digits
digits = datasets.load_digits()

X = digits.data
y = digits.target

t1 = time()
# Split into training and test set
knn_cv.fit(X, y)
t2 = time()

print (knn_cv.best_params_)
print (knn_cv.best_score_)
print ("Took {} seconds".format(str(t2-t1)))