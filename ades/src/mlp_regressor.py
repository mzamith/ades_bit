import model
import data
from sklearn.neural_network import MLPRegressor

X = data.import_feature("processed_shrinked_30")
y = data.import_feature("labels")

# activation = 'relu'
# solver = ['lbfgs', 'sgd', 'adam']
# learning_rate = ['constant', 'invscaling', 'adaptive']

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
Time spent on fitting model: 2.11 minutes.
****************************************

Making prediction...
Variance Score: 0.985223147885
MAE: 0.0239015361123
MSE: 0.00663348963072
R2 SCORE: 0.985149021263
Time spent on making prediction: 0.07 minutes.
****************************************
"""


"""
Starting for learning_rate constant
Fitting model...
Time spent on fitting model: 1.73 minutes.
****************************************

Making prediction...
Variance Score: 0.97457855475
MAE: 0.0327082989387
MSE: 0.0113563411158
R2 SCORE: 0.974575556783
Time spent on making prediction: 0.04 minutes.
****************************************

Starting for learning_rate invscaling
Fitting model...
Time spent on fitting model: 1.09 minutes.
****************************************

Making prediction...
Variance Score: 0.958159407177
MAE: 0.0509118650102
MSE: 0.0186895294295
R2 SCORE: 0.958158100846
Time spent on making prediction: 0.04 minutes.
****************************************

Starting for learning_rate adaptive
Fitting model...
Time spent on fitting model: 6.07 minutes.
****************************************

Making prediction...
Variance Score: 0.979801955372
MAE: 0.0284042224114
MSE: 0.00902187294856
R2 SCORE: 0.97980193672
Time spent on making prediction: 0.04 minutes.
****************************************
"""