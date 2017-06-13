import model
import data
from sklearn.neural_network import MLPRegressor

X = data.import_feature("no_categorical_new_new_100")
y = data.import_feature("labels")

activation =  ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']

tree, X_test, y_test = model.fit(MLPRegressor(random_state=42), X, y, export=True)
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
"""

"""
Starting for solver sgd
Fitting model...
Time spent on fitting model: 1.7 minutes.
****************************************

Making prediction...
Variance Score: 0.827676914563
MAE: 0.0784612583708
MSE: 0.0752931927525
RMSE: 0.274396050906
R2 SCORE: 0.827566420975
Time spent on making prediction: 0.05 minutes.
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

Variance Score: 0.838594121853
MAE: 0.0928624670983
MSE: 0.0708426704014
RMSE: 0.26616286443
R2 SCORE: 0.837758836378
****************************************

Starting for learning_rate invscaling
Fitting model...
Time spent on fitting model: 1.09 minutes.
****************************************

Variance Score: 0.838594121853
MAE: 0.0928624670983
MSE: 0.0708426704014
RMSE: 0.26616286443
R2 SCORE: 0.837758836378
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

"""
with new features
Time spent on fitting model: 3.04 minutes.

Variance Score: 0.841085631601
MAE: 0.0832768028231
MSE: 0.069590938374
RMSE: 0.263800944604
R2 SCORE: 0.840625505005
Time spent on making prediction: 0.04 minutes.
"""


"""
**************************
TESTING FOR ACTIVATION
**************************
"""

"""
Activation identity

Variance Score: 0.759060987156
MAE: 0.0780947388861
MSE: 0.105275138854
RMSE: 0.324461305634
R2 SCORE: 0.758902919225
Time spent on making prediction: 0.02 minutes.
"""

"""
Activation Logistic

Variance Score: 0.84135259582
MAE: 0.0901685071008
MSE: 0.0699682810403
RMSE: 0.264515181115
R2 SCORE: 0.839761329319
Time spent on making prediction: 0.04 minutes.
"""

"""
Activation tanh

Variance Score: 0.832526447689
MAE: 0.0973699265716
MSE: 0.0731274729766
RMSE: 0.270420918156
R2 SCORE: 0.83252626925
"""

"""
Activation Relu

Variance Score: 0.836527931538
MAE: 0.0826359910744
MSE: 0.0714875973403
RMSE: 0.267371646478
R2 SCORE: 0.836281849466
"""


"""
With PCA 100

Variance Score: 0.849068630433
MAE: 0.0817282543428
MSE: 0.0659263891176
RMSE: 0.256761346619
R2 SCORE: 0.849017915005
Time spent on making prediction: 0.13 minutes.
"""