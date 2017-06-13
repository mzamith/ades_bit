'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import data
from sklearn.model_selection import train_test_split
import metrics


batch_size = 128
num_classes = 10
epochs = 20

X = data.import_feature("no_categorical_new_new")
y = data.import_feature("labels")

# X = X.apply(LabelEncoder().fit_transform)

# the data, shuffled and split between train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

n_cols = X_train.shape[1]
n_rows = X_train.shape[0]

model = Sequential()
model.add(Dense(300, activation='relu', input_shape=(n_cols,)))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=[keras.metrics.mae, keras.metrics.mse])

history = model.fit(X_train.as_matrix(), y_train.as_matrix())

model.save("/Users/mzamith/Desktop/MESW/ADS/ades_bit/ades/src/models/deep.h5")

score = model.evaluate(X_test.as_matrix(), y_test.as_matrix(), verbose=0)
predictions = model.predict(X_test.as_matrix())
print(score)

metrics.print_metrics(y_test.as_matrix(), predictions)


"""
Variance Score: 0.819308447774
MAE: 0.0802154791419
MSE: 0.0789028222245
RMSE: 0.280896461751
R2 SCORE: 0.819299786157

# 3 layers
Variance Score: 0.843542752554
MAE: 0.0959867180774
MSE: 0.0703985410233
RMSE: 0.265327233852
R2 SCORE: 0.838775964427
"""