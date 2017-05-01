from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np


def apply_linear_regression(X, y, test_size=0.3, random_state=21):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    reg_all = LinearRegression()
    reg_all.fit(X_train, y_train)
    y_pred = reg_all.predict(X_test)

    return y_pred


def apply_linear_regression_cv(X, y, test_size=0.3, random_state=21, cv=5):
    reg = LinearRegression()
    cv_results = cross_val_score(reg, X, y, cv=cv)
    return cv_results


def score_linear_regression(regr, scores, out=True):

    # scores should be a tuple with (x_test, y_test)
    # The coefficients

    X_test = scores[0]
    y_test = scores[1]

    coef = regr.coef
    mean_sq = np.mean((regr.predict(scores[0]) - scores[1]) ** 2)
    variance = regr.score(scores[0], scores[1])

    if out:
        print('Coefficients: \n', coef)
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_sq)
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % variance)

    return coef, mean_sq, variance
