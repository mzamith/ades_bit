import sklearn.metrics as m
from math import sqrt


def print_metrics(y_test, y_pred):
    """
    Print the standard metrics for regression tasks


    :param y_test: the true labels
    :param y_pred: the predicted labels
    """

    print ("Variance Score: " + str(m.explained_variance_score(y_test, y_pred)))
    print ("MAE: " + str(m.mean_absolute_error(y_test, y_pred)))
    print ("MSE: " + str(m.mean_squared_error(y_test, y_pred)))
    print ("RMSE: " + str(sqrt(m.mean_squared_error(y_test, y_pred))))
    print ("R2 SCORE: " + str(m.r2_score(y_test, y_pred)))


def print_time(seconds, task):
    """
    This is utilitary. Helps print the time it took to accomplish a given task.

    :param seconds: seconds it took
    :param task: task
    :return:
    """

    print ("Time spent on " + task + ": " + str(round((seconds / 60.0), 2)) + " minutes.")
