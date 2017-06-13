import data
import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from time import time
from pandas import DataFrame
import eda_trees


def fit(model, X, y, export=False, mode="test"):
    """
    Fits a given model to the test data. Can use random sub sampling for testing


    :param model: Machine Learning model -> Python object
    :param X: Data Frame, with features
    :param y: Labels
    :param export: boolean, true if it is intended to export the fitted model to the models folder
    :param mode: test for random subsampling, other to fit the model to the whole data
    :return: fitted model
            if mode = test, also returs the X and y for testing
    """

    alg = model

    print ("")
    print ("********************************************")
    print ("Fitting model...")

    start = time()

    if mode == "test":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
        alg.fit(X_train, y_train)
        # eda_trees.show_importance(alg, X)
        return_value = (alg, X_test, y_test)

    else:
        alg.fit(X, y)
        return_value = alg

    metrics.print_time(time() - start, "fitting model")
    print ("********************************************")
    print ("")

    if export:
        data.export_to_pickle(alg, model.__class__.__name__)

    return return_value


def predict(fitted_model, X_test, y_test, export=False):
    """
    Given a fitted model, and test data, makes predictions

    :param fitted_model: Model, already fitted
    :param X_test: Test features
    :param y_test: Test labels
    :param export: If true, exports the predicted y
    :return: predicted labels
    """

    begin_time = time()
    print ("Making prediction...")

    y_pred = fitted_model.predict(X_test)
    metrics.print_metrics(y_test, y_pred)

    metrics.print_time(time() - begin_time, "making prediction")
    print ("********************************************")
    print ("")

    if export:
        df = DataFrame()
        df["y_test"] = y_test
        df["y_pred"] = y_pred

        df.to_csv("/Users/mzamith/Desktop/MESW/ADS/ades_bit/ades/src/models/pred.csv")

    return y_pred


def apply_cross_validation(X, y, model, folds=5, scoring='r2'):

    """
    Applies cross validation and shows scores

    :param X: Train features
    :param y: Train labels
    :param folds: Number of folds for CV
    :param scoring

    Possible values for scoring:

    neg_mean_absolute_error
    neg_mean_squared_error
    neg_median_absolute_error
    r2
    """

    begin = time()
    print ("********************************************")
    print ("Cross Validation in progress...")
    print ("n-folds: " + str(folds))

    scores = cross_val_score(model, X, y, cv=folds, scoring=scoring)

    metrics.print_time(time() - begin, "calculating cross validation scores")
    print ("")

    mean = scores.mean()
    mean = mean if scoring is "r2" else (mean * -1.0)
    max = scores.max()
    min = scores.min()
    std = scores.std()

    print ("Scoring is " + scoring)
    print ("Mean: " + str(mean))
    print ("Maximum: " + str(max))
    print ("Minimum: " + str(min))
    print ("Standard Dev.: " + str(std))
    print ("********************************************")
    print ("")


# df = pd.DataFrame(data=y_pred, columns=["ypred"])
# df["ytest"] = y_test.tolist()

# print(df.head())
# print(df.info())
# print(df.describe())

#
# sns.distplot(y_test)
# sns.distplot(y_pred)
# sns.plt.show()



