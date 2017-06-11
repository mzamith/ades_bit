import data
import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from time import time


def fit(model, X, y, export=False, mode="test"):

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


def predict(fitted_model, X_test, y_test):

    begin_time = time()
    print ("Making prediction...")

    y_pred = fitted_model.predict(X_test)
    metrics.print_metrics(y_test, y_pred)

    metrics.print_time(time() - begin_time, "making prediction")
    print ("********************************************")
    print ("")
    return y_pred


def apply_cross_validation(X, y, model, folds=5, scoring='r2'):

    """
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




