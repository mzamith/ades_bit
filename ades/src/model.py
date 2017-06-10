import data
import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from time import time


def fit(model, X, y, export=False, mode="test"):

    alg = model

    print("Fitting model...")

    start = time()

    if mode == "test":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
        alg.fit(X_train, y_train)
        return_value = (alg, X_test, y_test)

    else:
        alg.fit(X, y)
        return_value = alg

    metrics.print_time(time() - start, "fitting model")
    print ("****************************************")
    print ("")

    if export:
        data.export_to_pickle(alg, model.__class__.__name__)

    return return_value


def predict(fitted_model, X_test, y_test):

    begin_time = time()
    print("Making prediction...")

    y_pred = fitted_model.predict(X_test)
    metrics.print_metrics(y_test, y_pred)

    metrics.print_time(time() - begin_time, "making prediction")
    print ("****************************************")
    print ("")
    return y_pred


def apply_cross_validation(X, y, model, folds=5, scoring='r2'):

    begin = time()
    print("Cross Validation in progress...")

    scores = cross_val_score(model, X, y, cv=folds, scoring=scoring)

    metrics.print_time(time() - begin, "calculating cross validation scores")
    print (scoring + ": " + str(scores.mean() * -1.0))
    print ("Standard Dev.: " + str(scores.std()))
    print ("****************************************")
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




