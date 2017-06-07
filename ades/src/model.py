import data
import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score


def fit(model, data_label="processed_no_encoding", resampling="cv"):

    print ("")
    print ("****************************************")
    print ("Getting data...")

    dt = data.get_data(label=data_label)
    X = dt[0]
    y = dt[1]

    alg = model
    X = X.apply(LabelEncoder().fit_transform)

    print("Fitting model...")

    if not resampling == "cv":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        alg.fit(X_train, y_train)
    else:
        alg.fit(X, y)

    return alg, X_test, y_test


def predict(fitted_model, X_test, y_test):

    print("Making prediction...")

    y_pred = fitted_model.predict(X_test)
    metrics.print_metrics(y_test, y_pred)

    return y_pred


def apply_cross_validation(X, y, fitted_model, folds=5, scoring='r2'):

    print("Cross Validation in progress...")

    scores = cross_val_score(fitted_model, X, y, cv=folds, scoring=scoring)

    print (scores.mean() * -1.0)
    print (scores.std())


# df = pd.DataFrame(data=y_pred, columns=["ypred"])
# df["ytest"] = y_test.tolist()

# print(df.head())
# print(df.info())
# print(df.describe())

#
# sns.distplot(y_test)
# sns.distplot(y_pred)
# sns.plt.show()




