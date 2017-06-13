import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_importance(forest, X):
    """
    This graph can only be shown once the model is fitted.
    Is shows a bar chart with the importances given by the tree to the different features

    :param forest: fitted forest
    :param X: DataFrame, test data
    """

    labels = X.columns
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    l = []
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, labels[indices[f]], importances[indices[f]]))
        l.append(labels[indices[f]])

    print l
    print indices

    # Plot the feature importances of the forest
    fig, ax = plt.subplots()
    ax.bar(range(X.shape[1]), importances,
           color="r", yerr=std[indices], align="center")
    ax.set_title("Feature importances")
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(labels, rotation=30, fontsize='small')

    print("")
    print("Presenting Bar Chart...")
    plt.show()


def time_and_estimators_chart():
    """
    Shows a bar chart comparing processing time and RMSE results for the same model,
    but with different number of estimators
    """

    rmse = [0.24592021021, 0.240401483644, 0.236216140806, 0.235931985178, 0.23460944331, 0.235724524522,
            0.235229720985, 0.236398327605, 0.237369168813, 0.237706348999, 0.237742713909, 0.238003367883,
            0.238102165727, 0.238817947393, 0.239370449414]

    time = [1.02, 1.35, 1.75, 1.95, 2.17, 2.67, 2.6, 3.2, 3.39, 3.54, 3.67, 4.03, 5.1, 4.92, 5.26]

    df = pd.DataFrame()
    df["rmse"] = rmse
    df["time"] = time

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    df.rmse.plot(kind='bar', color='red', ax=ax1, width=0.4,  position=1)
    df.time.plot(kind='bar', color='blue', ax=ax2, width=0.4, position=0)

    ax1.set_xticks(range(len(rmse)))
    ax2.set_xticks(range(len(time)))

    ax1.set_xticklabels(range(1, len(rmse) + 1))

    ax1.set_ylabel("RMSE")
    ax2.set_ylabel("Time (min)")

    ax2.set_ylim([0.5, 6])
    ax1.set_ylim([0.23, np.max(rmse) + 0.003])
    ax1.set_xlim([-1, 15])

    ax1.legend(['RMSE'], loc='upper left')
    ax2.legend(['Time'], loc='upper right')

    plt.show()
