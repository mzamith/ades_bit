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

    rmse = [0.24592021021, 0.240401483644, 0.236216140806, 0.235931985178, 0.23460944331, 0.236009200197,
            0.234972141018, 0.235101277955, 0.235786199878,0.237082467487, 0.237237053019, 0.237237053019,
            0.237713944329, 0.237807217547, 0.238687937936]

    time = [1.04, 1.58, 2.03, 2.03, 2.64, 2.53, 3.12, 3.19, 3.37, 3.47, 3.57, 3.83, 3.85, 4.22, 4.9]

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
