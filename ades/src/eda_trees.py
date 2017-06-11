import numpy as np
import matplotlib.pyplot as plt


def show_importance(forest, X):

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
    ax.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    #plt.figure()
    ax.set_title("Feature importances")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=30, fontsize='small')
    # plt.xlim([-1, X.shape[1]])
    plt.show()
