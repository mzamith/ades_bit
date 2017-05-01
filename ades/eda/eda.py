import matplotlib.pyplot as plt


def scatter(df, x, y):
    df.plot(kind='scatter', x=x, y=y)
    plt.show()


def histogram(df, column):
    df[column].plot(kind='hist', rot=70)
    plt.show()


def box_plot(df, column="", by=""):

    if column != "" & by != "":
        df.boxplot()
    elif column != "":
        df.boxplot(column=column)
    elif by != "":
        df.boxplot(by=by)
    else:
        df.boxplot(column=column, by=by)

    plt.show()