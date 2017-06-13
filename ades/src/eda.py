import matplotlib.pyplot as plt
import data
import seaborn as sns


def boxplot(series):
    plt.boxplot(series)
    plt.show()


def pie_chart(series):

    s = series.value_counts()

    plt.pie(s, labels=s.index)
    plt.show()

# a = data.import_full()
# b = data.get_data("processed_no_encoding")

# negs = a[a["quantity_time_key"] < 0]

# print (a.columns)
# pie_chart(a.location_cd)

# print len(a.location_cd.unique())
# print len(a.sku.unique())

# print b[0].info()
# print b[0].describe()

# pie_chart(b[0]["time_of_month"])

# df = pd.DataFrame(data=y_pred, columns=["ypred"])
# df["ytest"] = y_test.tolist()

# print(df.head())
# print(df.info())
# print(df.describe())

#

a = data.import_feature("categorical_new_new")

print (a.head())
print (a.info())
print (a.describe())

# sns.distplot(a)
# sns.distplot(y_pred)
# sns.plt.show()


