import matplotlib.pyplot as plt
import data
import numpy as np


def boxplot(series):
    plt.boxplot(series)
    plt.show()


def pie_chart(series):
    plt.pie(series)
    plt.show()

a = data.import_full()

negs = a[a["quantity_time_key"] < 0]

# print (a.columns)
# pie_chart(a.location_cd)

# print len(a.location_cd.unique())
# print len(a.sku.unique())

print negs.info()
print negs.describe()


