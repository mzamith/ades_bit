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

print (a.columns)
# pie_chart(a.location_cd)
print ("codigo loja")
print len(a.location_cd.unique())

print ("codigo produto")
print len(a.sku.unique())

