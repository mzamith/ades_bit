import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies(df)
df_origin = df_origin.drop('origin_Asia', axis=1)

X_train, X_test, y_train, y_test  = train_test_split()


print (df_origin.head())