import pandas as pd


# Importing a file from CSV file
def read(file_path, sep=","):
    return pd.read_csv(file_path, sep)


# selecting the label column from the imported dataframe
def separate(df, label):
    X = df.drop(label, axis=1)
    y = df[label]
    return X, y
