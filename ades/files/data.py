import pandas as pd
import os
import glob


def get_path(package, name=""):
    directory = os.path.dirname(__file__)
    rel_path = os.path.join(directory, package)
    return os.path.join(rel_path, name)


# Importing a file from CSV file
def read(file_path, sep=","):
    return pd.read_csv(file_path, sep)


# selecting the label column from the imported dataframe
def separate(df, label):
    X = df.drop(label, axis=1)
    y = df[label]
    return X, y


def import_one(year, month):

    package = "resources"
    month_s = "0"+str(month) if (len(str(month)) == 1) else str(month)

    file_name = str(year) + month_s + ".csv"

    dir = os.path.dirname(__file__)
    rel_path = os.path.join(dir, package)
    file_path = os.path.join(rel_path, file_name)

    return read(file_path)


def import_full():

    rel_path = get_path("resources")

    all_files = glob.glob(rel_path + "/*.csv")

    list_ = []
    for file_ in all_files:
        print("Importing " + file_)
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    return frame


def export(df, name):

    package = "features"
    name += ".pickle"

    file_path = get_path(package, name)

    df.to_pickle(file_path)


def import_feature(name):

    package = "features"
    name += ".pickle"

    file_path = get_path(package, name)

    return pd.read_pickle(file_path)
