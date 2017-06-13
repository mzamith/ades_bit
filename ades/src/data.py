import pandas as pd
import os
import glob
import preprocessing as pp
import pickle
from datetime import datetime


LABELS = "labels"
FULL_DATA_SET = "full_dataset"
PROCESSED_DATA_SET = "processed_dataset"
PROCESSED_DATA_SET_CATEGORICAL = "processed_dataset_categorical"
PROCESSED_DATA_SET_CATEGORICAL_2 = "processed_dataset_categorical_2"


def get_path(package, name=""):
    """
    Utility function. Gets a OS-independent path

    :param package: package name
    :param name: file name
    :return: OS-independent path string
    """

    directory = os.path.dirname(__file__)
    rel_path = os.path.join(directory, package)
    return os.path.join(rel_path, name)


# Importing a file from CSV file
def read(file_path, sep=","):
    """
    Reads a CSV file and returns a pandas data frame

    :param file_path: String with file path
    :param sep: delimiter of csv file. Comma by default
    :return: Data Frame
    """
    return pd.read_csv(file_path, sep)


# selecting the label column from the imported dataframe
def separate(df, label):
    """
    Deparates label column from feature dataframe

    :param df: Complete Data Frame
    :param label: name of label column
    :return: Data frame and pandas Series with target colum
    """
    X = df.drop(label, axis=1)
    y = df[label]
    return X, y


def import_one(year, month):
    """
    Useful for testing
    Imports only one month worth of data from the training csvs

    :param year: integer with year
    :param month: integer with month
    :return: pandas data frame
    """

    package = "resources"
    month_s = "0"+str(month) if (len(str(month)) == 1) else str(month)

    file_name = str(year) + month_s + ".csv"

    file_path = get_path(package, file_name)

    return read(file_path)


def import_full():
    """
    Imports full data set of csvs

    :return: Pandas data frame
    """

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
    """
    Exports a given data frame to the features directory, in hdf5 file format

    :param df: Data frame to export
    :param name: Name to give the file
    :return:
    """

    package = "features"
    name += ".h5"

    file_path = get_path(package, name)

    df.to_hdf(file_path, 'table', append=True)


def export_to_pickle(obj, name):
    """
    Exports a given python objet to the models folder in pickle format

    :param obj: Python object -> Any
    :param name: Name to give file
    :return:
    """

    package = "models"
    name += ".pkl"

    file_path = get_path(package, name)

    with open(file_path, 'wb') as output:
        pickle.dump(obj, output, protocol=pickle.HIGHEST_PROTOCOL)


def import_feature(name):
    """
    Imports a given file from the features folder to memory

    :param name: name of file
    :return: Data frame with train features
    """

    package = "features"
    name += ".h5"

    file_path = get_path(package, name)

    return pd.read_hdf(file_path, 'table')


def import_model(name):
    """
    Imports a saved model to memory from the models folder

    :param name: name of model
    :return: model
    """

    package = "models"
    name += ".pkl"

    file_path = get_path(package, name)

    return pd.read_pickle(file_path)


def import_test(name):
    """
    Imports the test data from the test folder

    :param name: name of file
    :return: pandas data frame
    """

    package = "test"

    file_path = get_path(package, name)

    return pd.read_csv(file_path, 'table')


def get_data(label):
    """
    Imports data.
    This function can be used only if the data set contains the labels column
    This is not the case for non-categorical data.

    :return: Features and labels
    """

    df = import_feature(label)
    y = df["quantity_time_key"]
    return pp.drop_label(df, "quantity_time_key"), y


def get_labels():
    """
    Gets the labels

    :return: pandas Series with labels
    """

    df = import_feature("labels")
    return df["quantity_time_key"]


def get_sample(nrows):
    """
    This function was useful for testing.
    Gets a sample of the complete data, given a number of instances

    :param nrows: number of instances
    :return:
    """

    df = import_feature(PROCESSED_DATA_SET)
    sample = df.sample(n=nrows)

    # print(sample.head())

    target = sample["quantity_time_key"]
    sample.drop("quantity_time_key", axis=1, inplace=True)
    return sample, target


def export_train(categorical):

    name = "features_nominal_" if categorical else "features_quant_"
    name += str(datetime.now().time())

    export(pp.pre_process(import_full(), categorical=categorical), name)

