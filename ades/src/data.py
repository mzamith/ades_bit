import pandas as pd
import os
import glob
import preprocessing as pp
import pickle


LABELS = "labels"
FULL_DATA_SET = "full_dataset"
PROCESSED_DATA_SET = "processed_dataset"
PROCESSED_DATA_SET_CATEGORICAL = "processed_dataset_categorical"
PROCESSED_DATA_SET_CATEGORICAL_2 = "processed_dataset_categorical_2"


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

    file_path = get_path(package, file_name)

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
    name += ".h5"

    file_path = get_path(package, name)

    # store = pd.HDFStore(file_path)
    # store['df'] = df

    # np.savetxt(file_path, df)
    # df.to_pickle(file_path)

    df.to_hdf(file_path, 'table', append=True)


def export_to_pickle(obj, name):

    package = "models"
    name += ".pkl"

    file_path = get_path(package, name)

    with open(file_path, 'wb') as output:
        pickle.dump(obj, output, protocol=pickle.HIGHEST_PROTOCOL)


def import_feature(name):

    package = "features"
    name += ".h5"

    file_path = get_path(package, name)

    return pd.read_hdf(file_path, 'table')


def get_data(label=PROCESSED_DATA_SET_CATEGORICAL):

    df = import_feature(label)
    y = df["quantity_time_key"]
    return pp.drop_label(df, "quantity_time_key"), y


def get_labels(label=PROCESSED_DATA_SET_CATEGORICAL):

    df = import_feature(label)
    return df["quantity_time_key"]


def get_sample(nrows):

    df = import_feature(PROCESSED_DATA_SET)
    sample = df.sample(n=nrows)

    # print(sample.head())

    target = sample["quantity_time_key"]
    sample.drop("quantity_time_key", axis=1, inplace=True)
    return sample, target


# export(pp.pre_process(import_full(), categorical=False), "processed_shrinked_30")

# df = import_full()
# export(df["quantity_time_key"], "labels")

# df = import_feature(PROCESSED_DATA_SET_CATEGORICAL)

# print (df.info())
# print ("")
# print (df.describe())
