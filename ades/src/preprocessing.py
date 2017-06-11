import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
import holidays
from time import time
from sklearn.decomposition import IncrementalPCA
import metrics

LABELS = "labels"
FULL_DATA_SET = "full_dataset"
PROCESSED_DATA_SET = "processed_dataset"


"""
##################
CONVERTING DATES
##################
"""


def get_week(date_key):
    """
    Converts a date string into the corresponding week of the year(1-52)

    :param date_key: string in format yyyyMMdd
    :return: integer with number of week
    """

    date_key = str(date_key)
    year = int(date_key[0:4])
    month = int(date_key[4:6])
    day = int(date_key[6:8])

    return datetime.date(year, month, day).isocalendar()[1]


def get_week_day(date_key):
    """
    Converts a date string into the corresponding day of week (1-7)

    :param date_key: string in format yyyyMMdd
    :return: integer with day if week
    """

    week_dict = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}

    date_key = str(date_key)
    year = int(date_key[0:4])
    month = int(date_key[4:6])
    day = int(date_key[6:8])

    week_day = datetime.date(year, month, day).isocalendar()[2]

    return week_dict[week_day]


def get_time_of_month(date_key):

    date_key = str(date_key)
    day = int(date_key[6:8])

    if day < 10:
        return "Beginning"
    if day > 20:
        return "End"
    else:
        return "Middle"


def get_holiday(date_key):

    holidays_pt = holidays.Portugal()

    date_key = str(date_key)
    year = int(date_key[0:4])
    month = int(date_key[4:6])
    day = int(date_key[6:8])

    date = datetime.date(year, month, day)

    if (date in holidays_pt) | (date + datetime.timedelta(days=1) in holidays_pt):
        return 1
    else:
        return 0


def get_high_week(date_key):

    week_list = [31, 32, 49, 50, 51]
    week = get_week(date_key)

    return int(week in week_list)


def get_month(date_key):
    """
    Gets number of month from date string

    :param date_key:
    :return:
    """
    date_key = str(date_key)
    return int(date_key[4:6])


def get_day(date_key):
    """
    Gets number of tye day from date string

    :param date_key:
    :return:
    """
    date_key = str(date_key)
    return int(date_key[6:8])


def convert_date_column(df):
    """
    Converts the date column into the desired columns

    :param df: Data frame
    :return: same data frame with different formatted columns
    """

    df["week_year"] = map(lambda x: get_week(x), df["time_key"])
    df["day_week"] = map(lambda x: get_week_day(x), df["time_key"])
    # df["high_week"] = map(lambda x: get_high_week(x), df["time_key"])
    df["holiday"] = map(lambda x: get_holiday(x), df["time_key"])
    # df["time_of_month"] = map(lambda x: get_time_of_month(x), df["time_key"])
    df["day_of_month"] = map(lambda x: get_day(x), df["time_key"])

    df.drop("time_key", axis=1, inplace=True)
    return df


"""
##################
TREATING MISSING VALUES
##################
"""


def drop_nulls(df, null_ratio=1):
    """
    Drops columns if they have large number of null values
    Also drops promotion column, that has no predictive value

    :param df: Data frame
    :param null_ratio: defines the ration of null values - threshold to drop
    :return:
    """

    df.drop("promotion", axis=1, inplace=True)

    for column in df:
        if df[column].isnull().sum() >= df.shape[0] * null_ratio:
            df.drop(column, axis=1, inplace=True)

    return df


def fill_nulls(df, strategy='mean'):
    """
    Fills nulls according to some criteria

    :param df: Data frame to convert
    :param strategy: criteria. can be mean, most_frequent
    :return:
    """

    for column in df:
        if column.find("ind") >= 0:
            df[column].fillna(0, inplace=True)
            df[column] = df[column].astype(int)

    return df

"""
##################
SCALING VALUES
##################
"""


def normalize(df):
    """
    This was a big pain.
    Scaling Huge DataFrames takes up a lot of application memory

    :param df:
    :return:
    """

    scaler = StandardScaler(copy=False)

    n = df.shape[0]  # number of rows
    batch_size = 1000  # number of rows in each call to partial_fit
    index = 0  # helper-var

    while index < n:
        partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
        partial_x = df[index:index + partial_size]
        scaler.partial_fit(partial_x)
        index += partial_size
        print ("Got to... " + str(index))

    print ("Starting transform")

    index = 0
    scaled = pd.DataFrame(data=np.zeros(df.shape), columns=df.columns)
    while index < n:
        partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
        partial_x = df[index:index + partial_size]
        scaled[index:index + partial_size] = scaler.transform(partial_x)
        index += partial_size
        print ("Transformed... " + str(index))
    # scaled = scaler.transform(df)
    return scaled


"""
##########################
CONVERT NOMINAL ATTRIBUTES
##########################
"""


def convert_nominal(df):

    return pd.get_dummies(df)


"""
##########################
CONVERT NOMINAL ATTRIBUTES
##########################
"""


def set_data_types(df, categorical=True):

    df["location_cd"] = df["location_cd"].map('{:.0f}'.format)
    df["sku"] = df["sku"].map('{:.0f}'.format)
    # df["week_year"] = df["week_year"].map('{:.0f}'.format)

    return df



"""
##################
DROP TARGET LABEL
##################
"""


def drop_label(df, label):
    """
    Drops the target column

    :param df:
    :param label:
    :return:
    """

    df.drop(label, axis=1, inplace=True)
    return df


def treat_price_retail(df, strategy='mean'):

    imp = Imputer(missing_values='NaN', strategy=strategy, axis=1)
    df["price_retail"] = imp.fit_transform(df["price_retail"]).T.reshape(-1, 1)

    return df


def print_information(df):

    print("Some useful info:")
    print ("Columns: " + str(len(df.columns)))


def encode(df):

    encoder = LabelEncoder()

    for column in df:
        if df[column].dtype == "object":
            df[column] = encoder.fit_transform(df[column])

    return df


def shrink(df, n_components=30):

    n = df.shape[0]  # how many rows we have in the dataset
    chunk_size = 1000  # how many rows we feed to IPCA at a time, the divisor of n
    ipca = IncrementalPCA(n_components=n_components, batch_size=16)

    for i in range(0, n // chunk_size):
        print("On: " + str(i))
        ipca.partial_fit(df[i * chunk_size: (i + 1) * chunk_size])

    pd.to_pickle(ipca, "/Users/mzamith/Desktop/MESW/ADS/ades_bit/ades/src/ipca.pkl")
    out = np.zeros((n, n_components))

    print ("TRANSFORM")
    for i in range(0, n // chunk_size):
        print("On: " + str(i))
        out[i * chunk_size: (i + 1) * chunk_size] = ipca.transform(df[i * chunk_size: (i + 1) * chunk_size])

    return pd.DataFrame(data=out)


def pre_process(df, categorical=True):
    """
    Complete pre processing routine

    :param df:
    :param categorical
    :return:
    """

    total_time = 0
    b = time()
    print("")
    print("**********************************************")
    print("Starting preprocessing routine...")
    print("**********************************************")
    print("Handling date attributes...")
    df = convert_date_column(df)
    print_information(df)

    metrics.print_time(time() - b, "handling date attributes")
    total_time += time()

    print("**********************************************")
    print("Dropping null attributes...")
    df = drop_nulls(df)
    print_information(df)
    print("**********************************************")
    print("Estimating null attributes...")
    df = fill_nulls(df)
    print_information(df)
    print("**********************************************")
    print("Estimating null attributes on price...")
    df = treat_price_retail(df)
    print_information(df)
    print("**********************************************")
    print("Getting the data types right...")
    df = set_data_types(df)
    print_information(df)

    if not categorical:

        df = drop_label(df, "quantity_time_key")
        print("**********************************************")
        print("Converting nominal attributes...")
        df = pd.get_dummies(df)
        print_information(df)
        print("**********************************************")
        print("Shrinking...")
        df = shrink(df)
        print_information(df)
        print("**********************************************")
        print("Scaling data...")
        df = normalize(df)
        print_information(df)

    # else:
    #
    #     print("**********************************************")
    #     print("Encoding...")
    #     df = encode(df)
    #     print_information(df)
    #     print (df.head())
    #     print (df.info())
    #     print (df.describe())

    # df = drop_label(df, "quantity_time_key")

    return df

