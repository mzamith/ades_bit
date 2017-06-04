import datetime
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

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


def get_month(date_key):
    """
    Gets number of month from date string

    :param date_key:
    :return:
    """

    return int(date_key[4:6])


def convert_date_column(df):
    """
    Converts the date column into the desired columns

    :param df: Data frame
    :return: same data frame with different formatted columns
    """

    # df["week_year"] = map(lambda x: get_week(x), df["time_key"])
    df["day_week"] = map(lambda x: get_week_day(x), df["time_key"])
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

    scaled = StandardScaler().fit_transform(df)
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
    df["price_retail"] = imp.fit_transform(df["price_retail"]).T

    return df


def print_information(df):

    print("Some useful info:")
    print ("Columns: " + str(len(df.columns)))


def pre_process(df, categorical='True'):
    """
    Complete pre processing routine

    :param df:
    :return:
    """

    print("**********************************************")
    print("Starting preprocessing routine...")
    print("**********************************************")
    print("Handling date attributes...")
    df = convert_date_column(df)
    print_information(df)
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

        print("**********************************************")
        print("Converting nominal attributes...")
        df = pd.get_dummies(df)
        print_information(df)
        # print("**********************************************")
        # print("Scaling data...")
        # df = normalize(df)
        # print_information(df)


    # df = drop_label(df, "quantity_time_key")

    return df

