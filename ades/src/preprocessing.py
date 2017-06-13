import warnings
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
import holidays
from time import time
from sklearn.decomposition import IncrementalPCA
import metrics

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    """
    DEPRICATED (no longer used)

    Devides month into beggining, middle and end according to the day

    :param date_key: format yyyyMMdd
    :return: string Beggining, Middle or End
    """

    date_key = str(date_key)
    day = int(date_key[6:8])

    if day < 10:
        return "Beginning"
    if day > 20:
        return "End"
    else:
        return "Middle"


def get_holiday(date_key):
    """
    Determines if the date is a holiday or de day before

    :param date_key: format yyyyMMdd
    :return: 1 if holiday or eve, 0 otherwise
    """

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
    """
    DEPRICATED, no longer used
    Returns if the week of presented date is a typical holiday week.
    First two weeks of august or first three weeks of December

    :param date_key: format yyyyMMdd
    :return: 1 if it is a high week, 0 otherwise
    """

    week_list = [31, 32, 49, 50, 51]
    week = get_week(date_key)

    return int(week in week_list)


def get_month(date_key):
    """
    Gets number of month from date string

    :param date_key: format yyyyMMdd
    :return: int with number of month : 1-12
    """
    date_key = str(date_key)
    return int(date_key[4:6])


def get_day(date_key):
    """
    Gets number of tye day from date string

    :param date_key: format yyyyMMdd
    :return: number of day 1-31
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


def drop_nulls(df):
    """
    Drops columns if they have large number of null values
    Also drops promotion column, that has no predictive value
    Also drops gross_sls_amt_eur_time_key column, which as no place in training

    :param df: Data frame
    :param null_ratio: defines the ration of null values - threshold to drop
    :return:
    """

    unwanted_list = ["promotion", "gross_sls_amt_eur_time_key", "price_promotion", "mailing_ind", "topo_in",
                     "ilha_ind", "price_grp", "card_group", "lxpy", "monofolha_ind"]

    for att in unwanted_list:
        if att in df.columns:
            df.drop(att, axis=1, inplace=True)

    # df.drop("promotion", axis=1, inplace=True)
    # df.drop("gross_sls_amt_eur_time_key", axis=1, inplace=True)
    #
    # df.drop("price_promotion", axis=1, inplace=True)
    # df.drop("mailing_ind", axis=1, inplace=True)
    # df.drop("topo_in", axis=1, inplace=True)
    # df.drop("ilha_ind", axis=1, inplace=True)
    # df.drop("price_grp", axis=1, inplace=True)
    # df.drop("card_group", axis=1, inplace=True)
    # df.drop("lxpy", axis=1, inplace=True)
    # df.drop("monofolha_ind", axis=1, inplace=True)

    # for column in df:
    #     if df[column].isnull().sum() >= df.shape[0]:
    #         df.drop(column, axis=1, inplace=True)
    #         print column

    return df


def fill_nulls(df):
    """
    Fills nulls of binary columns with zeros

    :param df: Data frame to convert
    :return:
    """

    for column in df:
        if column.find("ind") >= 0:
            df[column].fillna(0, inplace=True)
            df[column] = df[column].astype(int)

    return df


def treat_price_retail(df, strategy='mean'):
    """
    Treats price_retail column
    For the null instances that have other similar instances (same products)
    the values are replaced with the mode of those instances
    For products that do not supply price info, nulls are replaced by the mean of all
    prices.

    :param df: DataFrame
    :param strategy: mean is default
    :return:
    """

    treat_price_retail_new(df)

    imp = Imputer(missing_values='NaN', strategy=strategy, axis=1)
    df["price_retail"] = imp.fit_transform(df["price_retail"]).T.reshape(-1, 1)

    return df


def treat_price_retail_new(df):
    """
    Treats null values for instances whose product already has some information
    In this case, the nulls are replaced by the mode of the products prices

    :param df: DataFrame
    :return:
    """

    for a in df["sku"].unique():
        if df["price_retail"][df["sku"] == a].isnull().sum() != len(df["price_retail"][df["sku"] == a]):

            imp = Imputer(missing_values='NaN', strategy="most_frequent", axis=1)
            df.loc[df["sku"] == a, "price_retail"] = imp.fit_transform(df["price_retail"][df["sku"] == a]).T.reshape(-1, 1)

"""
##################
SCALING VALUES
##################
"""


def normalize(df):
    """
    This was a big pain.
    Scaling Huge DataFrames takes up a lot of application memory
    this implementation does it by chunks of data, iteratively.
    This function scales the data with min -3 and max 3

    :param df: Dataframe
    :return: scaled dataframe
    """

    scaler = StandardScaler(copy=False)

    n = df.shape[0]  # number of rows
    batch_size = 1000  # number of rows in each call to partial_fit
    index = 0  # helper-var

    print("Starting to fit Scaler...")

    while index < n:
        partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
        partial_x = df[index:index + partial_size]
        scaler.partial_fit(partial_x)
        index += partial_size
        # print ("Scaler: on batch " + str(index))

    print ("Starting transform...")

    index = 0
    scaled = pd.DataFrame(data=np.zeros(df.shape), columns=df.columns)
    while index < n:
        partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
        partial_x = df[index:index + partial_size]
        scaled[index:index + partial_size] = scaler.transform(partial_x)
        index += partial_size
        # print ("Transformer: on batch " + str(index//batch_size))
    # scaled = scaler.transform(df)

    return scaled


"""
##########################
CONVERT NOMINAL ATTRIBUTES
##########################
"""


def convert_nominal(df):
    """
    Performs categorical to dummy/indicator conversion


    :param df: Dataframe
    :return: Dataframe with only quantitative values.
    """

    return pd.get_dummies(df)


"""
##########################
Assigning data types
##########################
"""


def set_data_types(df, categorical=True):
    """
    Makes sure all nominal varibles are considered as such

    :param df:
    :param categorical:
    :return:
    """

    df["location_cd"] = df["location_cd"].map('{:.0f}'.format)
    df["sku"] = df["sku"].map('{:.0f}'.format)
    df["day_of_month"] = df["day_of_month"].map('{:.0f}'.format)
    df["week_year"] = df["week_year"].map('{:.0f}'.format)
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

"""
##################
UTILS
##################
"""


def print_information(df):
    """
    Utilitary function. prints the number of columns
    """

    print ("Some useful info:")
    print ("Columns: " + str(len(df.columns)))


def encode(df):

    encoder = LabelEncoder()

    for column in df:
        if df[column].dtype == "object":
            df[column] = encoder.fit_transform(df[column])

    return df


def shrink(df, n_components=100):
    """
    Performs Principal Component Analysis and reduces the dimentionallity of the data to
    a supplied number of components.
    This reduction is done iteratively because the entire dataframe does not fit in memory.


    :param df: DataFarame
    :param n_components: number of components for PCA
    :return: reduced DataFrame
    """

    n = df.shape[0]  # how many rows we have in the dataset
    chunk_size = 1000  # how many rows we feed to IPCA at a time, the divisor of n
    ipca = IncrementalPCA(n_components=n_components, batch_size=16)

    print("Starting PCA fitting...")

    for i in range(0, n // chunk_size):
        # print("Fitting PCA: on batch " + str(i))
        ipca.partial_fit(df[i * chunk_size: (i + 1) * chunk_size])

    # pd.to_pickle(ipca, "/Users/mzamith/Desktop/MESW/ADS/ades_bit/ades/src/ipca.pkl")
    out = np.zeros((n, n_components))

    print ("Starting PCA transformation....")
    for i in range(0, n // chunk_size):
        # print("Transforming PCA: on batch " + str(i))
        out[i * chunk_size: (i + 1) * chunk_size] = ipca.transform(df[i * chunk_size: (i + 1) * chunk_size])

    return pd.DataFrame(data=out)


def pre_process(df, categorical=True):
    """
    Complete pre processing routine

    :param df: the dataframe with the raw features
    :param categorical boolean variable, indicating if the desired model accepts nominal attributes
    :return: processed DataFrame
    """

    total_time = 0
    b = time()
    print("")
    print("**********************************************")
    print("Starting preprocessing routine...")

    if categorical:
        print("(~9 min)")
    else:
        print("(~30 min - takes a while because of PCA and scaling)")

    print("**********************************************")
    print("1) Handling date attributes... (~8min)")
    df = convert_date_column(df)
    print_information(df)

    metrics.print_time(time() - b, "handling date attributes")
    total_time += time()

    print("**********************************************")
    print("2) Dropping null and unwanted attributes...")
    df = drop_nulls(df)
    print_information(df)
    print("**********************************************")
    print("3) Filling null attributes on binomial features...")
    df = fill_nulls(df)
    print("**********************************************")
    print("4) Estimating null attributes on price...")
    df = treat_price_retail(df)
    print("**********************************************")
    print("5) Getting the data types right...")
    df = set_data_types(df)

    if not categorical:

        # If we are going to apply PCA, we have to drop the label column
        df = drop_label(df, "quantity_time_key")
        print("**********************************************")
        print("6) Converting nominal attributes... (~1 min)")
        c = time()
        df = pd.get_dummies(df)
        metrics.print_time(time() - c, "getting dummy columns")
        print_information(df)
        print("**********************************************")
        print("7) Shrinking... (~20 min)")
        d = time()
        df = shrink(df)
        metrics.print_time(time() - d, "applying PCA")
        print_information(df)
        print("**********************************************")
        print("8) Scaling data... (~ 1 min)")
        e = time()
        df = normalize(df)
        metrics.print_time(time() - e, "normalizing data")

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

    metrics.print_time(time() - b, "preprocessing")
    print("")
    print("**********************************************")
    "Preprocessing finished."
    return df

