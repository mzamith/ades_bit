import pandas as pd


def print_head(df):
    print(df.head())


def print_tail(df):
    print(df.tail())


def print_columns(df):
    print(df.columns)


def print_info(df):
    print(df.info())


def frequency(df, column, dropna=False, out=True):
    freq = df[column].value_counts(dropna=dropna)
    if out:
        print(freq)
    return freq


def print_describe(df):
    print(df.describe())

