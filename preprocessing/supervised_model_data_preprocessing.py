# Data preprocessing for Supervised Learning model

# Import relevant libraries
import sys

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def load_data(path_file, dtype, sep):
    """It loads in a .csv file and turn it into a pandas df.
    Parameters
    __________
    :param path_file: string. Path file
    :param dtype: dict. dtype for pd.read_csv pandas method
    :param sep: string. separator for pd.read_csv pandas method
    __________
    :return: .csv file converted into a Pandas DataFrame
    """
    return pd.read_csv(path_file, dtype=dtype, sep=sep)


def map_unknowns(attributes, df):
    """It maps unknown values identified during data exploration to NaN's.
    Parameters
    __________
    :param attributes: Attributes pandas DataFrame.
    :param df: MAILOUT Pandas DataFrame
    __________
    :return: mapped_df: MAILOUT Pandas DataFrame with unknown values mapped to NaN's
    """
    # create a dict with original dtypes for each column
    original_dtypes = dict()
    for col in df.columns:
        original_dtypes[col] = str(df[col].dtype)

    # convert all columns to object type
    df.astype(dtype='str')
    # loop through all attributes
    for attribute in attributes.index:
        # for each attribute, retrieve a list with unknown values
        unknowns_list = attributes['Unknown'].loc[attribute].strip('][').split(', ')[0].split(',')
        # if there are unknown values, map them to NaN's
        if unknowns_list != ['']:
            if attribute in ['CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'CAMEO_INTL_2015']:
                df.loc[df[attribute].isin(['X','XX', '-1']), attribute] = np.nan
            else:
                df.loc[df[attribute].isin(unknowns_list), attribute] = np.nan

    # transform columns to original dtypes
    df.astype(original_dtypes, errors='ignore')

    mapped_df = df

    return mapped_df


def clean_df(mailout_train_df, mailout_test_df, train=True):
    """It performs data cleaning on AZDIAS or CUSTOMERS dataframe.
    Parameters
    __________
    :param mailout_train_df: MAILOUT TRAIN Pandas DataFrame
    :param mailout_test_df: MAILOUT TEST Pandas DataFrame
    :param train: Boolean. If True, it performs cleaning on MAILOUT TRAIN df.
    If False, it performs cleaning on MAILOUT TEST.
    __________
    :return: Cleaned MAILOUT TRAIN or MAILOUT TEST Pandas DataFrame
    """
    # drop rows with missing values in the RESPONSE
    mailout_train_df = mailout_train_df.loc[mailout_train_df['RESPONSE'].isin([0, 1]), :].copy()

    # columns to be dropped due to missing values proportion
    mailout_train_toDrop = check_mv_prop(mailout_train_df, 0.6, toDrop=True)
    mailout_test_toDrop = check_mv_prop(mailout_test_df, 0.6, toDrop=True)
    mailout_train_toDrop.extend(mailout_test_toDrop)
    toDrop = list(set(mailout_train_toDrop))   # find unique values

    # check which dataframe is going to be cleaned
    if train:
        # drop cols not useful for analysis or with lots of mv
        toDrop.extend(['Unnamed: 0', 'LNR', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ'])
        mailout_train_df = mailout_train_df.drop(toDrop, axis=1)

        # drop individuals with more than 150 missed values
        rows_toDrop = list(mailout_train_df.isnull().sum(axis=1).loc[mailout_train_df.isnull().sum(axis=1) > 150].index)
        mailout_train_df = mailout_train_df.drop(rows_toDrop, axis=0)

        # remove outliers in 'ANZ_HAUSHALTE_AKTIV' and 'ANZ_PERSONEN'
        mailout_train_df = mailout_train_df.loc[mailout_train_df['ANZ_HAUSHALTE_AKTIV'] < 10, :]  # based on 1.5*IQR rule and attributes information
        mailout_train_df = mailout_train_df.loc[mailout_train_df['ANZ_PERSONEN'] < 3, :]  # based on 1.5*IQR rule and attributes information

        df_cleaned = mailout_train_df

        return df_cleaned

    else:

        # drop cols not useful for analysis or with lots of mv
        toDrop.extend(['Unnamed: 0', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ'])
        mailout_test_df = mailout_test_df.drop(toDrop, axis=1)

        df_cleaned = mailout_test_df

        return df_cleaned


def check_mv_prop(df, p, toDrop=True):
    """It checks the proportion of missing values for each col and prints which cols have more than p% missing values.
    INPUT:
    df: Pandas dataframe.
    p: float. Missing values proportion threshold.
    toDrop: Boolean. If true, condition is propotion of mv > p. condition is propotion of mv < p otherwise.

    OUTPUT:
    toDrop_lst: list of columns be dropped if toDrop = True.
    toImpute_lst: list of columns to be imputed if toImpute_lst = True.
    """
    mvs = df.isnull().sum()
    if toDrop:
        toDrop_lst = []
        for col in df.columns:
            if mvs.loc[col] / df.shape[0] > p:
                print("{:.2f}% of {} are missing values".format((df.isnull().sum().loc[col] / df.shape[0]) * 100, col))
                toDrop_lst.append(col)
        return toDrop_lst
    else:
        toImpute_lst = []
        for col in df.columns:
            if mvs.loc[col] / df.shape[0] <= p:
                print("{:.2f}% of {} are missing values".format((df.isnull().sum().loc[col] / df.shape[0]) * 100, col))
                toImpute_lst.append(col)
        return toImpute_lst


def impute_mv(df, strategy):
    """It performs imputation of missing values using skelarn SimpleImputer.
    Parameters
    __________
    :param df: MAILOUT Pandas DataFrame
    :param strategy: string. The imputation strategy for SimpleImputer
    __________
    :return: MAILOUT df with imputed values
    """
    # instantiate SimpleImputer
    imputer = SimpleImputer(strategy=strategy)

    # impute missing values
    data_with_no_mv = imputer.fit_transform(df)

    # put back column names from df as fit_transform returns an array of shape (n_samples, n_features_new)
    imputed_df = pd.DataFrame(data_with_no_mv, columns=df.columns)

    return imputed_df


def label_encode_cameo(df):
    """It performs label encoding on 'CAMEO_DEU_2015' feature using sklearn LabelEncoder.
    Parameters
    __________
    :param df: MAILOUT Pandas DataFrame
    __________
    :return: MAILOUT Pandas DataFrame with encoded 'CAMEO_DEU_2015'.
    """
    # instantiate LabelEncoder
    lab_encoder = LabelEncoder()

    # pull out list of unique classes
    classes = list(df['CAMEO_DEU_2015'].unique())

    # fit encoder
    lab_encoder.fit(classes)

    # label encode 'CAMEO_DEU_2015'
    df['CAMEO_DEU_2015'] = lab_encoder.transform(df['CAMEO_DEU_2015'])

    encoded_df = df

    return encoded_df


def save_pickle_df(df, file_path, file_name):
    """It saves preprocessed data frame as a pickle file in the /data folder.
    Parameters
    __________
    :param df: Preprocessed MAILOUT Pandas DataFrame
    :param file_path: string. File path for the pickle file
    :param  file_name: string. File name for the pickle file
    __________
    :return: None
    """
    # save df as a pickle
    df.to_pickle(file_path + file_name)


def main():
    if len(sys.argv) == 3:
        train_file_path, test_file_path = sys.argv[1:]

    # load in attributes.csv
    print("Loading attributes.csv...")
    attributes = pd.read_csv('../data/attributes.csv', sep=';', names=['Type', 'Unknown'])

    # load in MAILOUT TRAIN data
    print("Loading MAILOUT TRAIN data...")
    raw_mailout_train_df = load_data(train_file_path,
                               dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'},
                               sep=','
                               )

    # load in MAILOUT TEST data
    print("Loading MAILOUT TEST data...")
    raw_mailout_test_df = load_data(test_file_path,
                               dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'},
                               sep=','
                               )
    # Preprocess MAILOUT TRAIN dataset
    # test say 20000 rows
    # raw_mailout_train_df = raw_mailout_train_df[:20000]
    # test say 20000 rows
    # raw_mailout_test_df = raw_mailout_test_df[:20000]

    # map unknown values to missing values
    print("Mapping unknown values to NaN's...")
    mailout_train_df = map_unknowns(attributes=attributes, df=raw_mailout_train_df)

    # cleaning
    print("Cleaning AZDIAS data...")
    mailout_train_df = clean_df(mailout_train_df, raw_mailout_test_df, train=True)

    # impute missing values with mode
    print("Imputing missing values...")
    mailout_train_df = impute_mv(mailout_train_df, 'most_frequent')

    # encode 'CAMEO_DEU_2015'
    print("Encoding 'CAMEO_DEU_2015' variable...")
    preprocessed_mailout_train_df = label_encode_cameo(mailout_train_df)

    assert preprocessed_mailout_train_df.isnull().any().mean() == 0.0, "There are still missing values in the data."

    print("MAILOUT TRAIN data is preprocessed.")
    print("Saving preprocessed data into a pickle file...")

    # save preprocessed dataframe as a pickle file
    save_pickle_df(preprocessed_mailout_train_df, "../data/", "MAILOUT_TRAIN_DF.pkl")
    print("MAILOUT data has been saved as a pickle file in the ../data folder with the name MAILOUT_TRAIN_DF.pkl.")
    print("Use pd.read_pickle to read it.")

    # Preprocess MAILOUT TEST dataset
    # map unknown values to missing values
    print("Mapping unknown values to NaN's...")
    mailout_test_df = map_unknowns(attributes=attributes, df=raw_mailout_test_df)

    # Cleaning
    print("Cleaning data...")
    mailout_test_df = clean_df(raw_mailout_train_df, mailout_test_df, train=False)

    # impute missing values with mode
    print("Imputing missing values...")
    mailout_test_df = impute_mv(mailout_test_df, 'most_frequent')

    # encode 'CAMEO_DEU_2015'
    print("Encoding 'CAMEO_DEU_2015' variable...")
    preprocessed_mailout_test_df = label_encode_cameo(mailout_test_df)

    assert preprocessed_mailout_test_df.isnull().any().mean() == 0.0, "There are still missing values in the data."

    # save preprocessed dataframe as a pickle file
    save_pickle_df(preprocessed_mailout_test_df, "../data/", "MAILOUT_TEST_DF.pkl")
    print("MAILOUT data has been saved as a pickle file in the ../data folder with the name MAILOUT_TEST_DF.pkl.")
    print("Use pd.read_pickle to read it.")


if __name__ == '__main__':
    main()
