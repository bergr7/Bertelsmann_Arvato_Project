# Data preprocessing for PCA analysis

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
    :param df: df: AZDIAS or CUSTOMERS Pandas DataFrame
    __________
    :return: mapped_df: AZDIAS or CUSTOMERS Pandas DataFrame with unknown values mapped to NaN's
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


def clean_df(azdias_df, cust_df, azdias=True):
    """It performs data cleaning on AZDIAS or CUSTOMERS dataframe.
    Parameters
    __________
    :param azdias_df: AZDIAS Pandas DataFrame
    :param cust_df: CUSTOMERS Pandas DataFrame
    :param azdias: Boolean. If True, it performs cleaning on AZDIAS df. If False, it performs cleaning on CUSTOMERS df.
    __________
    :return: Cleaned AZDIAS or CUSTOMERS Pandas DataFrame
    """
    # columns to be dropped due to missing values proportion
    azdias_toDrop = check_mv_prop(azdias_df, 0.6, toDrop=True)
    customers_toDrop = check_mv_prop(cust_df, 0.6, toDrop=True)
    azdias_toDrop.extend(customers_toDrop)
    toDrop = list(set(azdias_toDrop))   # find unique values

    # check which dataframe is going to be cleaned
    if azdias:
        # drop cols not useful for analysis or with lots of mv
        toDrop.extend(['Unnamed: 0', 'LNR', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ',
                       'KBA13_ANTG4', 'VERDICHTUNGSRAUM'])
        azdias_df = azdias_df.drop(toDrop, axis=1)

        # drop individuals with more than 150 missed values
        rows_toDrop = list(azdias_df.isnull().sum(axis=1).loc[azdias_df.isnull().sum(axis=1) > 150].index)
        azdias_df = azdias_df.drop(rows_toDrop, axis=0)

        # remove outliers in 'ANZ_HAUSHALTE_AKTIV' and 'ANZ_PERSONEN'
        azdias_df = azdias_df.loc[azdias_df['ANZ_HAUSHALTE_AKTIV'] < 10, :]  # based on 1.5*IQR rule and attributes information
        azdias_df = azdias_df.loc[azdias_df['ANZ_PERSONEN'] < 3, :]  # based on 1.5*IQR rule and attributes information

        df_cleaned = azdias_df

        return df_cleaned

    else:
        # drop {'CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP'}
        cust_df = cust_df.drop(['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP'], axis=1)

        # drop cols not useful for analysis or with lots of mv
        toDrop.extend(['Unnamed: 0', 'LNR', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ'])
        cust_df = cust_df.drop(toDrop, axis=1)

        # drop individuals with more than 150 missed values
        rows_toDrop = list(cust_df.isnull().sum(axis=1).loc[cust_df.isnull().sum(axis=1) > 150].index)
        cust_df = cust_df.drop(rows_toDrop, axis=0)

        # remove outliers in 'ANZ_HAUSHALTE_AKTIV' and 'ANZ_PERSONEN'
        cust_df = cust_df.loc[cust_df['ANZ_HAUSHALTE_AKTIV'] < 10, :]  # based on 1.5*IQR rule and attributes information
        cust_df = cust_df.loc[cust_df['ANZ_PERSONEN'] < 3, :]  # based on 1.5*IQR rule and attributes information

        df_cleaned = cust_df

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
    :param df: AZDIAS or CUSTOMERS Pandas DataFrame
    :param strategy: string. The imputation strategy for SimpleImputer
    __________
    :return: AZDIAS or CUSTOMERS df with imputed values
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
    :param df: AZDIAS or CUSTOMERS Pandas DataFrame
    __________
    :return: AZDIAS or CUSTOMERS Pandas DataFrame with encoded 'CAMEO_DEU_2015'.
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


def label_encode_groups(df):
    """It performs label encoding on 'CUSTOMER_GROUP' and 'PRODUCT_GROUP' columns in CUSTOMERS df.
    Parameters
    __________
    :param df: CUSTOMERS Pandas DataFrame
    __________
    :return: CUSTOMERS Pandas DataFrame with encoded 'CUSTOMER_GROUP' and 'PRODUCT_GROUP'.
    """
    # instantiate LabelEncoder
    lab_encoder = LabelEncoder()

    # loop through cols list and label encode them
    for col in ['CUSTOMER_GROUP', 'PRODUCT_GROUP']:
        # pull out list of unique classes
        classes = list(df[col].unique())

        # fit encoder
        lab_encoder.fit(classes)

        # label encode 'CAMEO_DEU_2015'
        df[col] = lab_encoder.transform(df[col])

    encoded_df = df

    return encoded_df


def save_pickle_df(df, file_path, file_name):
    """It saves preprocessed data frame as a pickle file in the /data folder.
    Parameters
    __________
    :param df: AZDIAS or CUSTOMERS Pandas DataFrame
    :param file_path: string. File path for the pickle file
    :param  file_name: string. File name for the pickle file
    __________
    :return: None
    """
    # save df as a pickle
    df.to_pickle(file_path + file_name)


def main():
    # load in attributes.csv
    print("Loading attributes.csv...")
    attributes = pd.read_csv('../data/attributes.csv', sep=';', names=['Type', 'Unknown'])

    # AZDIAS DATA PREPROCESSING FOR PCA ANALYSIS
    # load in AZDIAS data
    print("Loading AZDIAS data...")
    raw_azdias_df = load_data('../data/Udacity_AZDIAS_052018.csv',
                          dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=','
                          )

    # load in CUSTOMERS data
    print("Loading CUSTOMERS data...")
    raw_cust_df = load_data('../data/Udacity_CUSTOMERS_052018.csv',
                        dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=','
                        )

    # test say 20000 rows
    # raw_azdias_df = raw_azdias_df[:20000]
    # test say 20000 rows
    # raw_cust_df = raw_cust_df[:20000]

    # map unknown values to missing values
    print("Mapping unknown values to NaN's...")
    azdias_df = map_unknowns(attributes=attributes, df=raw_azdias_df)

    # cleaning
    print("Cleaning AZDIAS data...")
    azdias_df = clean_df(azdias_df, raw_cust_df, azdias=True)

    # impute missing values with mode
    print("Imputing missing values...")
    azdias_df = impute_mv(azdias_df, 'most_frequent')

    # encode 'CAMEO_DEU_2015'
    print("Encoding 'CAMEO_DEU_2015' variable...")
    preprocessed_azdias_df = label_encode_cameo(azdias_df)

    assert preprocessed_azdias_df.isnull().any().mean() == 0.0, "There are still missing values in the data."

    print("AZDIAS data is preprocessed and ready for PCA analysis")

    # save preprocessed_azdias_df as a pickle file
    save_pickle_df(preprocessed_azdias_df, "../data/", "AZDIAS_DF.pkl")
    print("AZDIAS data has been saved as a pickle file in the ../data folder with the name AZDIAS_DF.pkl.")
    print("Use pd.read_pickle to read it.")

    # CUSTOMERS DATA PREPROCESSING FOR PCA ANALYSIS

    # map unknown values to missing values
    print("Mapping unknown values to NaN's...")
    cust_df = map_unknowns(attributes=attributes, df=raw_cust_df)

    # Cleaning
    print("Cleaning data...")
    cust_df = clean_df(raw_azdias_df, cust_df, azdias=False)

    # impute missing values with mode
    print("Imputing missing values...")
    cust_df = impute_mv(cust_df, 'most_frequent')

    # encode 'CAMEO_DEU_2015'
    print("Encoding 'CAMEO_DEU_2015' variable...")
    preprocessed_cust_df = label_encode_cameo(cust_df)

    assert preprocessed_cust_df.isnull().any().mean() == 0.0, "There are still missing values in the data."

    print("CUSTOMERS data is preprocessed and ready for PCA analysis.")

    # save preprocessed_cust_df as a pickle file
    save_pickle_df(preprocessed_cust_df, "../data/", "CUSTOMERS_DF.pkl")
    print("CUSTOMERS data has been saved as a pickle file in the /data folder with the name CUSTOMERS_DF.pkl.")
    print("Use pd.read_pickle to read it.")


if __name__ == '__main__':
    main()
