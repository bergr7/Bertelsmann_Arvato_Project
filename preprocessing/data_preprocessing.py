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


def clean_df(df, drop_subset):
    """It performs data cleaning on AZDIAS or CUSTOMERS dataframe.
    Parameters
    __________
    :param df: AZDIAS or CUSTOMERS Pandas DataFrame
    :param drop_subset: list of strings. Columns to be dropped from AZDIAS or CUSTOMERS df
    __________
    :return: Cleaned AZDIAS or CUSTOMERS Pandas DataFrame
    """
    # drop not useful cols
    df = df.drop(drop_subset, axis=1)

    # map X or XX values in CAMEO_ variables to 0's as string
    for col in ['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015']:
        df.loc[df[col].isin(['X', 'XX']), col] = str(0)

    # transform CAMEO_DEUG_2015 and CAMEO_INTL_2015 data type into integer
    df.astype({'CAMEO_DEUG_2015': 'int64', 'CAMEO_INTL_2015': 'int64'}, errors='ignore')

    df_cleaned = df

    return df_cleaned


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
            df.loc[df[attribute].isin(unknowns_list), attribute] = np.nan

    # transform columns to original dtypes
    df.astype(original_dtypes, errors='ignore')

    mapped_df = df

    return mapped_df


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
    # AZDIAS DATA PREPROCESSING FOR PCA ANALYSIS
    # load in AZDIAS data
    print("Loading AZDIAS data...")
    azdias_df = load_data('../data/Udacity_AZDIAS_052018.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=',')

    # test say 1000 rows
    # azdias_df = azdias_df[:5000]

    # Cleaning
    print("Cleaning data...")
    azdias_df = clean_df(azdias_df,
                         ['Unnamed: 0', 'ALTER_KIND1', 'ALTER_KIND3', 'ALTER_KIND4', 'EXTSEL992', 'KK_KUNDENTYP',
                          'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ']
                         )

    # impute missing values with mode
    print("Imputing missing values...")
    azdias_df = impute_mv(azdias_df, 'most_frequent')

    # encode 'CAMEO_DEU_2015'
    print("Encoding 'CAMEO_DEU_2015' variable...")
    preprocessed_azdias_df = label_encode_cameo(azdias_df)

    assert preprocessed_azdias_df.isnull().any().mean() == 0.0, "There are still missing values in the data."
    assert preprocessed_azdias_df.shape == (azdias_df.shape[0], 358)

    print("AZDIAS data is preprocessed and ready for PCA analysis")

    # save preprocessed_azdias_df as a pickle file
    save_pickle_df(preprocessed_azdias_df, "../data/", "AZDIAS_DF.pkl")
    print("AZDIAS data has been saved as a pickle file in the ../data folder with the name AZDIAS_DF.pkl.")
    print("Use pd.read_pickle to read it.")

    # CUSTOMERS DATA PREPROCESSING FOR PCA ANALYSIS
    # load in CUSTOMERS data
    print("Loading CUSTOMERS data...")
    cust_df = load_data('../data/Udacity_CUSTOMERS_052018.csv',
                        dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=','
                        )

    # test say 1000 rows
    # cust_df = cust_df[:1000]

    # Cleaning
    print("Cleaning data...")
    cust_df = clean_df(cust_df,
                       ['Unnamed: 0', 'ALTER_KIND1', 'ALTER_KIND3', 'ALTER_KIND4', 'KK_KUNDENTYP',
                        'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ']
                       )

    # impute missing values with mode
    print("Imputing missing values...")
    cust_df = impute_mv(cust_df, 'most_frequent')

    # encode 'CAMEO_DEU_2015'
    print("Encoding 'CAMEO_DEU_2015' variable...")
    cust_df = label_encode_cameo(cust_df)

    # encode 'CUSTOMER_GROUP' and 'PRODUCT_GROUP'
    print("Encoding 'CUSTOMER_GROUP' and 'PRODUCT_GROUP' variables...")
    preprocessed_cust_df = label_encode_groups(cust_df)

    assert preprocessed_cust_df.isnull().any().mean() == 0.0, "There are still missing values in the data."
    assert preprocessed_cust_df.shape == (cust_df.shape[0], 362)

    print("CUSTOMERS data is preprocessed and ready for PCA analysis.")

    # save preprocessed_cust_df as a pickle file
    save_pickle_df(preprocessed_cust_df, "../data/", "CUSTOMERS_DF.pkl")
    print("CUSTOMERS data has been saved as a pickle file in the /data folder with the name CUSTOMERS_DF.pkl.")
    print("Use pd.read_pickle to read it.")


if __name__ == '__main__':
    main()
