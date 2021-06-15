import os
import numpy as np
import pandas as pd

def load_csv(filename: str):
    '''Loads a csv file and return a pd.DataFrame or None'''
    if os.path.exists(filename) and os.path.isfile(filename):
        dataframe = pd.read_csv(filename)
        return dataframe
    print ("Either the file is missing or not readable")
    return None

def filter_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Filters out all the columns except for the courses'''
    dataframe = dataframe.drop(['Hogwarts House'], axis=1)
    dataframe = dataframe.drop(['Index'], axis=1)
    courses_df = dataframe.select_dtypes([np.number])
    return courses_df

def normalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''Standardizes the values of each columns of the dataframe'''
    for name in raw_df.columns:
        print(f"--- {name} ---")
        print (f"min: {raw_df[name].min()}")
        print (f"max: {raw_df[name].max()}\n")
        raw_df[name] = (raw_df[name] - raw_df[name].mean()) / raw_df[name].std()
    return raw_df

def preprocess_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    # take only the courses as features except for arithmancy (not relevant)
    features_df = dataframe.iloc[:, 7:]
    features_df = features_df.drop(columns=['Care of Magical Creatures'])
    # fill missing values by the mean of all rows of the specific column
    features_df = features_df.fillna(features_df.mean())
    # normalize the values
    features_df = normalize_dataframe(features_df)
    return features_df