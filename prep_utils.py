""" This module contains all preprocessing functions. """
import os
import numpy as np
import pandas as pd

def load_csv(filename):
    '''Loads a csv file and return a pd.DataFrame or None'''
    if os.path.exists(filename) and os.path.isfile(filename):
        dataframe = pd.read_csv(filename)
        return dataframe
    print ("Either the file is missing or not readable")
    return None

def filter_df(dataframe):
    '''Filters out all the columns except for the courses and returns a new pd.DataFrame'''
    dataframe = dataframe.drop(['Hogwarts House'], axis=1)
    dataframe = dataframe.drop(['Index'], axis=1)
    courses_df = dataframe.select_dtypes([np.number])
    return courses_df

def normalize_df(raw_df):
    '''Standardizes the values of each columns of the dataframe'''
    for name in raw_df.columns:
        raw_df[name] = (raw_df[name] - raw_df[name].mean()) / raw_df[name].std()
    return raw_df

def preprocess_dataset(dataframe):
    '''Filter out the features that are not relevant for training our model'''
    features_df = dataframe.iloc[:, 7:]
    features_df = features_df.drop(columns=['Care of Magical Creatures'])
    features_df = features_df.fillna(features_df.mean())
    features_df = normalize_df(features_df)
    return features_df
