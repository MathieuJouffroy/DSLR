import numpy as np
import pandas as pd
import os
import sys

def load_csv(filename: str):
	'''Loads a csv file and return a pd.DataFrame or None'''
	if os.path.exists(filename) and os.path.isfile(filename):
			dataframe = pd.read_csv(filename)
			return dataframe
	else:
		print ("Either the file is missing or not readable")
		return

def filter_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
	'''Filters out all the columns except for the courses'''
	dataframe = dataframe.drop(['Hogwarts House'], axis=1)
	dataframe = dataframe.drop(['Index'], axis=1)
	df = dataframe.select_dtypes([np.number])
	return df

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	'''Standardizes the values of each columns of the dataframe'''
	for column_name in df.columns:
		print(f"--- {column_name} ---")
		print (f"min: {df[column_name].min()}")
		print (f"max: {df[column_name].max()}\n")
		df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
	return df