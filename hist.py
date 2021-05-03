from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import load_csv, filter_dataframe, normalize_dataframe

def get_grades(df, df_norm, house_name, course_name) -> np.array:
	'''Returns the grades for a specific course and a specific house'''
	output_df = df_norm[df["Hogwarts House"] == house_name][course_name]
	return output_df[~np.isnan(output_df)]

def plot_course_hist(df, df_norm, course):
	'''Plots the histograms of each house for a specific course'''
	bins = np.linspace(min(df_norm[course]), max(df_norm[course]), 80)
	plt.figure(figsize=(10, 6))
	houses = df['Hogwarts House'].unique()
	for house in houses:
		plt.hist(get_grades(df, df_norm, house, course),bins=bins, alpha=0.5, label=house)
	plt.legend(loc='upper right')
	plt.title(f"Histogram of {course} grades among Hogwarts houses")
	plt.show()

def plot_all_courses(df, df_norm):
	'''Plots the histograms of each house for each course'''
	for course in df_norm.columns:
		plot_course_hist(df, df_norm, course)

def main():
	if len(argv) == 2:
		dataframe = load_csv(argv[1])
		if dataframe is None:
			print ("Input a valid file to run the program")
			return
	else:
		print ("Input the dataset to run the program.")
		return

	courses_df = filter_dataframe(dataframe)
	norm_df = normalize_dataframe(courses_df)
	plot_all_courses(dataframe, norm_df)
	return

if __name__ == "__main__":
	main()
