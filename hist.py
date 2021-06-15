from sys import argv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prep_utils import load_csv, filter_dataframe, normalize_dataframe

def get_grades(raw_df: pd.DataFrame, df_norm: pd.DataFrame, house:str, course:str) -> pd.Series:
    '''Returns the grades for a specific house given a specific course, filtering out NaN values'''
    output_df = df_norm[raw_df["Hogwarts House"] == house][course]
    filter_nan_df = output_df[~np.isnan(output_df)]
    return filter_nan_df

def plot_course_hist(raw_df: pd.DataFrame, df_norm: pd.DataFrame, course:str):
    '''Plots the histograms of each house for a specific course'''
    bins = np.linspace(min(df_norm[course]), max(df_norm[course]), 80)
    plt.figure(figsize=(10, 6))
    houses = raw_df['Hogwarts House'].unique()
    for house in houses:
        grades = get_grades(raw_df, df_norm, house, course)
        plt.hist(grades, bins=bins, alpha=0.5, label=house)
    plt.legend(loc='upper right')
    plt.title(f"Histogram of {course} grades among Hogwarts houses")
    plt.show()

def plot_all_courses(raw_df: pd.DataFrame, df_norm: pd.DataFrame):
    '''Plots all histograms for each course'''
    for course in df_norm.columns:
        plot_course_hist(raw_df, df_norm, course)

def main():
    if len(argv) == 2:
        raw_df = load_csv(argv[1])
        if raw_df is None:
            print ("Input a valid file to run the program")
            return
    else:
        print ("Input the dataset to run the program.")
        return

    courses_df = filter_dataframe(raw_df)
    norm_df = normalize_dataframe(courses_df)
    plot_all_courses(raw_df, norm_df)
    return

if __name__ == "__main__":
    main()
