from sys import argv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prep_utils import load_csv, filter_df

def get_grades(raw_df, courses_df, house, course) -> pd.Series:
    '''Returns the grades for a specific house given a specific course, filtering out NaN values'''

    output_df = courses_df[raw_df["Hogwarts House"] == house][course]
    filter_nan_df = output_df[~np.isnan(output_df)]
    return filter_nan_df

def plot_course_hist(raw_df, courses_df, course):
    '''Plots the histograms of each house for a specific course'''

    bins = np.linspace(min(courses_df[course]), max(courses_df[course]), 80)
    plt.figure(figsize=(10, 6))

    houses = raw_df['Hogwarts House'].unique()
    for house in houses:
        grades = get_grades(raw_df, courses_df, house, course)
        print (grades)
        plt.hist(grades, bins=bins, alpha=0.5, label=house)

    plt.legend(loc='upper right')
    plt.title(f"Histogram of {course} grades among Hogwarts houses")
    plt.show()

def plot_all_courses(raw_df, courses_df):
    '''Plots all histograms for each course'''
    for course in courses_df.columns:
        plot_course_hist(raw_df, courses_df, course)

def main():
    if len(argv) == 2:
        raw_df = load_csv(argv[1])
        if raw_df is None:
            print ("Input a valid file to run the program")
            return
    else:
        print ("Input the dataset to run the program.")
        return

    courses_df = filter_df(raw_df)
    plot_all_courses(raw_df, courses_df)
    return

if __name__ == "__main__":
    main()
