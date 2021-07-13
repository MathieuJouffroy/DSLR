from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prep_utils import load_csv, filter_df

def kernel_density(courses_df: pd.DataFrame, course: str):
    """Plots the kernel density plot for the specific course"""
    courses_df[course].plot.kde()
    plt.title(course)
    plt.show()

def box_plot(orig_df: pd.DataFrame, courses_df: pd.DataFrame, course: str):
    """Plots the box plot for the specific course"""
    plot = sns.boxplot(x=orig_df['Hogwarts House'], y=courses_df[course])
    plot.set_title(course)
    plt.show()

def plot_all_courses(orig_df: pd.DataFrame, courses_df: pd.DataFrame):
    """Plots the box plot and kernel density plot for each courses"""
    for course in courses_df.columns:
        box_plot(orig_df, courses_df, course)
        kernel_density(courses_df, course)

def main():
    if len(argv) == 2:
        dataframe = load_csv(argv[1])
        if dataframe is None:
            print ("Input a valid file to run the program")
            return
    else:
        print ("Input the dataset to run the program.")
        return

    courses_df = filter_df(dataframe)
    plot_all_courses(dataframe, courses_df)

if __name__ == "__main__":
    main()
