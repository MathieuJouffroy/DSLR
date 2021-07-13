from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
from prep_utils import load_csv, filter_df, normalize_df

def scatter_plot(courses_df: pd.DataFrame):
    "Scatter plot all combinations of courses"
    courses = list(courses_df.columns)
    for first in courses:
        c_lst = courses[1:]
        for second in c_lst:
            plt.scatter(courses_df[first], courses_df[second], label='Students', s=20, alpha=0.8)
            plt.xlabel(first)
            plt.ylabel(second)
            plt.legend()
            plt.title(f'Scatter plot of {first} vs {second}')
            plt.show()
        courses = courses[1:]

def	main():
    if len(argv) == 2:
        print (argv[1])
        dataframe = load_csv(argv[1])
        if dataframe is None:
            print ("Input a valid file to run the program")
            return
    else:
        print ("Input the dataset to run the program.")
        return

    courses_df = normalize_df(filter_df(dataframe))
    scatter_plot(courses_df)
    return

if __name__ == "__main__":
    main()
