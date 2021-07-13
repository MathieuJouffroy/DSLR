from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from prep_utils import load_csv

def pair_plot(courses: pd.DataFrame, features: list):
    """Plot multiple pairwise bivariate distributions of the features in the dataset"""
    sns.pairplot(courses[features], markers="x", height=0.66, aspect=1.7, corner=True, hue='Hogwarts House')
    plt.show()

def main():
    if len(argv) == 2:
        dataframe = load_csv(argv[1])
        if dataframe is None:
            print ("Input a valid file to run the program")
            return
    else:
        print ("Input the dataset to run the program.")
        return

    dataframe = dataframe.drop(['Index'], axis=1)
    nbr_df = dataframe.select_dtypes([np.number])
    courses_df = dataframe.drop(dataframe.columns[[1, 2, 3, 4]], axis=1)
    
    plt.rcParams.update({'font.size': 5.1})
    pair_plot(courses_df, courses_df.columns)
    sns.heatmap(nbr_df.corr())
    plt.show()

if __name__ == "__main__":
    main()
