from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from prep_utils import load_csv

def pair_plot(dataset: pd.DataFrame, features: list):
    """Plot multiple pairwise bivariate distributions of the features in the dataset"""
    g = sns.pairplot(dataset[features], markers="x", height=1, hue='Hogwarts House')
    g.fig.set_size_inches(15,15)
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
    # answer = those where we can see at least 3 distinct categories on plot
    # not : 'Arithmancy', maybe not 'Care of Magical Creatures'
    relevant = ['Hogwarts House', 'Astronomy', 'Herbology', 'Divination',
                'Muggle Studies', 'Ancient Runes', 'History of Magic',
               'Transfiguration', 'Potions', 'Charms', 'Flying',
               'Defense Against the Dark Arts']

    relevant_df = dataframe[relevant]
    pair_plot(relevant_df, relevant_df.columns)
    sns.heatmap(nbr_df.corr())
    plt.show()

if __name__ == "__main__":
    main()
