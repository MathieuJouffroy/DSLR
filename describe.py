from sys import argv
import pandas as pd
import numpy as np
from maths import *
from prep_utils import load_csv, filter_df

def describe(dataset):
    """ Mimics the Pandas Describe function. """
    output_df = pd.DataFrame(columns=[name for (name, data) in dataset.iteritems()],
                             index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    for (name, data) in dataset.iteritems():
        values = [x for x in data.values[~np.isnan(data.values)]]
        values.sort()
        count = count_(values)
        mean = mean_(values)
        std = std_(values)
        min = min_(values)
        perc_25 = percentile_(values, 0.25)
        median = percentile_(values, 0.50)
        perc_75 = percentile_(values, 0.75)
        max = max_(values)
        output_df[name] = [count, mean, std, min, perc_25, median, perc_75, max]
    return output_df

def main():
    if len(argv) > 1:
        df_raw = load_csv(argv[1])
        if df_raw is None:
            return print('Please input a valid path to the dataset.')

        #print (df_raw.describe())
        dataset = filter_df(df_raw, keep_idx=False)
        describe_df = describe(dataset)
        print(describe_df)
    else:
        return print ("Input the dataset to run the program.")

if __name__ == "__main__":
    main()
