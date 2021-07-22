from sys import argv
import numpy as np
import pandas as pd
import os.path
from collections import OrderedDict
from log_regression import LogisticRegression
from prep_utils import load_csv, preprocess_dataset
#from visualization import accuracy

def main():
    if len(argv) == 3:
        test_df = load_csv(argv[1])
        if test_df is None:
            return print ("Input a valid file to run the program")
        test_features = preprocess_dataset(test_df)
        x_test = np.c_[np.ones(test_features.shape[0]), test_features]

        classes = np.array(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'])

        if argv[2] == 'weights.npy':
            weights = np.load(argv[2], allow_pickle=True)
        else:
            weights = np.zeros((x_test.shape[1], 1))

        model = LogisticRegression(alpha=0.01, n_iters=300, lambd=0.5)
        model.set_weights(weights)

        y_test_pred = model.prediction(x_test, classes)

        #y_true = load_csv('')
        #y_true = np.array(y_true.drop(['Index'], axis=1))
        #print (y_test_pred)
        #print (f"\nOur model has an accuracy of {accuracy(y_true, y_test_pred):.5f} on the test set")

        y_test_pred = y_test_pred.reshape(y_test_pred.shape[0])
        houses = pd.DataFrame(OrderedDict({'Index': range(len(y_test_pred)), 'Hogwarts House': y_test_pred}))
        houses.to_csv('houses.csv', index=False)
        print (f"\nLoading are prediction on the file Houses.csv")

    else:
        print ("Input the dataset and the weights to run the program.")
        return

if __name__ == "__main__":
    main()