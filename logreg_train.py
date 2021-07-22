from sys import argv
import numpy as np
import pandas as pd
from prep_utils import load_csv, preprocess_dataset
from log_regression import LogisticRegression
from visualization import accuracy, confusion_matrix, classification_report, show_cost_fct
from collections import OrderedDict

def main():
    if len(argv) >= 2:
        logging = False
        if len(argv) > 2 and argv[2] == '-v':
            logging = True
        raw_df = load_csv(argv[1])
        if raw_df is None:
            print ("Input a valid file to run the program")
            return
        print (raw_df.describe())
        features_df = preprocess_dataset(raw_df)
        print (features_df.info())
        y_train = np.array(raw_df.loc[:, 'Hogwarts House'])
        y_train = y_train[:, np.newaxis]
        x_train = np.c_[np.ones(features_df.shape[0]), features_df]

        model = LogisticRegression(alpha=0.01, n_iters=400, lambd=0.5)
        model.set_weights(np.zeros((x_train.shape[1], 1)))

        assert y_train.shape[1] == 1
        assert x_train.shape[0] == y_train.shape[0]
        assert x_train.shape[1] == model.theta.shape[0]
        assert model.theta.shape[1] == 1

        weights, all_cost = model.fit_with_batch_gd(x_train, y_train)
        #print (weights.shape)
        np.save('weights', weights)
        print("\nThe weights of our model are saved in weights.npy")

        if logging:
            classes = np.unique(y_train)
            print (type(classes))
            model.set_weights(weights)
            print (f"\nWe trained our model with {model.n_iters} iterations\n")

            for cost, c in zip(all_cost, classes):
                show_cost_fct(cost, c)

            y_pred = model.prediction(x_train, classes)
            score = accuracy(y_train, y_pred)
            final_cm = confusion_matrix(classes, y_train, y_pred)
            print (f"\n---------------- Confusion Matrix -----------------\n\n{final_cm}\n")
            cr = classification_report(final_cm, classes)
            print (f"\n-------------- Classification report --------------\n\n{cr}\n")
            print (f"\nOur model has an accuracy of {score}")

    else:
        print ("Input the dataset to run the program.")
        return

if __name__ == "__main__":
    main()