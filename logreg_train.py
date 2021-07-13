from sys import argv
import numpy as np
from prep_utils import load_csv, preprocess_dataset
from log_regression import LogisticRegression
from visualization import accuracy, confusion_matrix, classification_report
import pandas as pd
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

        features_df = preprocess_dataset(raw_df)
        print (features_df.info())
        y_train = np.array(raw_df.loc[:, 'Hogwarts House'])
        y_train = y_train[:, np.newaxis]
        x_train = np.c_[np.ones(features_df.shape[0]), features_df]

        model = LogisticRegression(alpha=0.01, n_iters=300, lambd=0.5)
        model.set_weights(np.zeros((x_train.shape[1], 1)))

        assert y_train.shape[1] == 1
        assert x_train.shape[0] == y_train.shape[0]
        assert x_train.shape[1] == model.theta.shape[0]
        assert model.theta.shape[1] == 1

        weights, all_cost = model.fit_with_batch_gd(x_train, y_train)
        print (weights.shape)
        ## save model
        np.save('weights', weights)
        print("\nThe weights of our model are saved in weights.npy")

        ### --- visu :
        if logging:
            classes = np.unique(y_train)
            model.set_weights(weights)
            print (model.n_iters)
            y_pred = model.prediction(x_train, classes)
            score = accuracy(y_train, y_pred)
            final_cm = confusion_matrix(classes, y_train, y_pred)
            print (f"\n-------------- Confusion Matrix --------------\n\n{final_cm}\n")
            cr = classification_report(final_cm, classes)
            print (f"\n-------------- Classification report --------------\n\n{cr}\n")
            print (f"\nOur model has an accuracy of {score}")

            # ---- PREDICT ----
            test_df = load_csv('datasets/dataset_test.csv')
            test_features = preprocess_dataset(test_df)
            x_test = np.c_[np.ones(test_features.shape[0]), test_features]
            y_test = load_csv('datasets/truth.csv')
            y_true = np.array(y_test.drop(['Index'], axis=1))

            weights = np.load('weights.npy', allow_pickle=True)
            model.set_weights(weights)

            y_test_pred = model.prediction(x_test, classes)
            print (f"\nOur model has an accuracy of {accuracy(y_true, y_test_pred):.5f} on the test set")
            y_test_pred = y_test_pred.reshape(y_test_pred.shape[0])
            houses = pd.DataFrame(OrderedDict({'Index': range(len(y_test_pred)), 'Hogwarts House': y_test_pred}))
            houses.to_csv('houses.csv', index=False)

    else:
        print ("Input the dataset to run the program.")
        return

if __name__ == "__main__":
    main()