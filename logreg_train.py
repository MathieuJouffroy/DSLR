from sys import argv
import numpy as np
from prep_utils import load_csv, preprocess_dataset
from log_regression import LogisticRegression
from visualization import accuracy, confusion_matrix

# the programs generates a file containing the weights that will be used for the prediction

# option : 
#   -v : plot cost history, plot metrics (accuracy f1 recall precision), plot ROC curve 
# in prediction -> get the count for each hogwartz house

def main():
    if len(argv) >= 2:
        raw_df = load_csv(argv[1])
        if raw_df is None:
            print ("Input a valid file to run the program")
            return

        features_df = preprocess_dataset(raw_df)
        y_train = np.array(raw_df.loc[:, 'Hogwarts House'])

        # 2-D vector of shape (m, 1)
        y_train = y_train[:, np.newaxis]
        # add intercept term (bias) to X -> 2-D matrix of shape (m, (n + 1))
        x_train = np.c_[np.ones(features_df.shape[0]), features_df]

        
        model = LogisticRegression(n_iters=1000)
        # theta transpose
        model.set_weights(np.zeros((x_train.shape[1], 1)))

        assert y_train.shape[1] == 1
        assert x_train.shape[0] == y_train.shape[0]
        assert x_train.shape[1] == model.theta.shape[0]
        assert model.theta.shape[1] == 1

        theta, J_hist = model.fit_with_batch_gd(x_train, y_train)
        print (theta.shape)
        
        classes = np.unique(y_train)
        model.set_weights(theta)
        y_pred = model.prediction(x_train, classes)
        score = accuracy(y_train, y_pred)
        final_cm = confusion_matrix(classes, y_train, y_pred)
        print (final_cm)
        print (f"\nOur model has an accuracy of {score}")
    else:
        print ("Input the dataset to run the program.")
        return

if __name__ == "__main__":
    main() 