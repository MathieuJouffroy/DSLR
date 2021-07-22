import numpy as np

class LogisticRegression():
    """
    Logistic Regression Classifier

    Parameters
    ----------
    alpha : float, default=0.1 (between 0.0 and 1.0)
      Learning rate : the tuning parameter for the optimization algorithm
      that determines the step size at each iteration while moving towards
      a minimum of the cost function.

    n_iters : int, default=50
      Number of iterations taken for the optimization algorithm to converge

    lambd : float, default=0 (between 0.0 and 1.0)
      Regularization term

    Attributes
    ----------
    _theta: Vector of shape [n_feature, 1]
      Weights used for our predictions
    """

    def __init__(self, alpha, n_iters, lambd):
        self.alpha = alpha
        self.n_iters = n_iters
        self.lambd = lambd

    def set_weights(self, theta):
        """ Set the weights for our model. """
        self.theta = theta

    def sigmoid(self, Z):
        '''Returns the sigmoid activation function.'''
        return 1 / (1 + np.exp(-Z))

    def hypothesis(self, X) -> np.ndarray:
        '''Returns the sigmoid of Z where Z denotes (w.T * x + b). ('''
        Z = np.dot(X, self.theta)
        return self.sigmoid(Z)

    def prediction(self, X, classes):
        """
        Multi-Class Classification.
        Apply all classifiers to an unseen sample x and predict the label k
        for which the corresponding classifier reports the highest confidence
        score ŷ = argmax fk(x). For each example we will predict the class
        that has the maximum confidence score.

        Arguments:
        X -- samples
        classess -- list of possible classes

        Returns:
            A list of class
        """
        preds = self.hypothesis(X)
        # preds.argmax(1) -> returns index of the maximum values along an axis (1=column)
        return np.array([classes[p] for p in preds.argmax(1)])

    def cost_function(self, X, Y):
        """ Returns the logistic regression cost function given a sample X. """
        m = X.shape[0]
        y_pred = self.hypothesis(X)
        losses = Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred)
        cost = (-1/m) * np.sum(losses)

        if self.lambd != 0:
            cost += (self.lambd/(2*m)) * np.sum(np.square(self.theta[1:]))

        return cost

    def fit_with_batch_gd(self, X, Y):
        """
        Vectorized gradient descent for multi class logistic regression. The
        One-vs-all strategy involves training a single classifier per class,
        with the samples of that class as positive samples and all other samples as negatives.
        We then calculate the predictions given the hypothesis. Then calculate the errors
        given our predictions and then we calculate the derivatives. Finally we update
        the weights and cache in the cost and the weights for the corresponding class.

        Arguments:
        X -- input variable
        Y -- output variable

        Returns:
        class_weights -- a matrix of dimension [n_class, n_features, 1]
        all_cost -- a list of list (containing the cost history for each class)
        """

        m = X.shape[0]
        all_cost = []
        classes = np.unique(Y)
        class_weights = []
        all_cost = []

        for c in classes:
            print (f"\n{c} vs All")
            J_history = []
            y_bools = np.where(Y == c, 1, 0)
            self.set_weights(np.zeros((X.shape[1], 1)))

            for iter in range(self.n_iters):
                y_pred = self.hypothesis(X)
                gradient = np.dot(X.T, (y_pred - y_bools))

                if self.lambd != 0:
                    weights = self.theta
                    weights[0] = 0
                    gradient += (self.lambd/m) * weights

                self.theta -= (self.alpha/m) * gradient
                cost = self.cost_function(X, y_bools)
                J_history.append(cost)

                if (iter + 1) % 50 == 0:
                    if (iter + 1) >= 100:
                        print (f'iter:{iter+1:d}\t\tcost:{cost:.5f}')
                    else:
                        print (f'iter:{iter+1:d}\t\t\tcost:{cost:.5f}')

            class_weights.append(self.theta)
            all_cost.append(J_history)
        return (np.array(class_weights), np.array(all_cost))
