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

    lambd : int, (default=0)
      Regularization term

    Attributes
    ----------
    _theta: Matrix of shape [n_class, n_feature]
     Weights 
    """
    
    def __init__(self, alpha=0.01, n_iters=50, lambd=0):
        self.alpha = alpha
        self.n_iters = n_iters
        self.lambd = lambd

    
    def set_weights(self, theta):
        self.theta = theta

    def sigmoid(self, z):
        '''Sigmoid of z where z denotes (w.T * x + b)'''
        return 1 / (1 + np.exp(-z))
    
    def hypothesis(self, X):
        # assuming X has intercept term -> (w.T * x + b)
        z = np.dot(X, self.theta)
        return self.sigmoid(z)
    
    def prediction(self, X, classes):
        preds = self.hypothesis(X)
        # preds -> array of shape (m, n), n representing nbr of class
        # you want to filter the max prob of class for each training examples
        return np.array([classes[p] for p in preds.argmax(1)])
    
    def cost_function(self, X, Y):
        m = X.shape[0]
        Y_pred = self.hypothesis(X)
        losses = Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred)
        cost = (-1/m) * np.sum(losses)
        #Note that we should not regularize the biais parameter 
        if self.lambd != 0:
            cost += (self.lambd/(2*m)) * np.sum(np.square(self.theta[1:]))
        return cost                                   
    
    def fit_with_batch_gd(self, X, Y):
        # DIMENSIONS:
        #   theta = (n+1) x 1
        #   X     = m x (n+1)
        #   y     = m x 1
        #   grad  = (n+1) x 1
        #   J     = Scalar
        m = X.shape[0]
        J_history = []
        classes = np.unique(Y)
        class_weights = []

        for c in classes:
            # binary classification one vs all
            # 1 for the class 0 for the rest
            # numpy.where(condition, x, y)
            # x, y -> Values from which to choose. 
            # x, y and condition need to be broadcastable to some shape.
            y_bools = np.where(Y == c, 1, 0)
            self.set_weights(np.zeros((X.shape[1], 1)))
            for iter in range(self.n_iters):
                Y_pred = self.hypothesis(X)
                gradient = np.dot(X.T, (Y_pred - y_bools))
                if self.lambd != 0:
                    weights = self.theta
                    weights[0] = 0
                    gradient += (self.lambd/m) * weights
                self.theta -= (self.alpha/m) * gradient
                cost = self.cost_function(X, y_bools)
                J_history.append(cost)
                if iter % 50 == 0:
                    print (f'iter:{iter:d}\t\t\tcost:{cost:.5f}')
            class_weights.append(self.theta)
        return (np.array(class_weights), J_history)


# def plot cost hitory          