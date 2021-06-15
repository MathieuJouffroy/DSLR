import numpy as np

class LogisticRegression():
    """ Logistic Regression Classifier
    Parameters
    ----------
    alpha: float
      Learning rate (between 0.0 and 1.0)
    n_iters: int
      Number of iterations over the whole training dataset

    Attributes
    ----------
    _theta: Matrix of shape = [n_class, n_feature]
      θ₀,θ₁ ... weights after fitting
    """
    
    def __init__(self, alpha=0.01, n_iters=20):
        self.alpha = alpha
        self.n_iters = n_iters
    
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
        print (classes)

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
                self.theta -= (self.alpha/m) * gradient
                cost = self.cost_function(X, y_bools)
                J_history.append(cost)
                if iter % 50 == 0:
                    print (f'iter:{iter:d}\t\t\tcost:{cost:.5f}')
            class_weights.append(self.theta)
        return (np.array(class_weights), J_history)

#def fit_with_mini_batch_gd

# def plot cost hitory          

#def score

# def confusion matrix 