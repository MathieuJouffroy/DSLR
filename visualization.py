from numpy import dtype
import pandas as pd

def accuracy(y_true, y_pred):
    ''' Calculate the accuracy score of our predictions. '''
    accuracy = (y_pred == y_true).mean()
    return accuracy

def confusion_matrix(classes, y_true, y_pred):
    ''' 
    Generate the confusion matrix for our data as a dataframe.
    The diagonal values represents the true positive.
    The horizontal axis represents the false positives (we predicted
    a class when the the real value was not that class).
    The vertical axis represents the false negatives (we predicted
    that it was not a class when the real value was the class).
    
    Arguments:
    classes -- list of unique classes
    y_true -- true outputs (numpy array)
    y_pred -- predicted outputs (numpy array)
    
    Returns:
    cf_matrix -- the confusion matrix as a pandas dataframe
    '''
    cf_matrix = pd.DataFrame(0, columns=classes, index=classes)
    for i in range(len(y_pred)):
        for house in cf_matrix.columns:
            other = [x for x in classes if x != house]
            if y_pred[i] == house and y_true[i] == house:
                cf_matrix.loc[house][house] += 1

            for j in range(len(other)):
                if y_pred[i] == house and y_true[i] == other[j]:
                    cf_matrix.loc[other[j]][house] += 1
    return cf_matrix

def classification_report(cf_matrix, classes):
    """
    Support is the number of actual occurrences of the class in the specified dataset.
    precision = tp / (tp + fp)   
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    tp on diagonal axis
    fp on the horizontal axis
    fn on the vertical axis
    """
    class_rep = pd.DataFrame(0, index=classes, columns=['Precision', 'Recall', 'F1', 'Support'])
    class_rep = class_rep.astype({'Precision': float, 'Recall': float, 'F1': float, 'Support': int})

    for col in cf_matrix.columns:
        tp = tn = fp = fn = 0
        tp = cf_matrix.loc[col, col]
        vs_all = [x for x in classes if x != col]
        for i in range(len(vs_all)):
            tn += cf_matrix.loc[vs_all[i], vs_all[i]]
            fp += cf_matrix.loc[vs_all[i], col]
            fn += cf_matrix.loc[col, vs_all[i]]

            precision = tp / (tp + fn)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            class_rep.loc[col, 'Precision'] = precision
            class_rep.loc[col, 'Recall'] = recall
            class_rep.loc[col, 'F1'] = f1
            class_rep.loc[col, 'Support'] = tp + fp
    return class_rep
    