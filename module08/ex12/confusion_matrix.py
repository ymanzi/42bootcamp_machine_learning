import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix

def check(y: np.ndarray, y_hat: np.ndarray, categorie1, categorie2):
    count = 0
    for e_real, e_predict in zip(y, y_hat):
        if e_real == categorie1 and e_predict == categorie2:
            count += 1
    return count

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    By definition a confusion matrix  is such that  
    is equal to the number of observations known to be in group  and predicted to be in group .
    Args:
        y_true:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        labels: optional, a list of labels to index the matrix. 
                This may be used to reorder or select a subset of labels. (default=None)
    Returns: 
        The confusion matrix as a numpy ndarray.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    matrix = []
    if labels == None:
        labels = np.unique(y_hat)
    for cat1 in labels:
        tmp = []
        for cat2 in labels:
            tmp.append(check(y_true, y_hat, cat1, cat2))
        matrix.append(tmp)
    if not df_option:
        return np.array(matrix)
    return pd.DataFrame(matrix, columns=labels, index=labels)

# y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
# y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])

# print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))

# print(confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"]))