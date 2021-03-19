import numpy as np

def check_positive_negative(y: np.ndarray, y_hat: np.ndarray, categorie):
    dic_pos_neg = { "true positives" : 0,
                    "false positives": 0,
                    "true negatives": 0,
                    "false negatives": 0}
    for e_real, e_predict in zip(y, y_hat):
        if e_real == e_predict and e_real == categorie:
            dic_pos_neg["true positives"] += 1
        elif e_real == e_predict and e_real != categorie:
            dic_pos_neg["true negatives"] += 1
        elif e_real != e_predict and e_real == categorie:
            dic_pos_neg["false negatives"] += 1
        elif e_real != e_predict and e_predict == categorie:
            dic_pos_neg["false positives"] += 1
    return dic_pos_neg

def accuracy_score_(y: np.ndarray, y_hat: np.ndarray):
    result = np.array([e1 == e2 for e1, e2 in zip(y, y_hat)]).astype(int)
    return np.sum(result) / result.size

def precision_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false positives"])

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false negatives"])

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return (2 * precision_score_(y, y_hat, pos_label) * recall_score_(y, y_hat, pos_label)) /\
         (precision_score_(y, y_hat, pos_label) + recall_score_(y, y_hat, pos_label))


# y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
# y =     np.array([1, 0, 0, 1, 0, 1, 0, 0])
# print(accuracy_score_(y, y_hat))
# print(precision_score_(y, y_hat))
# print(recall_score_(y, y_hat))
# print(f1_score_(y, y_hat))

y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(accuracy_score_(y, y_hat))
print(precision_score_(y, y_hat, pos_label='dog'))
print(recall_score_(y, y_hat, pos_label='dog'))
print(f1_score_(y, y_hat, pos_label='dog'))

# print(accuracy_score_(y, y_hat))
# print(precision_score_(y, y_hat, pos_label='norminet'))
# print(recall_score_(y, y_hat, pos_label='norminet'))
# print(f1_score_(y, y_hat, pos_label='norminet'))