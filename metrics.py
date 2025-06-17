from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    assert y.size>0
    # TODO: Write here
    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    
    # print(y.size)
    # print("Y:")
    # print(y)
    # print("Y_hat:")
    # print(y_hat)
    k =  1e-4
    accuracy_score = (correct_predictions+k )/ (total_predictions+k)

    return accuracy_score


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    # print(cls)
    # print(y_hat.unique())
    assert y_hat.size == y.size
    # assert cls in y.unique()
    # assert cls in y_hat.unique()
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()

    precision_score = tp / (tp + fp) if (tp + fp) != 0 else 0

    return precision_score


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    cls = cls.item() if isinstance(cls, np.generic) else cls

    true_positives = sum((y_hat == cls) & (y == cls))
    false_negatives = sum((y_hat != cls) & (y == cls))

    recall_score = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    return recall_score


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    assert y.size>0
    squared_errors = (y_hat - y) ** 2
    mean_squared_error = squared_errors.mean()
    rmse_score = np.sqrt(mean_squared_error)

    return rmse_score


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    assert y.size>0
    absolute_errors = abs(y - y_hat)

    mean_absolute_error = absolute_errors.mean()

    return mean_absolute_error
