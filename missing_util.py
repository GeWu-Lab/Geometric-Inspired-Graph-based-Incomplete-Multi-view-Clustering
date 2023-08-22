import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from random import random


def get_mask(num_view, num_data, missing_ratio):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.
        Args:
          num_view: view number
          num_data: number of samples
          missing_ratio: Defined in section 4.2 of the paper
        Returns:
          mask num_view * num_data
    """
    one_rate = 1.0 - missing_ratio
    if one_rate <= (1 / num_view):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, num_view, size=(num_data, 1))).toarray()
        return view_preserve.T
    error = 1
    if one_rate == 1:
        matrix = np.ones((num_data, num_view))
        return matrix.T
    
    enc = OneHotEncoder() 
    view_preserve = enc.fit_transform(randint(0, num_view, size=(num_data, 1))).toarray()
    one_num = num_view * num_data * one_rate - num_data
    ratio = one_num / (num_view * num_data)
    matrix_iter = (randint(0, 100, size=(num_data, num_view)) < int(ratio * 100)).astype(np.int)
    matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
    ratio = np.sum(matrix) / (num_view * num_data)  #constructed
    error = one_rate - ratio                        # >0 more 1 in fact
    epoch = 0
    while abs(error) >= 5e-3:
        epoch += 1
        if error > 0:
            trans_rate = error / (1 - ratio)            #0->1
        else :
            trans_rate = error / ratio                  #1->0

        for i in range(num_data):
            for j in range(num_view):
                if view_preserve[i][j] != 1:
                    if trans_rate > 0 and matrix[i][j] == 0:
                       matrix[i][j] = (random() < trans_rate).astype(np.int)
                    if trans_rate < 0 and matrix[i][j] == 1:
                       matrix[i][j] = (random() < -trans_rate).astype(np.int)
        ratio = np.sum(matrix) / (num_view * num_data)
        error = one_rate - ratio
        if epoch > 20:
            break
    return matrix.T

