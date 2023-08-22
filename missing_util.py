import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from random import random


def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask view_num * data_len

    """
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve.T
    error = 1
    if one_rate == 1:
        matrix = np.ones((data_len, view_num))
        return matrix.T
    
    enc = OneHotEncoder() 
    view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
    one_num = view_num * data_len * one_rate - data_len
    ratio = one_num / (view_num * data_len)
    matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
    matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
    ratio = np.sum(matrix) / (view_num * data_len)  #constructed
    error = one_rate - ratio                        # >0 more 1 in fact
    epoch = 0
    while abs(error) >= 5e-3:
        epoch += 1
        if error > 0:
            trans_rate = error / (1 - ratio)            #0->1
        else :
            trans_rate = error / ratio                  #1->0

        for i in range(data_len):
            for j in range(view_num):
                if view_preserve[i][j] != 1:
                    if trans_rate > 0 and matrix[i][j] == 0:
                       matrix[i][j] = (random() < trans_rate).astype(np.int)
                    if trans_rate < 0 and matrix[i][j] == 1:
                       matrix[i][j] = (random() < -trans_rate).astype(np.int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = one_rate - ratio
        if epoch > 20:
            break
    return matrix.T

