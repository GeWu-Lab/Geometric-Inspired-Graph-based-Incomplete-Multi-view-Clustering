import numpy as np
from scipy.sparse.csgraph import connected_components
from utils import simplex_opt, simplex_opt_1, get_data, calc_eigen, get_laplacian


def squ_l2_dist(x, y):
    return np.sum((x - y) ** 2)


def CLR(W, k, lambda_c=1, iter_num=40, S = None):
    eps = 1e-8
    n = W.shape[0]
    W = (W + W.T)/2
    if S is None:
        S = W.copy()
    # S = (S + S.T) /2
    L = get_laplacian(S, normalization=0)
    lamb, F = calc_eigen(L, k)
    epoch = 0
    while epoch < iter_num:
        # print("EPOCH,", epoch, lambda_c)
        epoch += 1
        v = np.zeros((n, n))
        # tic = time.time()s
        squ = np.zeros(n)
        for i in range(n):
            squ[i] = np.linalg.norm(F[i]) ** 2
        v_1 = 2 * F @ F.T
        for i in range(n):
            for j in range(n):
                v[i, j] = squ[i] + squ[j] - v_1[i, j]
            if n < 100:
                S[i] = simplex_opt(W[i] - lambda_c / 2 * v[i])
            else :
                S[i] = simplex_opt_1(W[i] - lambda_c / 2 * v[i], 40)

        F_old = F.copy()
        L = get_laplacian(S, normalization=0)
        lamb, F = calc_eigen(L, k)

        if np.sum(lamb[0:k]) > eps:
            lambda_c = 2 * lambda_c
        else :
            if np.sum(lamb[0:k + 1]) < eps:
                lambda_c = lambda_c / 2
                F = F_old.copy()
            else :
                break  

    _, G = connected_components(S)
    if _ != k :
        print("Wrong Clustering", _)
    return S, G

