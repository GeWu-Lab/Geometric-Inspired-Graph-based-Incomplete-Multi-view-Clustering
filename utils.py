import os
import numpy as np

def dist_map(F):
    '''
        Obtain the matrix V, V_{ij} = ||f_i - f_j||^2_2
    '''
    N = F.shape[0]
    V = np.zeros((N, N))
    squ = np.zeros(N)
    for i in range(N):
        squ[i] = np.linalg.norm(F[i]) ** 2
    V_1 = 2 * F @ F.T
    for i in range(N):
        for j in range(N):
            V[i, j] = squ[i] + squ[j] - V_1[i, j]  
    return V       

def obtain_affinity_graph(data, m = 9):
    N = data.shape[0]
    e = dist_map(data)
    idx = np.zeros((N, m + 1))
    for i in range(N):
        idx[i] = np.argsort(e[i])[:m + 1]
    idx = idx.astype(np.int16)
    W = np.zeros((N, N))
    eps = 1e-8
    for i in range(N):
        id = idx[i, 1:m + 1]
        d = e[i, id]
        W[i, id] = (d[m - 1] - d) / (m * d[m - 1] - np.sum(d) + eps)
    return W

def simplex_opt(v):
    '''
    Column Vector :param v:
    :return:
    '''
    n = v.shape[0]
    u = v - v.mean() + 1 / n

    if np.min(u) < 0:
        f = 1
        turn = 0
        lambda_b = 0
        while (abs(f) > 1e-8):
            turn += 1
            u_1 = u - lambda_b
            p_idx = (u_1 > 0)
            q_idx = (u_1 < 0)
            f = np.sum(np.maximum(-u_1[q_idx], 0)) - n * lambda_b
            g = np.sum(q_idx) - n
            lambda_b = lambda_b - f / g
            if turn > 100:
                print("Diverge!!!!")
                break
        x = np.maximum(u_1, 0)
    else:
        x = u
    return x

def simplex_opt_1(v, k = 30):
    '''
    Sparse solution for simplex optimization problem.
    return: 
    x: The desired vector. 
    '''
    N = v.shape[0]
    idx = np.argsort(v)[::-1][:k]
    t_sum = 0
    flag = 0
    for i in range(k - 1):
        t_sum += v[idx[i]]
        t_avg = (t_sum - 1) / (i + 1)
        if t_avg >= v[idx[i + 1]]:
            flag = 1
            break
    if not flag:
        t_avg = (t_sum + v[idx[k - 1]] - 1) / k
    mask = np.zeros(v.shape[0])
    mask[idx] = 1
    x = np.maximum(v - t_avg, 0)
    x = x * mask
    return x

def get_data(dataset="MSRC_v1"):
    dir = "./data/"
    data_v = []
    for filename in os.listdir(dir):
        if filename.split(".")[0] == dataset:
            datas = np.load(dir + filename, allow_pickle=True)
            data = datas.item()["X"].squeeze()
            label = datas.item()["Y"].squeeze()
            for i in range(data.shape[0]) :
                data1 = data[i].astype(float)       # data[i] : feature * sample
                data_v.append(data1.T)
            if label.shape[0] < 50:
                label = label[0].squeeze()
            label = label.astype(np.int)
            label -= np.min(label)
            k = label.max() + 1
            print("Data Gotten")
            return data_v, label, k
    print("Not Found")

def get_laplacian(W, normalization = 1):
    '''
    W: Adjacency matrix(num_samples*num_samples)
    normalization: whether or not apply normalized method.
    output:
    L: Laplace Matrix
    '''
    S = (W + W.T) / 2
    d = np.sum(S, axis=0)
    D = np.diag(np.abs(d))
    L = D - S
    if normalization == 1:
        D_w = np.diag(np.sqrt(1/(d)))
        # D_w[D_w > 100] = 0
        L = D_w @ L @ D_w
    return L 

def calc_eigen(L, k):
    '''
    L: Laplace Matrix
    k: Clustering number
    output:
    lamb: all_lambda (sorted)
    F: Eigenvector
    '''

    # lamb, V = np.linalg.eig(L) # Unstable!

    V, lamb, _ = np.linalg.svd(L)
    idx = np.argsort(lamb)[:k]
    lamb = np.sort(lamb)
    F = V[:, idx]
    for i in range(F.shape[1]) :
        F[:, i] = F[:, i] / np.sqrt(np.sum(F[:, i] ** 2))
    return lamb, F