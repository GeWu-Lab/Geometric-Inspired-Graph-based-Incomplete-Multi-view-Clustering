import numpy as np
import random
import scipy
from qpsolvers import solve_qp
import sys
sys.path.append("./")
from utils import get_data, obtain_affinity_graph
import evaluation
from missing_util import get_mask
import math
from scipy.sparse.csgraph import connected_components 
from CLR import CLR
eps = 1e-7


def calc_proj_matrix(W, alpha, N, idx_map, C):
    V = alpha.shape[0]
    A = np.zeros((N, N))
    E = np.zeros((V, N, N))
    proj_matrices = {}
    for hash in idx_map.keys():
        mask = decode_hash(hash, V)
        matrix, beta = get_projection(C, mask, alpha=alpha)
        proj_matrices[hash] = matrix
        for x, y in idx_map[hash]:
            for v in range(V):
                A[x, y] += beta[v] * W[v, x, y]
    return proj_matrices, A

def build_graph(data, mask, m = 9) :
    W = []
    num_sample = data[0].shape[0]
    for v in range(V):
        mask_v = mask[v]
        available_data = data[v][mask_v]
        map_v = obtain_affinity_graph(available_data, m = m)
        W_v = np.zeros((num_sample, num_sample))
        W_v[np.ix_(mask_v, mask_v)] = map_v
        W.append(W_v)
    return W

def solve_quad(P, R = None):
    '''
        Solve the quadratic problem. 
    '''
    V = P.shape[0]
    P = scipy.sparse.csr_matrix(P) 
    if R is None:
        q = np.zeros(V)
    else :
        q = R.squeeze()
    G = -np.eye(V)
    h = np.zeros(V)
    A = scipy.sparse.csr_matrix(np.ones((1, V)))
    b = np.ones(1)
    weight = solve_qp(P, q, G, h, A, b, solver="osqp")
    return weight


def update_alpha(S, W, idx_map, proj_matrices, lmda = 10):
    V = W.shape[0]
    N = S.shape[0] 
    alpha = np.zeros(V)
    Q = np.zeros((V, V))
    R = np.zeros((1, V))
    for hash in idx_map.keys():
        P = proj_matrices[hash]
        for i in range(len(idx_map[hash])): 
            x, y = idx_map[hash][i]      
            vec_p = (P.T @ W[:, x, y]).reshape(-1,1)
            Q += vec_p @ vec_p.T
            R += (W[:, x, y].reshape(1, -1) @ P * S[x, y])
    alpha = solve_quad(Q + lmda * np.eye(V), -2 * R)
    return alpha

def get_correlation(A, E):
    '''
    Get pair-wise relationship between different views. 
    '''
    V = A.shape[0]
    N = A.shape[1]
    A_flat = A.reshape(V, -1)  # Flatten the 3D array to a 2D array
    E_flat = E.reshape(V, -1)
    C = A_flat @ A_flat.T    # Matrix multiplication to compute correlation
    C_below = E_flat @ A_flat.T**2  # Matrix multiplication to compute denominator
    C_nor = (C / (C_below + eps)).T
    print(C_nor)
    return C_nor

def expension(alpha, mask):
    beta = np.zeros(alpha.shape)
    beta[np._id(mask)] = alpha
    return beta


def get_projection(C, mask, alpha):
    '''
        First obtaining U^-1 V for each missing condition; then calculating reallocated weight \beta
        C : Relationship among different views.
        mask l-d: The detailed missing condition with bool. 1-> available; 0->missing
        alpha l-d vector: Current view-specific weight 
    '''
    U = C[mask][:, mask].copy()
    V = C[mask].copy()
    num_view = mask.shape[0]
    H = np.zeros((num_view, mask.sum()))
    id = 0
    for v in range(num_view):
        if mask[v] == 1:
            H[v, id] = 1
            id += 1
    matrix = H @ np.linalg.inv(U) @ V 
    beta = matrix @ alpha
    weight = np.sum(beta)
    return matrix / weight, beta / weight

def GIGA_function(W, label, mask, missing_rate):
    mask = mask.astype(int)
    num_view = len(W)
    num_clusters = label.max() - label.min() + 1
    num_sample = label.shape[0]
    alpha = np.ones(num_view) / num_view
    E = []
    for v in range(num_view):
        E_v = np.tile(mask[v].reshape(-1, 1), (1, num_sample)) 
        print(E_v.shape, mask[v].shape)
        E.append(E_v)
    E = np.array(E)
    W = np.array(W)
    C = get_correlation(W, E)
    idx_map = {}
    for i in range(num_sample):
        for j in range(num_sample):
            hsh = get_hash(E[:, i, j])
            if hsh == 0:
                continue
            if hsh not in idx_map.keys():
                idx_map[hsh] = []
            idx_map[hsh].append((i, j))
    clustering_result = []
    for iteration in range(8):
        alpha_last = alpha.copy()
        proj_matrices, A = calc_proj_matrix(W, alpha, num_sample, idx_map, C)
        S, _ = CLR(A, num_clusters, lambda_c = 1)
        alpha = update_alpha(S, W, idx_map, proj_matrices, num_sample / 20)     
        if np.sum(np.abs(alpha_last - alpha)) < eps:
            break
    _, ans = connected_components(S) 
    print("Clustering_result:", missing_rate, evaluation.clustering_result(ans, label))    

    return clustering_result

def get_hash(vector):
    N = vector.shape[0]
    hsh = 0
    for i in range(N):
        hsh *= 2
        hsh += vector[i]
    return int(hsh)

def decode_hash(a, N):
    a = int(a)
    mask = np.zeros(N).astype(bool)
    while a > 0:
        N -= 1
        mask[N] = a % 2
        a = a // 2
    return mask

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed) 
set_seed(20)

if __name__ == "__main__":
    Dataset = "MSRC_v1"
    data, label, k = get_data(dataset = Dataset)
    N = label.shape[0]
    V = len(data)
    results = {}
    for missing_rate in range(5):   
        mask = get_mask(V, N, missing_rate=missing_rate / 10).astype(bool)
        W = build_graph(data, mask, m = 14)
        clustering_result = GIGA_function(W, label, mask, missing_rate)