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


def calc_proj_matrix(W, alpha, idx_map, C):
    '''
    Obtaining the projection matrix 'proj_matrices' and the aggregation solution 'A'
    Input: 
        W : different views of affinity graph num_view * num_sample * num_sample;
        alpha : View-specific weight;
        idx_map : Record of the missing condition;
        C : The relationship matrix among views.
    Output"
        proj_matrices : the set of projection matrix; 
        A : aggregation solution derived by GIGA.
    '''
    num_view = alpha.shape[0]
    num_sample = W[0].shape[0]
    A = np.zeros((num_sample, num_sample))
    E = np.zeros((num_view, num_sample, num_sample))
    proj_matrices = {}
    for hash in idx_map.keys():
        mask = decode_missing_condition(hash, num_view)
        matrix, beta = get_projection(C, mask, alpha=alpha)
        proj_matrices[hash] = matrix
        for x, y in idx_map[hash]:
            for v in range(num_view):
                A[x, y] += beta[v] * W[v, x, y]
    return proj_matrices, A

def build_graph(data, mask, m = 9) :
    '''
    Graph construction. 
    Input : data : data of each view; mask: the missing indicater V * n.
    Output: W: Affinity graph. 
    '''
    W = []
    num_sample = data[0].shape[0]
    for v in range(num_view):
        mask_v = mask[v]
        available_data = data[v][mask_v]
        map_v = obtain_affinity_graph(available_data, m = m)
        W_v = np.zeros((num_sample, num_sample))
        W_v[np.ix_(mask_v, mask_v)] = map_v
        W.append(W_v)
    return W

def solve_quad(P, R = None):
    '''
        Solve quadratic problem. 
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
    '''
    Fill the weight in missing view with 0.
    Input alpha : Weights in available views;
          mask : Missing indicator.
    Output beta: Practical weights with 0 padded on missing view.
    '''
    beta = np.zeros(alpha.shape)
    beta[np._id(mask)] = alpha
    return beta


def get_projection(C, mask, alpha):
    '''
        First obtaining U^-1 V for each missing condition; then calculating reallocated weight \beta
        C : Relationship among different views, num_view * num_view.
        mask num_view: The detailed missing condition with bool. 1-> available; 0->missing
        alpha num_view vector: Current view-specific weight 
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

def GIGA_function(W,  mask):

    '''
    Main function for GIGA
    Input: W: List of affinity of different views;
           mask: missing indicator num_view * num_sample
    Output: ans: Our prediction. 
    '''

    num_view = len(W)
    num_clusters = label.max() - label.min() + 1
    num_sample = label.shape[0]
    alpha = np.ones(num_view) / num_view
    E = []
    for v in range(num_view):
        E_v = np.tile(mask[v].reshape(-1, 1).astype(int), (1, num_sample)) 
        E.append(E_v)
    E = np.array(E)
    W = np.array(W)
    C = get_correlation(W, E)
    idx_map = {}
    for i in range(num_sample):
        for j in range(num_sample):
            hsh = encode_missing_condition(E[:, i, j])
            if hsh == 0:
                continue
            if hsh not in idx_map.keys():
                idx_map[hsh] = []
            idx_map[hsh].append((i, j))
    for iteration in range(8):
        alpha_last = alpha.copy()
        proj_matrices, A = calc_proj_matrix(W, alpha, idx_map, C)
        S, _ = CLR(A, num_clusters, lambda_c = 1)
        alpha = update_alpha(S, W, idx_map, proj_matrices, num_sample / 20)     
        if np.sum(np.abs(alpha_last - alpha)) < eps:
            break
    _, ans = connected_components(S) 
    return ans

def encode_missing_condition(vector):
    N = vector.shape[0]
    hsh = 0
    for i in range(N):
        hsh *= 2
        hsh += vector[i]
    return int(hsh)

def decode_missing_condition(a, view_id):
    a = int(a)
    mask = np.zeros(view_id).astype(bool)
    while a > 0:
        view_id -= 1
        mask[view_id] = a % 2
        a = a // 2
    return mask

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed) 
set_seed(20)

if __name__ == "__main__":
    Dataset = "MSRC_v1"
    data, label, k = get_data(dataset = Dataset)
    num_sample = label.shape[0]
    num_view = len(data)
    for missing_ratio in range(5):   
        mask = get_mask(num_view, num_sample, missing_ratio=missing_ratio / 10).astype(bool)
        W = build_graph(data, mask, m = 14)
        ans = GIGA_function(W, mask)
        result = evaluation.clustering_result(ans, label)
        print("Clustering_result:", missing_ratio, result)    