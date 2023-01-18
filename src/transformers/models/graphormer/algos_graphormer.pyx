# Copyright (c) Microsoft Corporation and HuggingFace
# Licensed under the MIT License.

import cython

cimport numpy

from cython.parallel cimport parallel, prange

import numpy as np


# Reduce this number if matrices are too big for large graphs
UNREACHABLE_NODE_DISTANCE = 510 

def floyd_warshall(adjacency_matrix):
    """
    Applies the Floyd-Warshall algorithm to the adjacency matrix, to compute the 
    shortest paths distance between all nodes, up to UNREACHABLE_NODE_DISTANCE.
    """
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(np.int32, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[numpy.int32_t, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[numpy.int32_t, ndim=2, mode='c'] path = -1 * np.ones([n, n], dtype=np.int32)

    cdef unsigned int i, j, k
    cdef numpy.int32_t M_ij, M_ik, cost_ikkj
    cdef numpy.int32_t* M_ptr = &M[0,0]
    cdef numpy.int32_t* M_i_ptr
    cdef numpy.int32_t* M_k_ptr

    # set unreachable nodes distance to UNREACHABLE_NODE_DISTANCE
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = UNREACHABLE_NODE_DISTANCE

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to UNREACHABLE_NODE_DISTANCE
    for i in range(n):
        for j in range(n):
            if M[i][j] >= UNREACHABLE_NODE_DISTANCE:
                path[i][j] = UNREACHABLE_NODE_DISTANCE
                M[i][j] = UNREACHABLE_NODE_DISTANCE

    return M, path


def get_all_edges(path, i, j):
    """
    Recursive function to compute all possible paths between two nodes from the graph adjacency matrix.
    """
    cdef int k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):
    """
    Generates the full edge feature and adjacency matrix.
    Shape: num_nodes * num_nodes * max_distance_between_nodes * num_edge_features
    Dim 1 is the input node, dim 2 the output node of the edge, dim 3 the depth of the edge, dim 4 the feature
    """
    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[numpy.int32_t, ndim=4, mode='c'] edge_fea_all = -1 * np.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=np.int32)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == UNREACHABLE_NODE_DISTANCE:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all
