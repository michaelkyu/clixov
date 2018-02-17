# coding: utf-8

# In[ ]:

import sys
import time
import numpy as np
import igraph

from numba import int64, float32, float64, boolean, guvectorize, jit, vectorize
import scipy.sparse
from scipy.sparse import issparse, csc_matrix, coo_matrix

import clixov_utils
from clixov_utils import *
import clique_maximal
from clique_maximal import get_cliques_igraph, BKPivotSparse2_Gnew_wrapper
from call_openblas import sgemm_openblas
import mkl_spgemm
from mkl_spgemm import dot, elt_multiply

# In[ ]:

def merge_clusters(X, GX, XTX, allG, G, dG, i_list, j_list, beta=0.5, merge_iter=0):
    threshold_loose, threshold_strict = get_thresholds_ij(X, X, XTX, i_list, j_list, beta)
    i_merge, j_merge = get_merge_mask1_ij(X, GX, X, GX, allG, i_list, j_list, threshold_loose, threshold_strict, True)
    dG = get_missing_edges(i_merge, j_merge, X, allG)
    dG = ((dG + dG.T) > 0).astype(np.int32)   
    return i_merge, j_merge, dG

def get_merge_mask1_ij(A, GA, B, GB, G, i_list, j_list, threshold_loose, threshold_strict, symm):
    """Pre-filtering approach based on N(a), followed by checking filtered (i,j) pairs"""
    
    degree = as_dense_flat(G.sum(0)).reshape(-1,1)
    A_min_degree = as_dense_flat(nonzero_min((degree * A).T))
    B_min_degree = as_dense_flat(nonzero_min((degree * B).T))
    pairs_to_test = (threshold_loose <= A_min_degree[i_list]) & (threshold_loose <= B_min_degree[j_list])    
    i_list, j_list = i_list[pairs_to_test], j_list[pairs_to_test]
    
    threshold_strict = threshold_strict[pairs_to_test]
    
    assert symm
    to_merge = test_merge(i_list, j_list, GA, A, threshold_strict)
    to_merge = to_merge.flatten()
    i_list, j_list = i_list[to_merge], j_list[to_merge]
    tmp = np.argsort(i_list * 1000000 + j_list)
    i_list, j_list = i_list[tmp], j_list[tmp]
    
    return i_list, j_list
                
def get_thresholds_ij(A, B, AB, i_list, j_list, beta):
    A_sizes, B_sizes = A.sum(0), B.sum(0)
    intersection = as_dense_flat(AB[i_list, j_list])
    union = A_sizes[i_list] + B_sizes[j_list] - intersection
    threshold_loose = beta * (union - 1)
    threshold_strict = threshold_loose + intersection
    return threshold_loose, threshold_strict
    
def get_missing_edges(i_merge, j_merge, X, G):
    dG = np.zeros(G.shape, np.bool)
    for i, j in zip(i_merge, j_merge):
        dG[np.ix_(X[:,i].nonzero()[0], X[:,j].nonzero()[0])] = True
    dG[G.nonzero()] = False
    dG[np.arange(dG.shape[0]), np.arange(dG.shape[0])] = 0
    return dG


# In[ ]:

def add_missing_edges(G, X_new):
    X_new = X_new.astype(np.bool)
    dG = np.zeros(G.shape, G.dtype)
    for i in range(X_new.shape[1]):
        tmp = X_new[:,i]
        dG[np.ix_(tmp,tmp)] = 1
        dG[tmp, tmp] = 0  # Set diagonal to 0
    dG[G==1.] = 0
    
    return dG

@guvectorize([(float64[:], float64[:])], '(n)->()', nopython=True, target='cpu')
def nonzero_min(Y, res):
    curr_min = np.int64(1e10)
    all_zeros = True
    for a in Y:
        if a!=0 and a<curr_min:
            curr_min = a
            all_zeros = False
    if all_zeros:
        curr_min = 0
    res[0] = curr_min
    
@guvectorize([(int64[:], int64[:], float32[:,:], float32[:,:], float32[:], boolean[:])],
             '(),(),(n,p),(n,p),()->()', nopython=True,
             target='parallel')
def test_merge(i, j, GX, X, threshold, to_merge):
    n_genes = GX.shape[0]
    to_merge[0] = True
    for a in range(n_genes):
        if X[a,i[0]] != X[a,j[0]]:
            if GX[a,i[0]] + GX[a,j[0]] < threshold[0]:
                to_merge[0] = False
                break
                
@guvectorize([(int64[:], int64[:], float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:], boolean[:])],
             '(),(),(n,p),(n,p),(n,q),(n,q),()->()', nopython=True,
             target='cpu')
def test_merge_AB(i, j, GA, A, GB, B, threshold, to_merge):
    n_genes = GA.shape[0]
    to_merge[0] = True
    for a in range(n_genes):
        if A[a,i[0]] != B[a,j[0]]:
            if GA[a,i[0]] + GB[a,j[0]] < threshold[0]:
                to_merge[0] = False
                break

def get_merge_mask1(A, GA, B, GB, G, threshold_loose, threshold_strict, symm, notest=None):
    """Pre-filtering approach based on N(a), followed by checking filtered (i,j) pairs"""
    
    degree = as_dense_flat(G.sum(0)).reshape(-1,1)
    A_min_degree = nonzero_min((degree * A).T)
    B_min_degree = nonzero_min((degree * B).T)
    pairs_to_test = np.logical_and(threshold_loose <= A_min_degree.reshape(-1, 1),
                                   threshold_loose <= B_min_degree.reshape(1, -1))
    
    if notest is not None:
        tmp1, tmp2 = notest
        pairs_to_test[tmp1, tmp2] = 0
        pairs_to_test[tmp2, tmp1] = 0

    if symm:
        pairs_to_test[np.tril_indices(pairs_to_test.shape[0], k=0)] = False
        print 'Test %s out of %s clique pairs:' % (pairs_to_test.sum() / 2, (A.shape[1] * (A.shape[1] - 1)) / 2)
    else:
        print 'Test %s out of %s clique pairs:' % (pairs_to_test.sum() / 2, (A.shape[1] * (B.shape[1] - 1)))
    sys.stdout.flush()
        
    i_list, j_list = pairs_to_test.nonzero()

    start = time.time()
    if symm:
        to_merge = test_merge(i_list, j_list, GA, A, threshold_strict[i_list, j_list])    
    else:
        to_merge = test_merge_AB(i_list, j_list, GA, A, GB, B, threshold_strict[i_list, j_list])
    print 'Merge time:', time.time() - start
    print 'To merge:', to_merge.sum()
    sys.stdout.flush()
    
    i_list, j_list = i_list[to_merge], j_list[to_merge]
    tmp = np.argsort(i_list * 1000000 + j_list)
    i_list, j_list = i_list[tmp], j_list[tmp]
    
    return i_list, j_list

@vectorize([float32(float32, float32)], nopython=True)
def norm_merge_mask3_helper2(a, ga):
    tmp = ga + a
    mask = a==0
    return tmp * ((not mask) - mask)
    
@jit
def norm_merge_mask3_helper1(A, GA, B, GB, symm):
    tmpA = norm_merge_mask3_helper2(A, GA)
    if symm:
        tmpB = tmpA
    else:
        tmpB = norm_merge_mask3_helper2(B, GB)
    tmpA = np.asfortranarray(tmpA.T)
    return tmpA, tmpB
    
def get_merge_mask3(A, GA, B, GB, threshold_strict, symm):
    """Do a min-plus multiplication"""
    
    tmpA = norm_merge_mask3_helper2(A, GA)
    if symm:
        tmpB = tmpA
    else:
        tmpB = norm_merge_mask3_helper2(B, GB)
    start_fortran = time.time()
    tmpA = np.asfortranarray(tmpA.T)
    
    minplus = sgemm_openblas(tmpA, tmpB, n_threads=48, minimize=True)
    to_merge = (minplus - 1) >= threshold_strict
    if symm:
        to_merge[np.tril_indices(to_merge.shape[0], k=0)] = False
        
    i_list, j_list = to_merge.nonzero()
    tmp = np.argsort(i_list * 1000000 + j_list)
    i_list, j_list = i_list[tmp], j_list[tmp]
    
    return i_list, j_list

def get_thresholds(A, B, beta):
    A_sizes, B_sizes = A.sum(0), B.sum(0)
    intersection = A.T.dot(B)
    union = (A_sizes.reshape(-1,1) + B_sizes.reshape(1,-1)) - intersection
    threshold_loose = beta * (union - 1)
    threshold_strict = threshold_loose + intersection

    return threshold_loose, threshold_strict


# def calc_and_do_merge(A, GA, G, beta, method='1'):
#     if method=='1':
#         threshold_loose, threshold_strict = get_thresholds(A, A, beta)
#         i_merge, j_merge = get_merge_mask1(A, GA, A, GA, G, threshold_loose, threshold_strict, symm=True)

#         return do_merge(A, i_merge, j_merge)
#     else:
#         raise Exception('Unsupported')

# def do_merge(X, i_merge, j_merge):
#     cliques = get_cliques_igraph(X.shape[1], zip(i_merge, j_merge), input_fmt='edgelist')
#     cliques_csc = tuples_to_csc(cliques, X.shape[1])    
#     merge = dot(X, cliques_csc)
#     if issparse(merge):
#         merge.data = (merge.data > 0).astype(X.dtype)
#         merge.eliminate_zeros()
#     else:
#         merge = (merge > 0).astype(X.dtype)
#     return merge

def calc_and_do_merge(X, A, GA, G, beta, prev_merge_i, prev_merge_j, method='1'):
    if method=='1':
        H = csr_matrix(subsumption(X).T)

        tmp_i, tmp_j = H.nonzero()        
        prev_merge_i = np.append(prev_merge_i, tmp_i)
        prev_merge_j = np.append(prev_merge_j, tmp_j)
                                 
        threshold_loose, threshold_strict = get_thresholds(A, A, beta)
        i_merge, j_merge = get_merge_mask1(A, GA, A, GA, G,
                                           threshold_loose, threshold_strict,
                                           symm=True,
                                           notest=(prev_merge_i, prev_merge_j))
        
        merge = do_merge(X, i_merge, j_merge, prev_merge_i, prev_merge_j, H=H)

        # max_size = 0
        # for i in range(merge.shape[1]):
        #     clique = tuple(sorted(merge[:,i].nonzero()[0]))
        #     max_size = max(max_size, len(clique))
        #     if len(clique)>=4:
        #         print '*', as_dense_array(G[clique,:][:,clique])
        #     assert_clique(clique, G)            
        # print 'max_size:', max_size
        
        return merge, i_merge, j_merge
    else:
        raise Exception('Unsupported')

def do_merge(X, i_merge, j_merge, prev_merge_i, prev_merge_j, H=None):
    n = X.shape[1]
    Gnew = coo_matrix((np.ones(2 * i_merge.size, np.int32),
                       (np.append(i_merge, j_merge),
                        np.append(j_merge, i_merge))),
                      (n,n)).tocsc()
    Gold = coo_matrix((np.ones(2 * prev_merge_i.size, np.int32),
                       (np.append(prev_merge_i, prev_merge_j),
                        np.append(prev_merge_j, prev_merge_i))),
                      (n,n)).tocsc()

    assert Gnew.multiply(Gold).sum() == 0


    tmp = dot(dot(H.T, Gnew) > 0, H) > 0
    tmp2 = Gnew.multiply(tmp) > 0
    print 'Transferred from Gnew to Gold:', tmp2.sum()
    Gold += tmp2
    Gnew -= tmp2
    
    cliques_csc_list = []

    # if H is None:
    #     H = csr_matrix(subsumption(X))

    last_unexplained = -1
    it = 0
    
    while Gnew.sum() > 0:        
        print 'Iteration:', it, 'Gold/Gnew sum:', Gold.sum(), Gnew.sum()
        
        # cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BK_hier_Gsep_wrapper(Gold, Gnew, H)
        cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BK_hier_Gnew_wrapper(Gold, Gnew, H)
        # print 'cliques:', [cliques[cliques_indptr[i]:cliques_indptr[i+1]].tolist() for i in range(cliques_n)]
        print 'Search tree nodes:', tree_size[0]
        cliques_csc = cliques_to_csc(cliques, cliques_indptr, cliques_n, n)

        # for i in range(cliques_csc.shape[1]):
        #     j = cliques_csc[:,i].nonzero()[0]
        #     if not (H[j,:][:,j].sum() == 0):
        #         for k in j:
        #             print '|', k, X[:,k].nonzero()[0]
        #         # import pdb
        #         # pdb.set_trace()
        #         assert H[j,:][:,j].sum() == 0
            
        tmp = csc_to_cliques_list(cliques_csc)
        assert len(tmp)==len(set(tmp))

        # cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BKPivotSparse2_Gnew_wrapper(Gold, Gnew)
        # cliques_csc = cliques_to_csc(cliques, cliques_indptr, cliques_n, n)

        print 'Merge meta cliques:', cliques_csc.shape[1], clixov_utils.format_clique_sizes(cliques_csc)

    #    print [(i, cliques_csc[:,i].nonzero()[0].tolist()) for i in range(cliques_csc.shape[1])]
        # print 'aa:', (cliques_csc[(13, 16, 2482), :].sum(0)==3).nonzero()

        cliques_csc_list.append(cliques_csc)

        cliques_csc_trans = cliques_csc + (dot(H.T, cliques_csc) > 0).astype(cliques_csc.dtype)

        # dot(H, dot(unexplained, H))

        # tmp = dot(unexplained, H.T) > 0
        # tmp2 = dot(dot(tmp, explained) > 0, H) > 0
        
#        unexplained = get_unexplained_edges(cliques_csc, Gnew)
        unexplained = get_unexplained_edges(cliques_csc_trans, Gnew)
                
        print 'Unexplained meta edges:', unexplained.sum()
#        print 'Unexplained with 55:', unexplained[:,55].nonzero()[0]
        
        if last_unexplained == unexplained.sum():
            import pdb
            pdb.set_trace()
        last_unexplained = unexplained.sum()

        Gold += Gnew - unexplained
        Gold = csc_matrix(Gold)
        Gnew = csc_matrix(unexplained)

        it += 1

    cliques_csc = csc_matrix(scipy.sparse.hstack(cliques_csc_list))

    merge = dot(X, cliques_csc)
    if issparse(merge):
        merge.data = (merge.data > 0).astype(X.dtype)
        merge.eliminate_zeros()
    else:
        merge = (merge > 0).astype(X.dtype)
    print 'Merge cliques:', merge.shape[1], clixov_utils.format_clique_sizes(merge)

    # print 'bb:', (merge[(13, 16, 2482), :].sum(0)==3).nonzero()[1]
    # for i in cliques_csc[:,11].nonzero()[0]:
    #     print 'cc:', i, X[:,i].nonzero()[0]

    tmp = csc_to_cliques_list(merge)
    assert len(tmp)==len(set(tmp))

    tmp = []
    in_X = set(csc_to_cliques_list(X))
    for i in range(merge.shape[1]):
        if tuple(sorted(merge[:,i].nonzero()[0])) not in in_X:
            tmp.append(i)
    merge = merge[:,tmp]

    print 'Merge cliques:', merge.shape[1], clixov_utils.format_clique_sizes(merge)

    X = csc_matrix(scipy.sparse.hstack([X, merge]))
    
    return merge

# def do_merge(X, i_merge, j_merge, prev_merge_i, prev_merge_j):
#     n = X.shape[1]
#     Gnew = coo_matrix((np.ones(2 * i_merge.size, np.int32),
#                        (np.append(i_merge, j_merge),
#                         np.append(j_merge, i_merge))),
#                       (n,n)).tocsc()
#     Gold = coo_matrix((np.ones(2 * prev_merge_i.size, np.int32),
#                        (np.append(prev_merge_i, prev_merge_j),
#                         np.append(prev_merge_j, prev_merge_i))),
#                       (n,n)).tocsc()

#     assert Gnew.multiply(Gold).sum() == 0

#     cliques_csc_list = []
    
#     H = subsumption(X)
#     H = csr_matrix(H.T)
    
#     while Gnew.sum() > 0:        
        
#         # cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BK_hier_Gsep_wrapper(Gold, Gnew, H)
#         cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BK_hier_Gnew_wrapper(Gold, Gnew, H)
#         print 'Search tree nodes:', tree_size[0]
#         cliques_csc = cliques_to_csc(cliques, cliques_indptr, cliques_n, n)

#         for i in range(cliques_csc.shape[1]):
#             j = cliques_csc[:,i].nonzero()[0]
#             if not (H[j,:][:,j].sum() == 0):
#                 for k in j:
#                     print '|', k, X[:,k].nonzero()[0]
#                 assert H[j,:][:,j].sum() == 0

#         tmp = csc_to_cliques_list(cliques_csc)
#         assert len(tmp)==len(set(tmp))

#         # cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BKPivotSparse2_Gnew_wrapper(Gold, Gnew)
#         # cliques_csc = cliques_to_csc(cliques, cliques_indptr, cliques_n, n)

#         print 'Merge cliques:', cliques_csc.shape[1], clixov_utils.format_clique_sizes(cliques_csc)
#     #    print [(i, cliques_csc[:,i].nonzero()[0].tolist()) for i in range(cliques_csc.shape[1])]

#         # print 'aa:', (cliques_csc[(13, 16, 2482), :].sum(0)==3).nonzero()

#         merge = dot(X, cliques_csc)
#         if issparse(merge):
#             merge.data = (merge.data > 0).astype(X.dtype)
#             merge.eliminate_zeros()
#         else:
#             merge = (merge > 0).astype(X.dtype)
#         print 'Merge cliques:', merge.shape[1], clixov_utils.format_clique_sizes(merge)

#         # print 'bb:', (merge[(13, 16, 2482), :].sum(0)==3).nonzero()[1]
#         # for i in cliques_csc[:,11].nonzero()[0]:
#         #     print 'cc:', i, X[:,i].nonzero()[0]

#         tmp = csc_to_cliques_list(merge)
#         assert len(tmp)==len(set(tmp))

#         tmp = []
#         in_X = set(csc_to_cliques_list(X))
#         for i in range(merge.shape[1]):
#             if tuple(sorted(merge[:,i].nonzero()[0])) not in in_X:
#                 tmp.append(i)
#         merge = merge[:,tmp]

#         print 'Merge cliques:', merge.shape[1], clixov_utils.format_clique_sizes(merge)

#         unexplained = get_unexplained_edges(cliques_csc, Gnew)
#         Gold += Gnew - unexplained
#         Gold = csc_matrix(Gold)
#         Gnew = csc_matrix(unexplained)
#         X = csc_matrix(scipy.sparse.hstack([X, merge]))
    
#     return merge

def get_unexplained_edges(X, G):
    """Return a gene-by-gene boolean matrix with 1 indicating that the
       gene pair is in a clique.

       X : gene-by-clique matrix
       G : gene-by-gene adjacency matrix
    """
    Y = dot(X, X.T)
    if issparse(Y):
        Y.data = (Y.data > 0).astype(X.dtype)
        diag = np.arange(Y.shape[0])
        Y[diag, diag] = 0
        Y.eliminate_zeros()
    else:
        Y = (Y > 0).astype(X.dtype)
        np.fill_diagonal(Y, 0)

    Y = G - elt_multiply(G, Y)
    return Y
