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

def add_missing_edges(G, X_new):
    X_new = X_new.astype(np.bool)
    dG = np.zeros(G.shape, G.dtype)
    for i in range(X_new.shape[1]):
        tmp = X_new[:,i]
        dG[np.ix_(tmp,tmp)] = 1
        dG[tmp, tmp] = 0  # Set diagonal to 0
    dG[G==1.] = 0
    
    return dG

@guvectorize([(float64[:], float64[:])], '(n)->()',
             nopython=True,
             target='parallel')
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

def get_merge_mask1(A, GA, B, GB, G, threshold_loose, threshold_strict, symm, notest=None, onlytest=None):
    """Pre-filtering approach based on N(a), followed by checking filtered (i,j) pairs"""
    
    degree = as_dense_flat(G.sum(0)).reshape(-1,1)
    A_min_degree = nonzero_min((degree * A).T)
    A_test = threshold_loose <= A_min_degree.reshape(-1, 1)
    if symm:
        B_min_degree, B_test = A_min_degree, A_test
    else:        
        B_min_degree = nonzero_min((degree * B).T)
        B_test = threshold_loose <= B_min_degree.reshape(1, -1)
    pairs_to_test = np.logical_and(A_test, B_test)
    
    if notest is not None:
        # if issparse(notest) or isinstance(notest, np.ndarray):
        #     tmp1, tmp2 = notest.nonzero()
        #     pairs_to_test[tmp1, tmp2] = 0
        # else:
        
        tmp1, tmp2 = notest
        pairs_to_test[tmp1, tmp2] = 0
        pairs_to_test[tmp2, tmp1] = 0

    if onlytest is not None:
        pairs_to_test = (pairs_to_test & onlytest)

    i_list, j_list = pairs_to_test.nonzero()
    
    if symm:
        # pairs_to_test[np.tril_indices(pairs_to_test.shape[0], k=0)] = False
        tmp = i_list < j_list
        i_list, j_list = i_list[tmp], j_list[tmp]
#    else:
        #print 'Test %s out of %s clique pairs:' % (pairs_to_test.sum() / 2, (A.shape[1] * (B.shape[1] - 1)))
    print 'Test %s out of %s clique pairs:' % (len(i_list), (A.shape[1] * (B.shape[1] - 1)))
    sys.stdout.flush()

    start = time.time()
    if symm:
        to_merge = test_merge(i_list, j_list, GA, A, threshold_strict[i_list, j_list])    
    else:
        to_merge = test_merge_AB(i_list, j_list, GA, A, GB, B, threshold_strict[i_list, j_list])
    print 'Test merge time:', time.time() - start
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
    raise Exception('Need to update this to support sparse matrices')

    A_sizes, B_sizes = A.sum(0), B.sum(0)
    intersection = A.T.dot(B)
    union = (A_sizes.reshape(-1,1) + B_sizes.reshape(1,-1)) - intersection
    threshold_loose = beta * (union - 1)
    threshold_strict = threshold_loose + intersection

    return threshold_loose, threshold_strict

def get_thresholds_symm(A, beta):
    A_sizes = as_dense_flat(A.sum(0))
    intersection = as_dense_array(dot(A.T, A))
    union = (A_sizes.reshape(-1,1) + A_sizes.reshape(1,-1)) - intersection
    threshold_loose = beta * (union - 1)
    threshold_strict = threshold_loose + intersection

    return threshold_loose, threshold_strict

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

def calc_and_do_merge(X_sp, X, GX, G, beta, prev_merge_i, prev_merge_j, method='1', Gnew=None):
    if method=='1':
        H = subsumption(X_sp).T
        H = csr_matrix(H)

        tmp_i, tmp_j = H.nonzero()        
        prev_merge_i = np.append(prev_merge_i, tmp_i)
        prev_merge_j = np.append(prev_merge_j, tmp_j)

        if Gnew is not None:
            XGX = dot(dot(X_sp.T, Gnew), X_sp) > 0
            onlytest = XGX.toarray()
            print 'onlytest:', onlytest.sum()
        else:
            onlytest = None

        notest = (prev_merge_i, prev_merge_j)
            
        threshold_loose, threshold_strict = get_thresholds_symm(X_sp, beta)
        
        start = time.time()
        i_merge, j_merge = get_merge_mask1(X, GX, X, GX, G,
                                           threshold_loose, threshold_strict,
                                           symm=True,
                                           notest=notest,
                                           onlytest=onlytest)
        print 'get_merge_mask1 time:', time.time() - start

        start = time.time()
        merge = do_merge(X_sp, i_merge, j_merge, prev_merge_i, prev_merge_j, H=H)
        print 'do_merge time:', time.time() - start

        return merge, i_merge, j_merge
    elif method=='2':
        threshold_loose, threshold_strict = get_thresholds(A, A, beta)
        i_merge, j_merge = get_merge_mask1(A, GA, A, GA, G, threshold_loose, threshold_strict, symm=True)
        return do_merge(A, i_merge, j_merge)
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
    
    cliques_csc_list = []

    last_unexplained = -1
    it = 0

    ngenes = X.shape[0]
    orig_Gnew_genes = Gnew[:ngenes,:ngenes]
                           
    while Gnew.sum() > 0:        
        print 'Iteration:', it, 'Gold/Gnew sum:', Gold.sum(), Gnew.sum()

        ## This transfer from Gnew to Gold should only occur temporarily within each iteration
        tmp = dot(dot(H.T, Gnew) > 0, H) > 0
        tmp2 = Gnew.multiply(tmp) > 0
        print '\t', 'Transferred from Gnew to Gold:', tmp2.sum()
        start = time.time()
        C, CP, CN, tree = clique_maximal.BK_hier_Gnew_wrapper(Gold + tmp2, Gnew - tmp2, H, verbose=False)
        print '\t', 'Search tree nodes / time:', tree[0], time.time() - start
        cliques_csc = cliques_to_csc(C, CP, CN, n)
        assert_unique_cliques(cliques_csc)
        
        print '\t', 'Meta cliques:', cliques_csc.shape[1], clixov_utils.format_clique_sizes(cliques_csc)

        cliques_csc_list.append(cliques_csc)

        cliques_csc_trans = cliques_csc + (dot(H.T, cliques_csc) > 0).astype(cliques_csc.dtype)
        unexplained = get_unexplained_edges(cliques_csc_trans, Gnew)
        print '\t', 'Unexplained meta edges:', unexplained.sum()
        
        if last_unexplained == unexplained.sum():
            import cPickle
            with open('tmp.npy', 'wb') as f:
                cPickle.dump((Gold, Gnew, H), f)
            raise
        last_unexplained = unexplained.sum()

        Gold += Gnew - unexplained
        Gold = csc_matrix(Gold)
        Gnew = csc_matrix(unexplained)

        it += 1

    cliques_csc = csc_matrix(scipy.sparse.hstack(cliques_csc_list))

    start = time.time()
    merge = dot(X, cliques_csc)
    if issparse(merge):
        merge.data = (merge.data > 0).astype(X.dtype)
        merge.eliminate_zeros()
    else:
        merge = (merge > 0).astype(X.dtype)
    print 'Merge cliques:', merge.shape[1], time.time() - start, clixov_utils.format_clique_sizes(merge)

    # tmp = csc_to_cliques_list(merge)
    # assert len(tmp)==len(set(tmp))

    # tmp = []                    
    in_X = set(csc_to_cliques_list(X))
    for i in range(merge.shape[1]):
        assert tuple(sorted(merge[:,i].nonzero()[0])) not in in_X
        # if tuple(sorted(merge[:,i].nonzero()[0])) not in in_X:
        #     tmp.append(i)
    # merge = merge[:,tmp]
    # print 'Merge cliques:', merge.shape[1], clixov_utils.format_clique_sizes(merge)
        
    # Filter the cliques to the set of largest clique covers for each
    # new edge between genes (importantly, we're not covering the new
    # edges between old cliques)
    start = time.time()
    idx = get_largest_clique_covers(merge, orig_Gnew_genes)
    merge = merge[:,idx]
    print 'orig_Gnew_genes:', orig_Gnew_genes.sum(), time.time() - start
    print 'Merge cliques (cover):', merge.shape[1], clixov_utils.format_clique_sizes(merge)

    X = csc_matrix(scipy.sparse.hstack([X, merge]))
    
    return merge

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
        try:
            Y[diag, diag] = 0
        except scipy.sparse.SparseEfficiencyWarning:
            pass
        Y.eliminate_zeros()
    else:
        Y = (Y > 0).astype(X.dtype)
        np.fill_diagonal(Y, 0)

    Y = G - elt_multiply(G, Y)
    return Y
