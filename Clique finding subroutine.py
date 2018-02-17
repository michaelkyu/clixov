
# coding: utf-8

# In[2]:

import sys
sys.path = sorted(sys.path, key=lambda x: 'anaconda' in x, reverse=True)
sys.path = ['/cellar/users/mikeyu/GI'] + sys.path
import cPickle, time, scipy, scipy.sparse, argparse, os, pickle, datetime, gzip, subprocess, StringIO, random, sys, tempfile, shutil, igraph, multiprocessing, glob
from itertools import combinations, chain, groupby, compress, permutations, product
import itertools
import numpy as np
from code.utilities import *
from collections import Counter
import pandas as pd
# from code.sketches.run_clixo import run_clixo

from ctypes import pointer, Structure, POINTER,c_void_p,c_int,c_char,c_double,byref,cdll, c_long, c_float, c_int64
import ctypes
mklso = cdll.LoadLibrary("libmkl_rt.so")

from code.Ontology import Ontology, get_smallest_ancestor

np.set_printoptions(precision=3, linewidth=200)

from IPython.core.debugger import Tracer

orig_stdout = sys.stdout
orig_stderr = sys.stderr

from numba import int64, float32, float64, boolean, guvectorize, jit, vectorize

get_ipython().magic(u'load_ext line_profiler')


# In[3]:

def print_time():
    print 'Time:', time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
    sys.stdout.flush()


# In[4]:

def print_time():
    return

def print_time2():
    print 'Time:', time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
    sys.stdout.flush()


# In[6]:

def get_column_hashes_sparse(X):
    hash_list = [np.concatenate([X.indices[i:j], X.data[i:j]]) for i, j in zip(X.indptr[:-1], X.indptr[1:])]
    for a in hash_list:
        a.flags.writeable = False
    hash_list = np.array([hash(a.data) for a in hash_list])
    return hash_list

def get_column_hashes_dense(X):
    assert X.flags['F_CONTIGUOUS']
    X.flags.writeable = False
#     import xxhash
#     hash_list = np.array([xxhash.xxh32(X[:,i].data) for i in range(X.shape[1])])
    hash_list = np.array([hash(X[:,i].data) for i in range(X.shape[1])])
    X.flags.writeable = True
    return hash_list

def get_column_hashes(X):
    start = time.time()
    
    if scipy.sparse.issparse(X):
        hash_list = get_column_hashes_sparse(X)
    else:
        hash_list = get_column_hashes_dense(X)
    
    global hash_time
    hash_time += time.time() - start
    return hash_list

def get_unique_cols(X, X_hashes, Y=None, Y_hashes=None, only_idx=False):
    ## Calculate unique columns in X
    
    if Y_hashes is None:
        _, idx = np.unique(X_hashes, return_index=True)
        idx = np.sort(idx)
        if only_idx:
            return idx
        else:
            return X[:, idx], X_hashes[idx], idx
    else:
        _, idx = np.unique(np.concatenate([X_hashes, Y_hashes]), return_index=True)
        idx = np.sort(idx)
        assert np.all(idx[:X_hashes.size] == np.arange(X_hashes.size))
        idx = idx[X_hashes.size : ] - X_hashes.size
        if only_idx:            
            return idx
        else:
            return Y[:, idx], Y_hashes[idx], idx
    # Need to verify uniqueness when there are collisions


# In[7]:

@vectorize([float32(float32, float32)], nopython=True, target='parallel')
def numba_max(x, y):
    return max(x,y)

@vectorize([float32(float32, float32)], nopython=True, target='parallel')
def numba_min(x, y):
    return min(x,y)

@vectorize([boolean(float32, float32)], nopython=True, target='parallel')
def numba_eq(x, y):
    return x == y


# In[210]:

from ctypes import pointer, Structure, POINTER,c_void_p,c_int,c_char,c_double,byref,cdll, c_long, c_float, c_int64
import ctypes
mklso = cdll.LoadLibrary("libmkl_rt.so")

##########################
# Multiply CSR times dense matrix with mkl.csrmm
##########################

#from numba import int64, float32, float64, boolean, guvectorize, jit, vectorize
# from numba import jit
# @jit(cache=True)
def csrmm(A, B):

    assert A.shape[1] == B.shape[0]
    (m, k), n = A.shape, B.shape[1]
    
    trans_pointer   = byref(c_char('N'))
    matdescra = np.array(['G', 'L', 'N', 'C'])
    matdescra_pointer = matdescra.ctypes.data_as(POINTER(c_char))
    m_pointer           = byref(c_int(m))     # Number of rows of matrix A
    n_pointer           = byref(c_int(n))     # Number of columns of matrix A
    k_pointer           = byref(c_int(k))     # Number of columns of matrix B

    C = np.empty((m, n), dtype=np.float32, order='C')
    c_pointer = C.ctypes.data_as(POINTER(c_float))
    
    val = A.data
    val = val.ctypes.data_as(POINTER(c_float))
    
    indx = A.indices
    indx = indx.ctypes.data_as(POINTER(c_int))
    
    pntrb = A.indptr[:-1]
    pntrb = pntrb.ctypes.data_as(POINTER(c_int))
    
    pntre = A.indptr[1:]
    pntre = pntre.ctypes.data_as(POINTER(c_int))
    
    b_pointer = B.ctypes.data_as(POINTER(c_float))
    
    start = time.time()
    ret = mklso.mkl_scsrmm(trans_pointer, 
                         m_pointer, n_pointer, k_pointer,
                         byref(c_float(1.0)),
                         matdescra_pointer,
                         val, indx, pntrb, pntre,
                         b_pointer, n_pointer,
                         byref(c_float(0.0)),
                         c_pointer, n_pointer)
#     print 'Time:', time.time() - start
    return C

##########################
# Multiply CSC times dense matrix with mkl.csrmm
##########################

def cscmm(A, B):

    assert A.shape[1] == B.shape[0]
    (m, k), n = A.shape, B.shape[1]
    
    trans_pointer   = byref(c_char('N'))
    matdescra = np.array(['G', 'L', 'N', 'C'])
    matdescra_pointer = matdescra.ctypes.data_as(POINTER(c_char))
    m_pointer           = byref(c_int(m))     # Number of rows of matrix A
    n_pointer           = byref(c_int(n))     # Number of columns of matrix A
    k_pointer           = byref(c_int(k))     # Number of columns of matrix B

    C = np.empty((m, n), dtype=np.float32, order='C')
    c_pointer = C.ctypes.data_as(POINTER(c_float))
    
    val = A.data
    val = val[:]
    val = val.ctypes.data_as(POINTER(c_float))
    
    indx = A.indices
    indx = indx[:]
    indx = indx.ctypes.data_as(POINTER(c_int))
    
    pntrb = A.indptr[:-1]
    pntrb = pntrb[:]
    pntrb = pntrb.ctypes.data_as(POINTER(c_int))
    
    pntre = A.indptr[1:]
    pntre = pntre[:]
    pntre = pntre.ctypes.data_as(POINTER(c_int))
    
    b_pointer = B.ctypes.data_as(POINTER(c_float))
    
#     pntrb, pntre = pntre, pntrb
    
    start = time.time()
    ret = mklso.mkl_scscmm(trans_pointer, 
                         m_pointer, n_pointer, k_pointer,
                         byref(c_float(1.0)),
                         matdescra_pointer,
                         val, indx, pntrb, pntre,
                         b_pointer, n_pointer,
                         byref(c_float(0.0)),
                         c_pointer, n_pointer)
#     print 'Time:', time.time() - start
    return C

def general_dot(X, Y):
    if scipy.sparse.issparse(X):        
        assert Y.flags['C_CONTIGUOUS']
        if scipy.sparse.isspmatrix_csr(X):
            return csrmm(X, Y)
        elif scipy.sparse.isspmatrix_csc(X):
            return cscmm(X, Y)
        else:
            raise Exception('Not supported')
    else:
        assert not scipy.sparse.issparse(Y)
#         return np.dot(X, Y)
        return X.dot(Y)


# In[206]:

# @jit
def remove_subsets_helper(X, Y, symm):
    start = time.time()
    intersection = np.dot(X.T, Y)
    global dot_time
    dot_time += time.time() - start
    
    X_colsums = X.sum(0).reshape(-1,1)
#     min_size = numba_min(X_colsums, Y.sum(0).reshape(1,-1))
    min_size = np.minimum(X_colsums, Y.sum(0).reshape(1,-1))
    row_is_min = min_size == X_colsums
    is_redundant = intersection == min_size
    if symm:
        np.fill_diagonal(is_redundant, False)
    rows_to_remove = np.logical_and(is_redundant, row_is_min).any(1)
    if symm:
        cols_to_remove = None
    else:
        cols_to_remove = np.logical_and(is_redundant, ~ row_is_min).any(0)
    
    global remove_subsets_time
    remove_subsets_time += time.time() - start
    return rows_to_remove, cols_to_remove

def remove_subsets(X, X_hashes, Y, Y_hashes, symm):
    assert X.shape[1] == X_hashes.size and Y.shape[1] == Y_hashes.size
    
    if symm:
        rows_to_remove, cols_to_remove = remove_subsets_helper(X, Y, symm)
        return X[:,~rows_to_remove], X_hashes[~rows_to_remove]
    else:
        # Make sure that there is no cliques in Y that are equal to cliques in X, or that will mess up the remove_subsets_helper function
        Y, Y_hashes, _ = get_unique_cols(X, X_hashes, Y, Y_hashes)
    
        rows_to_remove, cols_to_remove = remove_subsets_helper(X, Y, symm)
        return np.hstack([X[:,~rows_to_remove], Y[:,~cols_to_remove]]), np.concatenate([X_hashes[~rows_to_remove], Y_hashes[~cols_to_remove]])

def combiner(Y_list):
    if len(Y_list)==1:
        return Y_list[0]

    combined_Y_list = []
    for a, b in split_indices_chunk(len(Y_list), 2):
        if b==a+2:
            tmp = remove_subsets(Y_list[a][0], Y_list[a][1], Y_list[a+1][0], Y_list[a+1][1], False)
            combined_Y_list.append(tmp)
    if b==a+1:
        tmp = remove_subsets(combined_Y_list[-1][0], combined_Y_list[-1][1], Y_list[a][0], Y_list[a][1], False)
        combined_Y_list[-1] = tmp

    if len(combined_Y_list) >= 2:
        return combiner(combined_Y_list)
    else:
        len(combined_Y_list) == 1
        return combined_Y_list[0]

def maximal_sets(Y, Y_hashes, step=1000, verbose=False):
    if Y.shape[1]==0:
        return Y, Y_hashes

    start = time.time()
    
    # Setup the base case invariant where each Y bin is a set of only maximal cliques
    chunk_list = split_indices_chunk(Y.shape[1], step)
    combined_Y_list = []
    for a, b in chunk_list:
        dY, dY_hashes = Y[:, a:b], Y_hashes[a:b]
        dY, dY_hashes = remove_subsets(dY, dY_hashes, dY, dY_hashes, True)
        combined_Y_list.append((dY, dY_hashes))

    combined_Y_list = combiner(combined_Y_list)
    
    global maximal_time
    maximal_time += time.time() - start
    return combined_Y_list


# In[10]:

@jit(nopython=True, cache=True)
def numba_sparse_colany_mask(indices, indptr, rows):
    a = np.zeros(indptr.size - 1, dtype=boolean)
    for i in range(indptr.size - 1):
        curr_row = rows[0]
        count = 1
        for j in range(indptr[i], indptr[i+1]):
            while indices[j] > curr_row:
                curr_row = rows[count]
                count += 1
            if indices[j] == curr_row:
                a[i] = True
                break
    return a

@jit(nopython=True, cache=True)
def numba_sparse_colany(indices, indptr, rows):
    a = np.zeros(indptr.size - 1, dtype=boolean)
    for i in range(indptr.size - 1):
        curr_row = rows[0]
        count = 1
        for j in range(indptr[i], indptr[i+1]):
            while indices[j] > curr_row:
                curr_row = rows[count]
                count += 1
            if indices[j] == curr_row:
                a[i] = True
                break
    return a

# # print x.toarray(), tmp
# # print x.indices, x.indptr
# %time tmp2 = numba_sparse_colany(x.indices, x.indptr, tmp)
# %time tmp3 = colany_sp(x[tmp, :])
# # print tmp2, tmp3
# np.all(tmp2 == tmp3)
# # tmp2


# In[191]:

x = scipy.sparse.csc_matrix((np.random.random((1000,1000)) < 0.001).astype(np.float32))
y = np.random.random((1000,1000))
# %time np.array(x + x)
# %time y[x.nonzero()] = 1
# (x + x) == np.random.random(1000).reshape(-1,1)
x[(1,2), (1,2)] += 1


# In[224]:

def colany_sp(a):
    if not isinstance(a, scipy.sparse.csc_matrix):
        assert scipy.sparse.issparse(a)
        a = scipy.sparse.csc_matrix(a)
    return (a.indptr[1:] - a.indptr[:-1]) > 0
        
def colany_dense(a):
    return np.any(a, axis=0)

def general_hstack(X_list):
    tmp = [a for a in X_list if a.shape[1] > 0]
    if scipy.sparse.issparse(X_list[0]):        
        if len(tmp) > 0:
            return scipy.sparse.hstack(tmp)
        else:
            return scipy.sparse.csc_matrix((X_list[0].shape[0], 0), dtype=X_list[0].dtype)
    else:
        if len(tmp) > 0:
            return np.hstack(tmp)
        else:
            return np.zeros((X_list[0].shape[0], 0), dtype=X_list[0].dtype)

def make_mask(sub_mask, orig_mask):
    mask = np.zeros(orig_mask.size, dtype=np.bool)
    mask[orig_mask.nonzero()[0][sub_mask]] = True
    return mask

def grow_X(X, X_Corder, GX, X_hashes, X_mask, G, dG, i, N_i_new, N_i_old, mode='dense_mode'):
    
    ## Figure out sparse/dense parameters
    if mode=='sparse_mode':
        as_rowmaj = scipy.sparse.csr_matrix
        as_colmaj = scipy.sparse.csc_matrix
        colany = colany_sp
        assert isinstance(X_Corder, scipy.sparse.csr_matrix)
    elif mode=='dense_mode':
        as_rowmaj = np.ascontiguousarray
        as_colmaj = np.asfortranarray
        colany = colany_dense
        assert not X_Corder.flags['F_CONTIGUOUS']
        
    
    start = time.time()
    
    i_mask = np.zeros(G.shape[0], dtype=np.bool)
    i_mask[i] = True
    Vo = (~ (N_i_new | N_i_old | i_mask))

    X_sizes = np.array(X.sum(0)).flatten()
    
    if mode=='sparse_mode':
        X_has_i_float = X_Corder[i,:].toarray().flatten()
        X_has_i = X_mask & (X_has_i_float > 0)
#         assert np.all(X_has_i == make_mask((X_Corder[i, X_mask]==1.0).toarray().flatten(), X_mask))

        GX_i_old = GX[i, :].toarray().flatten()
        X_has_old = X_mask & (GX_i_old > 0)
#         assert np.all(X_has_old == make_mask(colany(X_Corder[N_i_old,:][:,X_mask]), X_mask))

        # Decide which matrix multiplication would be faster based on N_i_new vs V_o
        if N_i_new.sum() < Vo.sum():
            GX_i_new = general_dot(X.T, N_i_new.reshape(-1,1).astype(np.float32, 'C')).flatten()
            X_has_new = X_mask & (GX_i_new > 0)        
#             assert np.all(X_has_new == make_mask(colany(X_Corder[N_i_new,:][:,X_mask]), X_mask))

            GX_Vo = X_sizes - GX_i_new - GX_i_old - X_has_i_float
            X_has_Vo = X_mask & (GX_Vo > 0)
#             assert np.all(X_has_Vo == make_mask(colany(X_Corder[Vo,:][:,X_mask]), X_mask))
        else:
            GX_Vo = general_dot(X.T, Vo.reshape(-1,1).astype(np.float32, 'C')).flatten()
            X_has_Vo = X_mask & (GX_Vo > 0)        
#             assert np.all(X_has_Vo == make_mask(colany(X_Corder[Vo,:][:,X_mask]), X_mask))

            GX_i_new = X_sizes - GX_Vo - GX_i_old - X_has_i_float
            X_has_new = X_mask & (GX_i_new > 0)
#             assert np.all(X_has_new == make_mask(colany(X_Corder[N_i_new,:][:,X_mask]), X_mask))
    
#         X_has_Vo = make_mask(colany(X_Corder[Vo,:][:,X_mask]), X_mask)
#         X_has_new = make_mask(colany(X_Corder[N_i_new,:][:,X_mask]), X_mask)
#         X_has_old = make_mask(colany(X_Corder[N_i_old,:][:,X_mask]), X_mask)  
#         X_has_i = make_mask((X_Corder[i, X_mask]==1.0).toarray().flatten(), X_mask)
    
    elif mode=='dense_mode':
        m = np.ascontiguousarray(np.vstack([Vo, N_i_new, N_i_old]).T)
        tmp = numba_colany_rowcol_mask_Forder(X, X_mask.nonzero()[0], m)
        X_has_Vo = make_mask(tmp[:,0], X_mask)
        X_has_new = make_mask(tmp[:,1], X_mask)
        X_has_old = make_mask(tmp[:,2], X_mask)

    #     X_has_Vo = make_mask(X_Corder[np.ix_(Vo, X_mask)].any(0), X_mask)
    #     X_has_new = make_mask(X_Corder[np.ix_(N_i_new, X_mask)].any(0), X_mask)
    #     X_has_old = make_mask(X_Corder[np.ix_(N_i_old, X_mask)].any(0), X_mask)
        X_has_i = make_mask(X_Corder[i, X_mask]==1.0, X_mask)
    
    global mask_time, count_q
    count_q += 1
    mask_time += time.time() - start
    
    start = time.time()
    
    X1_mask = X_mask & (~X_has_Vo) & X_has_new & (~X_has_old) & (~X_has_i)
    X2_mask = X_mask & (X_has_Vo) & X_has_new & (~X_has_old) & (~X_has_i)
    X3_mask = X_mask & (~X_has_Vo) & X_has_new & (X_has_old) & (~X_has_i)
    X4_mask = X_mask & (X_has_Vo) & X_has_new & (X_has_old) & (~X_has_i)        
    X5_mask = X_mask & (~X_has_Vo) & (~X_has_new) & (X_has_old) & (X_has_i)
    
    # For the cliques that already contain i, update GX to include i
    i_delta = G[:,i].reshape(-1,1)
    tmp = np.ix_(N_i_new.nonzero()[0], X_has_i.nonzero()[0])
#     print tmp
#     GX[tmp] = GX[tmp] + 1.0
    GX[:, X_has_i] += N_i_new.reshape(-1,1)  # Might be faster if you first convert N_i_new to sparse
#     GX[:, X_has_i] += scipy.sparse.csc_matrix(N_i_new.reshape(-1,1).astype(np.float32))  # Might be faster if you first convert N_i_new to sparse
    if mode=='sparse_mode':
        GX[i, X_has_new] += GX_i_new[X_has_new]
#         assert np.all(GX_i_new[X_has_new] == np.array(X[:,X_has_new][N_i_new, :].sum(0)).flatten())
    elif mode=='dense_mode':
        GX[i, X_has_new] += X_Corder[np.ix_(N_i_new, X_has_new)].sum(0)

    global GX_time
    GX_time += time.time() - start
        
    start = time.time()
    
    # Calculate which of the X5 are no longer maximal
    X5 = X[:, X5_mask]
    X5_sizes = np.array(X5.sum(0)).flatten()
    to_remove = np.zeros(X.shape[1], dtype=np.bool)
    non_maximal = ((GX[:, X5_mask] + X5) == X5_sizes).sum(0) > X5_sizes
    non_maximal = np.array(non_maximal).flatten()
    to_remove[X5_mask.nonzero()[0][non_maximal]] = True
    
    # If X has a singleton clique containing i, then delete it
    i_clique = X_mask & (~X_has_Vo) & (~X_has_new) & (~X_has_old) & (X_has_i)
    assert i_clique.sum() <= 1
    X_mask_extra = X_mask & ~(X1_mask | X3_mask | i_clique | to_remove)
        
    global X5_time
    X5_time += time.time() - start
    
    if X1_mask.sum()>0 or X2_mask.sum()>0 or X3_mask.sum()>0 or X4_mask.sum()>0:
        start = time.time()

        Z_list, GZ_list = [], []
        for Z_mask, GZ_method in [[X2_mask,
                                   'from_Vo' if Vo.sum() > (N_i_new.sum() + 1) else 'from_scratch'],
                                  [X4_mask,
                                   'from_Vo' if Vo.sum() > (N_i_new.sum() + N_i_old.sum() + 1) else 'from_scratch'],
                                  [X1_mask | X3_mask,
                                   'none']]:
            
#             Z_mask = np.ones(X.shape[1], dtype=np.bool)
            
            if Z_mask.any():
#             if Z_mask_actual.any():
                Z = X[:, Z_mask]   # Since this is advanced boolean indexing, Z should be a copy
#                 Z = X_Corder
#                 assert isinstance(Z, scipy.sparse.csr_matrix)
                
#                 assert np.allclose(general_dot(Z.T, G.T).T, GX[:,Z_mask].toarray(), rtol=1e-1)
                
#                 # Consider
#                 if mode=='sparse_mode':
#                     Z = Z.toarray('C')
                
#                 if mode=='sparse_mode':
#                     assert np.all((Z[i,:] != 1.0).toarray().flatten())
#                 elif mode=='dense_mode':
#                     assert np.all(Z[i,:] != 1.0)
                    
                Z[i,:] = 1.0   # Add i
                if GZ_method=='from_Vo':
                    start_dot = time.time()
                    Vo_part = general_dot(Z[Vo, :].T, (G[:, Vo].T)).T
                    global dot_time
                    dot_time += time.time() - start_dot
                    
                    GZ = GX[:, Z_mask]
                    if scipy.sparse.issparse(GZ):
                        GZ = GZ.toarray('F')                    
                    GZ -= Vo_part
#                     GZ = GX[:, Z_mask]
#                     Vo_part = -1 * Vo_part
#                     Vo_part[GZ.nonzero()] += GZ[GZ.nonzero]
                    
                    Z[Vo, :] = 0
                elif GZ_method=='from_scratch':
                    start_dot = time.time()
                    GZ = general_dot(Z[~Vo, :].T, G[:, ~Vo].T).T
                    global dot_time
                    dot_time += time.time() - start_dot
        
                    Z[Vo, :] = 0
                elif GZ_method=='none':
                    GZ = GX[:, Z_mask]
                    if scipy.sparse.issparse(GZ):
                        GZ = GZ.toarray('F')
            
                if GZ_method in ['from_Vo', 'none']:
                    # Update to include the fact that i has been added
                    GZ += i_delta
                    start_dot = time.time()
                    GZ[i, :] = general_dot(Z.T, i_delta).T
                    global dot_time
                    dot_time += time.time() - start_dot
            
#                 print 'Curr size:', np.array(Z.sum(0)).flatten()
#                 print 'X_sizes[Z_mask]:', X_sizes[Z_mask]
#                 print 'GX_Vo[Z_mask]:', GX_Vo[Z_mask]
                
                # Size increased by 1 because of the addition of i, and reduced by removal of Vo nodes
                Z_sizes = X_sizes[Z_mask] + 1 - GX_Vo[Z_mask]
#                 assert np.all(Z_sizes == np.array(Z.sum(0)).flatten())

                start_maximal = time.time()
                # Remove those that are non-maximal
                tmp = GZ + Z
#                 print type(tmp == Z_sizes.reshape(1,-1))
#                 if scipy.sparse.isspmatrix(tmp):
                maximal = np.array((tmp == Z_sizes.reshape(1,-1)).sum(0) == Z_sizes).flatten()
                global maximal_time
                maximal_time += time.time() - start_maximal
                
#                 if not np.all(maximal):
#                     print 'Z after:'
#                     print Z.toarray()   
#                     print 'GZ:'
#                     print GZ
#                     print 'Recompute GZ:'
#                     print general_dot(Z.T, G.T).T
#                     print 'maximal:', maximal
#                 assert np.allclose(general_dot(Z.T, G.T).T, GZ, rtol=1e-1)

#                 maximal = Z_mask_actual & maximal
            
#                 assert np.all(maximal)  # Does not need to hold
                Z = Z[:, maximal]
                GZ = GZ[:, maximal]

                GZ = scipy.sparse.csc_matrix(GZ)

                Z_list.append(Z)
                GZ_list.append(GZ)
                
        global expand_time
        expand_time += time.time() - start
        
        start = time.time()
        
        Y, GY = general_hstack(Z_list), general_hstack(GZ_list)
#         Y, GY = as_colmaj(Y), np.asfortranarray(GY)
        Y, GY = as_colmaj(Y), as_colmaj(GY)
        if scipy.sparse.issparse(Y): Y.sort_indices()
#         assert Y.has_sorted_indices
        Y_hashes = get_column_hashes(Y)
        
        assert Y.shape[1] == GY.shape[1]
#         assert np.allclose(general_dot(Y.T, G.T).T, GY.toarray()), 'qwer'
    
        global Y_stacktime
        Y_stacktime += time.time() - start

        return X_mask_extra, Y, GY, Y_hashes
    else:
        return X_mask_extra, None, None, None
    
def grow_cliques2(X, X_hashes, G, dG, mode='dense_mode'):
    
    ## Figure out sparse/dense parameters
    if mode=='sparse_mode':
        as_rowmaj = scipy.sparse.csr_matrix
        as_colmaj = scipy.sparse.csc_matrix
        colany = colany_sp
    elif mode=='dense_mode':
        as_rowmaj = np.ascontiguousarray
        as_colmaj = np.asfortranarray
        colany = colany_dense
        
    G = G.copy(order='F')
    dG = dG.copy(order='F')
    
    degrees = dG.sum(0)
    
    visited_genes = set()
    
    assert G.dtype == np.float32
    GX = general_dot(X.T, G.T).T #G.dot(X)
    
    if mode=='sparse_mode':
        GX = scipy.sparse.csc_matrix(GX)
    
    X_list, GX_list, X_Corder_list = [X], [GX], [as_rowmaj(X)]
    X_hashes_list = [X_hashes]
    X_mask_list = [np.ones(X.shape[1], dtype=np.bool)]
    
    curr_hashes = set(X_hashes)
    
    max_X = 10 if (mode=='dense_mode') else 5
    
    count = 0
    while True:
        
        mask_sizes = np.array([b.sum() for b in X_mask_list])
        thresh = 20
#         to_merge = np.argsort(mask_sizes) <= (mask_sizes.size - max_X)
#         to_merge = (np.argsort(np.argsort(mask_sizes)) <= (mask_sizes.size - max_X)) | (mask_sizes < thresh)
        to_merge = (mask_sizes < thresh)
    
        if (count % 2) == 0:
            to_merge = np.ones(mask_sizes.size, dtype=np.bool)
        
        # Remerge all the X's if they are partitioned too much
#         if len(X_list) > max_X:
#         if (mask_sizes <= thresh).sum() > max_X:
        if to_merge.any():
            start = time.time()
            
#             print 'count:', count, 'Stacking Mask Sizes:', zip(mask_sizes, to_merge.astype(np.int64)), time.time()
            
            tmp = general_hstack([a[:, b] for a, b, c in zip(X_list, X_mask_list, to_merge) if c])
            if mode=='sparse_mode': tmp.sort_indices()
            X_list = [a for a, c in zip(X_list, to_merge) if not c] + [tmp]
            X_Corder_list = [a for a, c in zip(X_Corder_list, to_merge) if not c] + [as_rowmaj(tmp)]
            
            tmp = general_hstack([a[:, b] for a, b, c in zip(GX_list, X_mask_list, to_merge) if c])
            GX_list = [a for a, c in zip(GX_list, to_merge) if not c] + [tmp]
            
            tmp = np.concatenate([a[b] for a, b, c in zip(X_hashes_list, X_mask_list, to_merge) if c])
            X_hashes_list = [a for a, c in zip(X_hashes_list, to_merge) if not c] + [tmp]
            
            X_mask_list = [b for b, c in zip(X_mask_list, to_merge) if not c] + [np.ones(X_list[-1].shape[1], dtype=np.bool)]
            
            mask_sizes = np.array([b.sum() for b in X_mask_list])
#             print 'New Mask Sizes:', mask_sizes
            
#             X = general_hstack([a[:, b] for a, b in zip(X_list, X_mask_list)])
#             GX = general_hstack([a[:, b] for a, b in zip(GX_list, X_mask_list)])
#             X_hashes = np.concatenate([a[b] for a, b in zip(X_hashes_list, X_mask_list)])
            
#             X_list, GX_list, X_Corder_list = [X], [GX], [as_rowmaj(X)]
#             X_hashes_list = [X_hashes]
#             X_mask_list = [np.ones(X.shape[1], dtype=np.bool)]
    
            global stack_time
            stack_time += time.time() - start

        i = np.argmax(degrees)
                
        if degrees[i] == 0:
            print 'Scanned through %s genes' % count
            break
        else:
            count += 1
            assert i not in visited_genes
            visited_genes.add(i)
            
        start = time.time()
        
#         print '========================='
#         print 'gene:', i, 'count:', count, 'len(X_list):', len(X_list)
#         print 'G:'
#         print G.astype(np.int64)
#         print 'dG:'
#         print dG.astype(np.int64)
        
        N_i_old = G[:, i] == 1.0
        N_i_new = dG[:, i] == 1.0
        
        assert (N_i_old & N_i_new).sum()==0 and N_i_old[i]==0.0 and N_i_new[i]==0.0
        
        N_i = N_i_old | N_i_new
        dG[N_i, i] = 0.0
        dG[i, N_i] = 0.0
        G[N_i, i] = 1.0
        G[i, N_i] = 1.0
        degrees[i] = 0
        degrees[N_i_new] -= 1
        
        global neighbor_time
        neighbor_time += time.time() - start
#         print 'N_i_new:', N_i_new.nonzero()[0]
#         print 'N_i_old:', N_i_old.nonzero()[0]
        print_time()
        
        start = time.time()
        
#         edges = list(set(zip(*G.nonzero())))
#         graph = igraph.Graph(n=G.shape[0], edges=edges, directed=False)
#         %time ref_cliques = sorted([tuple(sorted(x)) for x in graph.maximal_cliques()])
#         print 'ref cliques:', ref_cliques

        for X, GX, X_hashes, X_Corder, X_mask in             itertools.islice(zip(X_list, GX_list, X_hashes_list, X_Corder_list, X_mask_list), 0, len(X_list)):
            
            if not X_mask.any():
                continue
                
#             print '-----------------------'
            X_mask_extra, Y, GY, Y_hashes = grow_X(X, X_Corder, GX, X_hashes, X_mask, G, dG, i, N_i_new, N_i_old, mode=mode)
            X_mask[:] = X_mask & X_mask_extra
            
#             print 'X mask:', X_mask.nonzero()[0]
#             my_cliques = sorted([tuple(X[:,j].nonzero()[0]) for j in range(X.shape[1]) if X_mask[j]])
#             print 'X:'
#             print my_cliques
            
            if Y is not None and Y.shape[1] > 0:
                if mode=='dense_mode':
                    assert GY.flags['F_CONTIGUOUS']
                
                start_Y = time.time()
                
                _, idx = np.unique(Y_hashes, return_index=True)
                idx = [j for j in idx if Y_hashes[j] not in curr_hashes]
                if len(idx) < Y_hashes.size:
                    Y = Y[:, idx]
                    GY = GY[:, idx]
                    Y_hashes = Y_hashes[idx]
                curr_hashes |= set(Y_hashes)
                
                global Y_unique_time
                Y_unique_time += time.time() - start_Y
                
                X_list.append(Y)
                GX_list.append(GY)
                X_Corder_list.append(as_rowmaj(Y))
                X_hashes_list.append(Y_hashes)
                X_mask_list.append(np.ones(Y.shape[1], dtype=np.bool))
                
                if mode=='sparse_mode': Y.sort_indices()
#                 if not np.all(Y_hashes == get_column_hashes(Y)):
#                     print 'mode:', mode
#                     print Y_hashes
#                     print [Y[:,i].nonzero()[0] for i in range(Y.shape[1])]
#                     print Y.sum(0)
#                     0 / asdf
                    
#                 assert np.all(Y_hashes == get_column_hashes(Y))
#                 assert np.unique(Y_hashes).size == Y_hashes.size
                
#                 my_cliques = sorted([tuple(Y[:,j].nonzero()[0]) for j in range(Y.shape[1])])
#                 print 'Y:'
#                 print my_cliques

        global loop_time
        loop_time += time.time() - start
    
    assert np.all([np.unique(a).size == a.size for a in X_hashes_list])
    
    X = general_hstack([a[:, b] for a, b in zip(X_list, X_mask_list)])
    GX = general_hstack([a[:, b] for a, b in zip(GX_list, X_mask_list)])
    
    if mode=='sparse_mode': X.sort_indices()
    X_hashes = np.concatenate([a[b] for a, b in zip(X_hashes_list, X_mask_list)])
    
#     my_cliques = sorted([tuple(X[:,i].nonzero()[0]) for i in range(X.shape[1])])
#     print my_cliques
    assert np.all(X_hashes == get_column_hashes(X))
    assert np.unique(X_hashes).size == X_hashes.size
    return X, GX, X_hashes


# In[225]:

# %%prun -s line
# %%lprun

# for count2 in range(100):
# dtype = np.bool
dtype = np.float32
# k = 4000
k = 100

G1 = np.zeros((k,k), dtype=np.bool)
# # G1[np.random.randint(0, k, k/2), np.random.randint(0, k, k/2)] = True
G1[np.random.randint(0, k, k * 8), np.random.randint(0, k, k * 8)] = True
G1 = np.logical_or(G1, G1.T)

np.fill_diagonal(G1, 0)
print 'G1 edges:', G1.sum()

G1 = G1.astype(dtype)

copy_time = 0
dot_time = 0
remove_time = 0

expand_time = 0
XY_mask_time = 0
XY_time = 0
remove_subsets_time = 0
neighbor_time = 0
maximal_time = 0
hash_time = 0
mask_time = 0
X5_time = 0
stack_time = 0
loop_time = 0
Y_unique_time = 0
GX_time = 0
count_q = 0
Y_stacktime = 0

start = time.time()
X = np.zeros((k, k), dtype=dtype, order='F')
np.fill_diagonal(X, 1)

# mode = 'sparse_mode'
mode = 'dense_mode'
if mode=='sparse_mode':
    X = scipy.sparse.csc_matrix(X)   
X_hashes = get_column_hashes(X)
# GX = np.asfortranarray(G1.dot(X))

get_ipython().magic(u'lprun -f grow_cliques2 -f grow_X X, GX, X_hashes = grow_cliques2(X, X_hashes, np.zeros(G1.shape, dtype=G1.dtype), G1, mode=mode)')

my_cliques = sorted([tuple(X[:,i].nonzero()[0]) for i in range(X.shape[1])])
edges = list(set(zip(*G1.nonzero())))
graph = igraph.Graph(n=G1.shape[0], edges=edges, directed=False)

print 'My runtime:', time.time() - start
print 'dot_time:', dot_time
# print 'remove_time:', remove_time
print 'expand_time:', expand_time
# print 'XY_time:', XY_time
# print 'copy_time:', copy_time
# print 'remove_subsets_time:', remove_subsets_time
print 'neighbor_time:', neighbor_time
print 'maximal_time:', maximal_time
print 'hash_time:', hash_time
print 'mask_time:', mask_time
# print 'XY_mask_time:', XY_mask_time
print 'X5_time:', X5_time
print 'stack_time:', stack_time
print 'loop_time:', loop_time
print 'Y_unique_time:', Y_unique_time
print 'GX_time:', GX_time
print 'count_q:', count_q
print 'Y_stacktime:', Y_stacktime
print 'equivalent loop time:', mask_time + GX_time + X5_time + Y_stacktime + expand_time

get_ipython().magic(u'time ref_cliques = sorted([tuple(sorted(x)) for x in graph.maximal_cliques()])')
# print my_cliques
# print ref_cliques
print len(my_cliques), 'cliques'
print len(ref_cliques), 'cliques'
assert my_cliques==ref_cliques


# In[ ]:

x = np.random.random((10000,10000))
x = np.asfortranarray(x)
get_ipython().magic(u'timeit np.delete(x, np.array([x.shape[0]/2]), axis=1).shape')
tmp = np.ones(x.shape[0], dtype=np.bool)
tmp[x.shape[0]/2] = False
get_ipython().magic(u'timeit x[:, tmp].shape')


# In[105]:

@jit(nopython=True, cache=True)
def numba_colany_mask_Forder(X, m):    
    a = np.zeros((X.shape[1], 4), dtype=X.dtype)    
    for i in range(X.shape[1]):
        chk1, chk2, chk3, chk4 = False, False, False, False
        for j in range(X.shape[0]):
            x_ji = X[j, i]
            chk1 = chk1 or (m[j,0] and x_ji)
            chk2 = chk2 or (m[j,1] and x_ji)
            chk3 = chk3 or (m[j,2] and x_ji)
            chk4 = chk4 or (m[j,3] and x_ji)
        a[i, 0] = chk1
        a[i, 1] = chk2
        a[i, 2] = chk3
        a[i, 3] = chk4
    return a

@jit(nopython=True, cache=True)
def numba_colany_4mask_Forder(X, m1, m2, m3, m4):    
    a = np.zeros((X.shape[1], 4), dtype=X.dtype)    
    for i in range(X.shape[1]):
        chk1, chk2, chk3, chk4 = False, False, False, False
        for j in range(X.shape[0]):
            x_ji = X[j, i]
            chk1 = chk1 or (m1[j] and x_ji)
            chk2 = chk2 or (m2[j] and x_ji)
            chk3 = chk3 or (m3[j] and x_ji)
            chk4 = chk4 or (m4[j] and x_ji)
        a[i, 0] = chk1
        a[i, 1] = chk2
        a[i, 2] = chk3
        a[i, 3] = chk4
    return a

@jit(nopython=True, cache=True)
def numba_colany_3mask_Forder(X, m1, m2, m3):    
    a = np.zeros((X.shape[1], 3), dtype=X.dtype)    
    for i in range(X.shape[1]):
        chk1 = False
        chk2 = False
        chk3 = False        
        for j in range(X.shape[0]):
            x_ji = X[j, i]
            chk1 = chk1 or (m1[j] and x_ji)
            chk2 = chk2 or (m2[j] and x_ji)
            chk3 = chk3 or (m3[j] and x_ji)
        a[i, 0] = chk1
        a[i, 1] = chk2
        a[i, 2] = chk3
    return a

@jit(nopython=True, cache=True)
def numba_colany_mask_Corder(X, m1, m2, m3, m4):    
    a = np.zeros((X.shape[1], 4), dtype=X.dtype)    
    for j in range(X.shape[0]):
        chk1, chk2, chk3, chk4 = False, False, False, False
        m1j = m1[j]
        m2j = m2[j]
        m3j = m3[j]
        m4j = m4[j]
        for i in range(X.shape[1]):
            x_ji = X[j, i]
            chk1 = chk1 or (m1j and x_ji)
            chk2 = chk2 or (m2j and x_ji)
            chk3 = chk3 or (m3j and x_ji)
            chk4 = chk4 or (m4j and x_ji)
        a[i, 0] = chk1
        a[i, 1] = chk2
        a[i, 2] = chk3
        a[i, 3] = chk4
    return a

# @jit(nopython=True, cache=True)
# def numba_colany_mask_Corder(X, m1, m2):    
# #     a = np.zeros((X.shape[1], 2), dtype=X.dtype)
#     a = np.zeros((2, X.shape[1]), dtype=X.dtype)
#     for j in range(X.shape[0]):
#         m1j = m1[j]
#         m2j = m2[j]
#         if m1j and m2j:
#             for i in range(X.shape[1]):
#                 x_ji = X[j, i]
#                 a[i,0] = a[i,0] or x_ji
#                 a[i,1] = a[i,1] or x_ji
#         elif m1j and (not m2j):
#             for i in range(X.shape[1]):
#                 x_ji = X[j, i]
#                 a[i,0] = a[i,0] or x_ji 
#         elif (not m1j) and m2j:
#             for i in range(X.shape[1]):
#                 x_ji = X[j, i]
#                 a[i,1] = a[i,1] or x_ji
#     return a


# In[221]:

# @jit(nopython=True, cache=True)
# def numba_colany_rowcol_mask_Forder(X, col_mask, m):
# #     a = np.zeros((col_mask.sum(), 4), dtype=boolean)
#     a = np.zeros((col_mask.sum(), 3), dtype=boolean)
#     count = 0
# #     X = X.T
#     for i in range(X.shape[1]):
# #     for i in range(X.shape[0]):
#         if col_mask[i]:
#             chk1, chk2, chk3 = False, False, False
# #             for j in range(X.shape[1]):
#             for j in range(X.shape[0]):
# #                 x_ji = X[i, j]
#                 x_ji = X[j, i]
#                 chk1 = chk1 or (m[j,0] and x_ji!=0)
#                 chk2 = chk2 or (m[j,1] and x_ji!=0)
#                 chk3 = chk3 or (m[j,2] and x_ji!=0)
#             a[count, 0] = chk1
#             a[count, 1] = chk2
#             a[count, 2] = chk3
#             count += 1
#     return a

@vectorize([boolean(float32)], nopython=True)
def numba_nz(x):
    return x != 0

@vectorize([boolean(boolean, boolean, boolean)], nopython=True)
def numba_helper(x, y, z):
    return (x or (y and z))
# chk1 or (m[j,0] and nnz)

@jit(nopython=True, cache=True)
def numba_colany_rowcol_mask_Forder(X, col_mask, m):
    a = np.zeros((col_mask.size, 3), dtype=boolean)
    for k in range(col_mask.size):
        i = col_mask[k]

        chk1, chk2, chk3 = False, False, False
        Y = numba_nz(X[:, i])
        for j in range(Y.size):
            nnz = Y[j]
            chk1 = chk1 or (m[j,0] and nnz)
            chk2 = chk2 or (m[j,1] and nnz)
            chk3 = chk3 or (m[j,2] and nnz)

        a[k, 0] = chk1
        a[k, 1] = chk2
        a[k, 2] = chk3
#         count += 1
    return a


# In[47]:

get_ipython().magic(u'time m = np.ascontiguousarray(np.vstack([m1,m2,m3,m4]).T)')
col_mask = np.random.random(x.shape[1]) < 0.5
for q in range(10):
    print q
#     %time tmp = numba_colany_rowcol_mask_Forder(x, col_mask, m)
    get_ipython().magic(u'time tmp = numba_colany_rowcol_mask_Forder(x, col_mask.nonzero()[0], m)')
#     %time tmp2 = x_c[np.ix_(m1, col_mask)].any(0)
#     assert np.allclose(tmp2, tmp[:,0], rtol=1e-1)
#     assert np.allclose(tmp2, make_mask(tmp[:,0], col_mask), rtol=1e-1)
    get_ipython().magic(u'time tmp2 = x_c[np.ix_(m1, col_mask)].any(0) ; tmp2 = x_c[np.ix_(m2, col_mask)].any(0) ; tmp2 = x_c[np.ix_(m3, col_mask)].any(0)')
# assert np.allclose(tmp2, tmp[:,0], rtol=1e-1)


# In[631]:

get_ipython().magic(u'time tmp = x[m1, :][:, m2].any(0)')
# print np.ix_(m1, m2).flags['F_CONTIGUOUS']
get_ipython().magic(u'time tmp = x[np.ix_(m1, m2)]')
print tmp.flags['F_CONTIGUOUS']
get_ipython().magic(u'time tmp = x.T[np.ix_(m1, m2)]')
print tmp.flags['F_CONTIGUOUS']
get_ipython().magic(u'time tmp = x_c[np.ix_(m1, m2)]')
print tmp.flags['F_CONTIGUOUS']
# %time tmp = x.T[np.ix_(m1, m2)].any(0)
# print tmp.flags['F_CONTIGUOUS']


# In[2]:

x = (np.random.random((10000, 10000)) < 0.0001).astype(np.float32)
x = np.asfortranarray(x)
# x = np.ascontiguousarray(x)
x_c = np.ascontiguousarray(x)
i = 5
# %time x[i, :]
get_ipython().magic(u'time x.any(0)')
part = np.random.randint(0, 4, x.shape[0])
m1, m2, m3, m4 = part==0, part==1, part==2, part==3
m = np.ascontiguousarray(np.vstack([m1,m2,m3,m4]).T)
# print m.dtype, m.shape, 0/asdf
get_ipython().magic(u'time tmp = numba_colany_mask_Forder(x, m)')
get_ipython().magic(u'time tmp = numba_colany_4mask_Forder(x, m1, m2, m3, m4)')
get_ipython().magic(u'time tmp = numba_colany_3mask_Forder(x, m1, m2, m3)')

# %time x = np.ascontiguousarray(x)
for i, m in enumerate([m1,m2,m3,m4]):
# for i, m in enumerate([m1]):
    get_ipython().magic(u'time tmp3 = x[m, :].any(0)')
#     %time tmp2 = tmp[i, :] == x[m, :].any(0)
#     assert np.all(tmp2)


# # Scratch

# In[ ]:

def grow_cliques2(X, X_hashes, G, dG):
    
    G = G.copy(order='F')
    dG = dG.copy(order='F')
    
    degrees = dG.sum(0)
    
    visited_genes = set()
    
    count = 0
    while True:
        i = np.argmax(degrees)
                
        if degrees[i] == 0:
            print 'Scanned through %s genes' % count
            break
        else:
            count += 1
            assert i not in visited_genes
            visited_genes.add(i)
        
        assert X.flags['F_CONTIGUOUS']
#         print 'gene:', i, 'count:', count
        print_time()
            
        start = time.time()
#         print 'G:'
#         print G.astype(np.int64)
#         print 'dG:'
#         print dG.astype(np.int64)
        
        N_i_old = G[:, i] == 1.0
        N_i_new = dG[:, i] == 1.0
        assert (N_i_old & N_i_new).sum() == 0
        
        N_i = N_i_old | N_i_new
        dG[N_i, i] = 0.0
        dG[i, N_i] = 0.0
        G[N_i, i] = 1.0
        G[i, N_i] = 1.0
        degrees[i] = 0
        degrees[N_i_new] -= 1
        
        global neighbor_time
        neighbor_time += time.time() - start
        
#         print 'N_i_new:', N_i_new.nonzero()[0]
#         print 'N_i_old:', N_i_old.nonzero()[0]
        print_time()
        
        start = time.time()
        
        i_mask = np.zeros(G.shape[0], dtype=np.bool)
        i_mask[i] = True
        Vo = (~ (N_i_new | N_i_old | i_mask))
        
#         X = np.ascontiguousarray(X)
        
#         tmp = numba_colany_3mask_Forder(X, Vo, N_i_new, N_i_old)
#         X_has_Vo = tmp[:,0]
#         X_has_new = tmp[:,1]
#         X_has_old = tmp[:,2]
        
        X_has_Vo = X[Vo, :].any(0)
        X_has_new = X[N_i_new, :].any(0)
        X_has_old = X[N_i_old, :].any(0)
        
        X_has_i = X[i, :]==1.0
        
#         X = np.asfortranarray(X)

        global mask_time
        mask_time += time.time() - start
        
#         print X_has_Vo
#         print X_has_new
#         print X_has_old
#         print X_has_i
        
        X1_mask = (~X_has_Vo) & X_has_new & (~X_has_old) & (~X_has_i)
        X2_mask = (X_has_Vo) & X_has_new & (~X_has_old) & (~X_has_i)
        X3_mask = (~X_has_Vo) & X_has_new & (X_has_old) & (~X_has_i)
        X4_mask = (X_has_Vo) & X_has_new & (X_has_old) & (~X_has_i)        
        X5_mask = (~X_has_Vo) & (~X_has_new) & (X_has_old) & (X_has_i)
        
        X5 = np.asfortranarray(X[:, X5_mask][N_i_new | N_i_old | i_mask, :])
        
        if X1_mask.sum()>0 or X2_mask.sum()>0 or X3_mask.sum()>0 or X4_mask.sum()>0:
            start = time.time()
            
#             Y1 = X[:, X1_mask][N_i_new | i_mask, :]            
#             Y2 = X[:, X2_mask][N_i_new | i_mask , :]            
#             Y3 = X[:, X3_mask][N_i_new | N_i_old | i_mask, :]            
#             Y4 = X[:, X4_mask][N_i_new | N_i_old | i_mask, :]
                
            Y1 = X[N_i_new | i_mask, :][:, X1_mask]
            Y2 = X[N_i_new | i_mask , :][:, X2_mask]
            Y3 = X[N_i_new | N_i_old | i_mask, :][:, X3_mask]
            Y4 = X[N_i_new | N_i_old | i_mask, :][:, X4_mask]
            
            start_copy = time.time()
            # Probably not necessary, but need to check that Y1, Y2, Y3, Y4 are views and their mutation does not affect X
#             Y1 = Y1.copy(order='F')
#             Y2 = Y2.copy(order='F')
#             Y3 = Y3.copy(order='F')
#             Y4 = Y4.copy(order='F')
            
            Y1[(N_i_new | i_mask)[:i].sum(),:] = 1.0
            Y2[(N_i_new | i_mask)[:i].sum(),:] = 1.0
            Y3[(N_i_new | N_i_old | i_mask)[:i].sum(),:] = 1.0
            Y4[(N_i_new | N_i_old | i_mask)[:i].sum(),:] = 1.0
            
            Y1_hashes = get_column_hashes(Y1)
            Y2_hashes = get_column_hashes(Y2)
            Y3_hashes = get_column_hashes(Y3)
            Y4_hashes = get_column_hashes(Y4)
            
            global expand_time
            expand_time += time.time() - start
      
            start = time.time()

            # Update: is necessary
            tmp = Y2.shape[1]
            Y2, Y2_hashes, _ = get_unique_cols(Y1, Y1_hashes, Y2, Y2_hashes)
#             assert Y2.shape[1] == tmp
            
            # Update: found to be necessary
            tmp = Y2.shape[1]
            Y2, Y2_hashes = maximal_sets(Y2, Y2_hashes)
#             assert Y2.shape[1] == tmp
            
            # Update: found to be necessary
            tmp = Y1.shape[1] + Y2.shape[1]
            YA, YA_hashes = remove_subsets(Y1, Y1_hashes, Y2, Y2_hashes, False)
            assert YA.shape[1] == tmp

#             print 'YA:', YA.shape[1], 'Y1:', Y1.shape[1], 'Y2:', Y2.shape[1]
            print_time()

            # Fill YA with rest of genes
            tmp = np.zeros((X.shape[0], YA.shape[1]), order='F', dtype=X.dtype)
            tmp[N_i_new | i_mask, :] = YA
            YA = tmp
            YA_hashes = get_column_hashes(YA)

            # Update: found to be necessary
            tmp = Y4.shape[1]
            Y4, Y4_hashes = get_unique_cols(Y4, Y4_hashes)[:2]
#             assert Y4.shape[1] == tmp
        
            # Update: found to be necessary
            tmp = Y4.shape[1]
            Y4, Y4_hashes = maximal_sets(Y4,Y4_hashes)
#             assert Y4.shape[1] == tmp
    
            # Update: found to be necessary. Current understanding: Some Y4's may be subset of Y3
            YB, YB_hashes = remove_subsets(Y3, Y3_hashes, Y4, Y4_hashes, False)
#             print 'YB:', YB.shape[1], 'Y3:', Y3.shape[1], 'Y4:', Y4.shape[1]
            print_time()
    
            # Combine YB with X5
            # 0 0 1 1            
            YB, YB_hashes = remove_subsets(X5, get_column_hashes(X5), YB, YB_hashes, False)
            
            # Fill YB with rest of genes
            tmp = np.zeros((X.shape[0], YB.shape[1]), order='F', dtype=X.dtype)
            tmp[N_i_new | N_i_old | i_mask, :] = YB
            YB = tmp
            YB_hashes = get_column_hashes(YB)
            
            # Combine YA and YB
            Y, Y_hashes = remove_subsets(YA, YA_hashes, YB, YB_hashes, False)
#             print 'Y, YA + YB, YA, YB:', Y.shape[1], YA.shape[1] + YB.shape[1], YA.shape[1], YB.shape[1]
            print_time()
            
            global remove_time
            remove_time += time.time() - start
            
            start = time.time()

            # If X has a singleton clique containing i, then delete it
            i_clique = (~X_has_Vo) & (~X_has_new) & (~X_has_old) & (X_has_i)
            if i_clique.sum() > 0:
                assert i_clique.sum() == 1
            X = X[:, ~(X1_mask | X3_mask | X5_mask | i_clique)]
            X_hashes = X_hashes[~(X1_mask | X3_mask | X5_mask | i_clique)]
            
            global XY_mask_time
            XY_mask_time += time.time() - start
            
            Y, Y_hashes, _ = get_unique_cols(X, X_hashes, Y, Y_hashes)
            X, X_hashes = np.hstack([X, Y]), np.concatenate([X_hashes, Y_hashes])
            assert X.flags['F_CONTIGUOUS']
            print_time()

#             # Check for correctness
#             tmp = X.shape[1]
#             X, X_hashes = maximal_sets(X, X_hashes)
#             assert X.shape[1] == tmp
            
            global XY_time
            XY_time += time.time() - start
            
#     my_cliques = sorted([tuple(X[:,i].nonzero()[0]) for i in range(X.shape[1])])
#     print my_cliques
    assert np.all(X_hashes == get_column_hashes(X))
    assert np.unique(X_hashes).size == X_hashes.size
    return X, X_hashes


# In[ ]:

def grow_X(X, X_Corder, GX, X_hashes, X_mask, G, dG, i, N_i_new, N_i_old):
    
    ## Figure out sparse/dense parameters
    if scipy.sparse.issparse(X):
        general_hstack = scipy.sparse.hstack
        as_rowmaj = scipy.sparse.csr_matrix
        as_colmaj = scipy.sparse.csc_matrix
        def colany(a):
            return (a.indices[1:] - a.indices[:-1]) > 0
    else:
        general_hstack = np.hstack
        as_rowmaj = np.ascontiguousarray
        as_colmaj = np.asfortranarray
        colany = lambda a: np.any(a, axis=0)
        
    i_mask = np.zeros(G.shape[0], dtype=np.bool)
    i_mask[i] = True
    Vo = (~ (N_i_new | N_i_old | i_mask))
    
    start = time.time()
    
    assert not X_Corder.flags['F_CONTIGUOUS']
    def make_mask(sub_mask, orig_mask):
        mask = np.zeros(orig_mask.size, dtype=np.bool)
        mask[orig_mask.nonzero()[0][sub_mask]] = True
        return mask

    if scipy.sparse.issparse(X):
        X_has_Vo = make_mask(X[:,X_mask][Vo,:].any(0), X_mask)
        X_has_new = make_mask(X[:,X_mask][N_i_new,:].any(0), X_mask)
        X_has_old = make_mask(X[:,X_mask][N_i_old,:].any(0), X_mask)
        X_has_i = make_mask(X_Corder[i, X_mask]==1.0, X_mask)        
    else:
        m = np.ascontiguousarray(np.vstack([Vo, N_i_new, N_i_old]).T)
        tmp = numba_colany_rowcol_mask_Forder(X, X_mask.nonzero()[0], m)
        X_has_Vo = make_mask(tmp[:,0], X_mask)
        X_has_new = make_mask(tmp[:,1], X_mask)
        X_has_old = make_mask(tmp[:,2], X_mask)

    #     X_has_Vo = make_mask(X_Corder[np.ix_(Vo, X_mask)].any(0), X_mask)
    #     X_has_new = make_mask(X_Corder[np.ix_(N_i_new, X_mask)].any(0), X_mask)
    #     X_has_old = make_mask(X_Corder[np.ix_(N_i_old, X_mask)].any(0), X_mask)
        X_has_i = make_mask(X_Corder[i, X_mask]==1.0, X_mask)

    global mask_time
    mask_time += time.time() - start

    start = time.time()
    
    X1_mask = X_mask & (~X_has_Vo) & X_has_new & (~X_has_old) & (~X_has_i)
    X2_mask = X_mask & (X_has_Vo) & X_has_new & (~X_has_old) & (~X_has_i)
    X3_mask = X_mask & (~X_has_Vo) & X_has_new & (X_has_old) & (~X_has_i)
    X4_mask = X_mask & (X_has_Vo) & X_has_new & (X_has_old) & (~X_has_i)        
    X5_mask = X_mask & (~X_has_Vo) & (~X_has_new) & (X_has_old) & (X_has_i)

    # Update GX to include i
    i_delta = G[:,i].reshape(-1,1)
    GX[:, X_has_i] += N_i_new.reshape(-1,1)
    GX[i, X_has_new] += X_Corder[N_i_new, :][:,X_has_new].sum(0)
    
    global GX_time
    GX_time += time.time() - start

    start = time.time()
    
    # Calculate which of the X5 are no longer maximal
    X5 = X[:, X5_mask]
    X5_sizes = X5.sum(0)
    to_remove = np.zeros(X.shape[1], dtype=np.bool)
    non_maximal = ((GX[:, X5_mask] + X5) == X5_sizes).sum(0) > X5_sizes
    to_remove[X5_mask.nonzero()[0][non_maximal]] = True
    
    # If X has a singleton clique containing i, then delete it
    i_clique = X_mask & (~X_has_Vo) & (~X_has_new) & (~X_has_old) & (X_has_i)
    assert i_clique.sum() <= 1
    X_mask_extra = X_mask & ~(X1_mask | X3_mask | i_clique | to_remove)
        
    global X5_time
    X5_time += time.time() - start
    
    if X1_mask.sum()>0 or X2_mask.sum()>0 or X3_mask.sum()>0 or X4_mask.sum()>0:
        start = time.time()

        Z_list, GZ_list = [], []
        for Z_mask, GZ_method in [[X2_mask,
                                   'from_Vo' if Vo.sum() > (N_i_new.sum() + 1) else 'from_scratch'],
                                  [X4_mask,
                                   'from_Vo' if Vo.sum() > (N_i_new.sum() + N_i_old.sum() + 1) else 'from_scratch'],
                                  [X1_mask | X3_mask,
                                   'none']]:
            if Z_mask.sum() > 0:
                Z = X[:, Z_mask]   # Since this is advanced boolean indexing, Z should be a copy
                assert np.all(Z[i,:] != 1.0)
                Z[i,:] = 1.0   # Add i
                if GZ_method=='from_Vo':
                    GZ = GX[:, Z_mask] - G[:, Vo].dot(Z[Vo, :])
                    Z[Vo, :] = 0
                elif GZ_method=='from_scratch':
                    GZ = G[:, ~Vo].dot(Z[~Vo, :])
                    Z[Vo, :] = 0
                elif GZ_method=='none':
                    GZ = GX[:, Z_mask]
                GZ += i_delta # Update to include the fact that i has been added
                GZ[i, :] = G[:,i].reshape(1,-1).dot(Z)
                Z_sizes = Z.sum(0)
                Z = Z[:, ((GZ + Z) == Z_sizes.reshape(1,-1)).sum(0) == Z_sizes]

                Z_list.append(Z)
                GZ_list.append(GZ)
            
        Y, GY = general_hstack(Z_list), general_hstack(GZ_list)
        Y, GY = as_colmaj(Y), as_colmaj(GY)
        Y_hashes = get_column_hashes(Y)

        global expand_time
        expand_time += time.time() - start

        return X_mask_extra, Y, GY, Y_hashes
    else:
        return X_mask_extra, None, None, None
    
def grow_cliques2(X, X_hashes, G, dG):
    
    ## Figure out sparse/dense parameters
    if scipy.sparse.issparse(X):
        general_hstack = scipy.sparse.hstack
        as_rowmaj = scipy.sparse.csr_matrix
        as_colmaj = scipy.sparse.csc_matrix
    else:
        general_hstack = np.hstack
        as_rowmaj = np.ascontiguousarray
        as_colmaj = np.asfortranarray
    
    G = G.copy(order='F')
    dG = dG.copy(order='F')
    
    degrees = dG.sum(0)
    
    visited_genes = set()
    
    assert G.dtype == np.float32
    GX = X.T.dot(G).T #G.dot(X)
    
    X_list, GX_list, X_Corder_list = [X], [GX], [as_rowmaj(X)]
    X_hashes_list = [X_hashes]
    X_mask_list = [np.ones(X.shape[1], dtype=np.bool)]
    
    curr_hashes = set(X_hashes)
    
    count = 0
    while True:
        
        # Remerge all the X's if they are partitioned too much
        if len(X_list) > 10:
            start = time.time()
            
            X = general_hstack([a[:, b] for a, b in zip(X_list, X_mask_list)])
            GX = general_hstack([a[:, b] for a, b in zip(GX_list, X_mask_list)])
            X_hashes = np.concatenate([a[b] for a, b in zip(X_hashes_list, X_mask_list)])
            
            X_list, GX_list, X_Corder_list = [X], [GX], [np.ascontiguousarray(X)]
            X_hashes_list = [X_hashes]
            X_mask_list = [np.ones(X.shape[1], dtype=np.bool)]
    
            global stack_time
            stack_time += time.time() - start

        i = np.argmax(degrees)
                
        if degrees[i] == 0:
            print 'Scanned through %s genes' % count
            break
        else:
            count += 1
            assert i not in visited_genes
            visited_genes.add(i)
            
        start = time.time()
        
#         print '========================='
#         print 'gene:', i, 'count:', count, 'len(X_list):', len(X_list), 
#         print 'G:'
#         print G.astype(np.int64)
#         print 'dG:'
#         print dG.astype(np.int64)
        
        N_i_old = G[:, i] == 1.0
        N_i_new = dG[:, i] == 1.0
        assert (N_i_old & N_i_new).sum() == 0
        
        N_i = N_i_old | N_i_new
        dG[N_i, i] = 0.0
        dG[i, N_i] = 0.0
        G[N_i, i] = 1.0
        G[i, N_i] = 1.0
        degrees[i] = 0
        degrees[N_i_new] -= 1
        
        global neighbor_time
        neighbor_time += time.time() - start
#         print 'N_i_new:', N_i_new.nonzero()[0]
#         print 'N_i_old:', N_i_old.nonzero()[0]
        print_time()
        
        start = time.time()
        
        for X, GX, X_hashes, X_Corder, X_mask in             itertools.islice(zip(X_list, GX_list, X_hashes_list, X_Corder_list, X_mask_list), 0, len(X_list)):
            
#             print '-----------------------'
            X_mask_extra, Y, GY, Y_hashes = grow_X(X, X_Corder, GX, X_hashes, X_mask, G, dG, i, N_i_new, N_i_old)
            X_mask[:] = X_mask & X_mask_extra
            
#             print 'X mask:', X_mask.nonzero()[0]
#             my_cliques = sorted([tuple(X[:,j].nonzero()[0]) for j in range(X.shape[1]) if X_mask[j]])
#             print 'X:'
#             print my_cliques
            
            if Y is not None and Y.shape[1] > 0:                
                assert GY.flags['F_CONTIGUOUS']
                
                start_Y = time.time()
                
                _, idx = np.unique(Y_hashes, return_index=True)
                idx = [j for j in idx if Y_hashes[j] not in curr_hashes]
                if len(idx) < Y_hashes.size:
                    Y = Y[:, idx]
                    GY = GY[:, idx]
                    Y_hashes = Y_hashes[idx]
                curr_hashes |= set(Y_hashes)
                
                global Y_unique_time
                Y_unique_time += time.time() - start_Y
                
                X_list.append(Y)
                GX_list.append(GY)
                X_Corder_list.append(np.ascontiguousarray(Y))
                X_hashes_list.append(Y_hashes)
                X_mask_list.append(np.ones(Y.shape[1], dtype=np.bool))
                
#                 assert np.all(Y_hashes == get_column_hashes(Y))
#                 assert np.unique(Y_hashes).size == Y_hashes.size
                
#                 my_cliques = sorted([tuple(Y[:,j].nonzero()[0]) for j in range(Y.shape[1])])
#                 print 'Y:'
#                 print my_cliques

        global loop_time
        loop_time += time.time() - start
    
    assert np.all([np.unique(a).size == a.size for a in X_hashes_list])
    
    X = general_hstack([a[:, b] for a, b in zip(X_list, X_mask_list)])
    GX = general_hstack([a[:, b] for a, b in zip(GX_list, X_mask_list)])
    X_hashes = np.concatenate([a[b] for a, b in zip(X_hashes_list, X_mask_list)])
    
#     my_cliques = sorted([tuple(X[:,i].nonzero()[0]) for i in range(X.shape[1])])
#     print my_cliques
    assert np.all(X_hashes == get_column_hashes(X))
    assert np.unique(X_hashes).size == X_hashes.size
    return X, GX, X_hashes

