import sys
import cPickle, scipy, scipy.sparse, argparse, os, pickle, datetime, gzip, subprocess, StringIO, random, sys, tempfile, shutil, igraph, multiprocessing
from itertools import combinations, chain, groupby, compress, permutations, product
import numpy as np
import glob
from collections import Counter
import pandas as pd

from ctypes import pointer, Structure, POINTER,c_void_p,c_int,c_char,c_double,byref,cdll, c_long, c_float, c_int64
import ctypes
mklso = cdll.LoadLibrary("libmkl_rt.so")

from ctypes import *
import scipy.sparse as spsp
import numpy as np
import multiprocessing as mp
import time



from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc


# # Sparse x Sparse

# In[2]:

# June 2nd 2016 version.

# http://stackoverflow.com/questions/37536106/directly-use-intel-mkl-library-on-scipy-sparse-matrix-to-calculate-a-dot-a-t-wit
# https://github.com/srkiranraj/spgemm/blob/master/spmm_mkl.cu

# Load the share library
mklso = cdll.LoadLibrary("libmkl_rt.so")

def get_csr_handle(A, to_1=True):
    a_pointer   = A.data.ctypes.data_as(POINTER(c_float))
    
    if to_1:
        # Convert to 1-based indices
        x, y = A.indices + 1, A.indptr + 1
        ja_pointer  = x.ctypes.data_as(POINTER(c_int))
        ia_pointer  = y.ctypes.data_as(POINTER(c_int))
    else:
        ja_pointer  = A.indices.ctypes.data_as(POINTER(c_int))
        ia_pointer  = A.indptr.ctypes.data_as(POINTER(c_int))
    
    return (a_pointer, ja_pointer, ia_pointer)

##########################
# Multiply CSR times CSR
##########################
def csr_dot_csr_t(A, B, to_1=True, verbose=False):

    A_copy = A.copy()
    A_copy.indices += 1
    A_copy.indptr += 1
    
    a_pointer, ja_pointer, ia_pointer = get_csr_handle(A_copy, to_1=False)

    trans_pointer   = byref(c_char('N'))
#     sort_pointer    = byref(c_int(0))
    sort_pointer = byref(c_int(0))

    assert A.shape[1] == B.shape[0]
    (m, n), k = A.shape, B.shape[1]
    
    sort_pointer        = byref(c_int(0))
    m_pointer           = byref(c_int(m))     # Number of rows of matrix A
    n_pointer           = byref(c_int(n))     # Number of columns of matrix A
    
    ### SHOULDN'T THIS BE byref(c_int(k))???
    k_pointer           = byref(c_int(n))     # Number of columns of matrix B
    
    B_copy = B.copy()
    B_copy.indices += 1
    B_copy.indptr += 1
    
    b_pointer, jb_pointer, ib_pointer = get_csr_handle(B_copy, to_1=False)

    info = c_int(-3)
    info_pointer = byref(info)

    ic = np.ones((A.shape[0] + 1, ), dtype=np.int32)
    ic_pointer = ic.ctypes.data_as(POINTER(c_int))
    
    ## If np.float64, then call dcsrmultcsr
    assert A.dtype == np.float32
    
    jc = np.empty((20, ), dtype=np.int32)
    jc_pointer = jc.ctypes.data_as(POINTER(c_int))
    c = np.empty((20, ), dtype=A.dtype)
    if A.dtype == np.float32:
        c_pointer = c.ctypes.data_as(POINTER(c_float))
    
    request_pointer_list = [byref(c_int(0)), byref(c_int(1)), byref(c_int(2))]
    
    start = time.time()
    ret = mklso.mkl_scsrmultcsr(trans_pointer, request_pointer_list[1], sort_pointer,
                m_pointer, n_pointer, k_pointer,
                a_pointer, ja_pointer, ia_pointer,
                b_pointer, jb_pointer, ib_pointer,
                c_pointer, jc_pointer, ic_pointer,
                byref(c_int(0)), info_pointer)
    if verbose: print 'Request 1:', time.time() - start
    
    info_val = info.value
    result_list = [(ret, info_val)]
    
    predicted_nnz = ic[A.shape[0]] - 1
    if verbose: print 'predicted nnz:', predicted_nnz
    
    jc = np.empty((predicted_nnz, ), dtype=np.int32)
    jc_pointer = jc.ctypes.data_as(POINTER(c_int))
    c = np.empty((predicted_nnz, ), dtype=A.dtype)
    assert A.dtype == np.float32
    if A.dtype == np.float32:
        c_pointer = c.ctypes.data_as(POINTER(c_float))
    
#     for i in range(8):
    for i in [7]:
        print i,
        sort_pointer = byref(c_int(i))
        
        start = time.time()
        ret = mklso.mkl_scsrmultcsr(trans_pointer, request_pointer_list[2], sort_pointer,
                    m_pointer, n_pointer, k_pointer,
                    a_pointer, ja_pointer, ia_pointer,
                    b_pointer, jb_pointer, ib_pointer,
                    c_pointer, jc_pointer, ic_pointer,
                    byref(c_int(0)), info_pointer)
        if verbose: print 'Request 2:', time.time() - start
    
    result_list.append((ret, info_val))
    
    C = scipy.sparse.csr_matrix((c, jc - 1, ic - 1), (A.shape[0], B.shape[1]))
    return C

def show_csr_internal(A, indent=4):
    # Print data, indptr, and indices
    # of a scipy csr_matrix A
    name = ['data', 'indptr', 'indices']
    mat  = [A.data, A.indptr, A.indices]
    for i in range(3):
        str_print = ' '*indent+name[i]+':\n%s'%mat[i]
        str_print = str_print.replace('\n', '\n'+' '*indent*2)
        print(str_print)


# # Sparse x Dense

# In[3]:

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
    ret = mkl.mkl_scsrmm(trans_pointer, 
                         m_pointer, n_pointer, k_pointer,
                         byref(c_float(1.0)),
                         matdescra_pointer,
                         val, indx, pntrb, pntre,
                         b_pointer, n_pointer,
                         byref(c_float(0.0)),
                         c_pointer, n_pointer)
    print 'Time:', time.time() - start
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
    ret = mkl.mkl_scscmm(trans_pointer, 
                         m_pointer, n_pointer, k_pointer,
                         byref(c_float(1.0)),
                         matdescra_pointer,
                         val, indx, pntrb, pntre,
                         b_pointer, n_pointer,
                         byref(c_float(0.0)),
                         c_pointer, n_pointer)
    print 'Time:', time.time() - start
    return C

def dot(X, Y):

    if issparse(X):     
        # assert Y.flags['C_CONTIGUOUS']
        # if isspmatrix_csr(X):
        #     return csrmm(X, Y)
        # elif isspmatrix_csc(X):
        #     return cscmm(X, Y)
        # else:
        #     raise Exception('Not supported')
        return X.dot(Y)
    elif issparse(Y):
        return Y.T.dot(X.T).T
    else:
#        assert not issparse(Y)
#         return np.dot(X, Y)
        return X.dot(Y)


def elt_multiply(X, Y):
    if issparse(X):
        return X.multiply(Y)
    elif issparse(Y):
        return Y.multiply(X)
    else:
        return X * Y
