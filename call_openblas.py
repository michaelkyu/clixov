from numba import guvectorize
import os
os.environ['OPENBLAS_NUM_THREADS']='48'
os.environ['GOTO_NUM_THREADS']='48'
os.environ['OMP_NUM_THREADS']='48'

import sys
import numpy as np

import ctypes
from ctypes import cdll, c_float, POINTER, c_int, byref, c_char, c_double
lib = cdll.LoadLibrary('/cellar/users/mikeyu/OpenBLAS/libopenblas.so')
import scipy, scipy.sparse, time


def dgemm_openblas(x, y, n_threads=None):
    m, k = x.shape
    n = y.shape[1]
    assert k==y.shape[0]
    assert x.dtype==np.float64 and y.dtype==np.float64
    
    if n_threads is not None:
        lib.goto_set_num_threads(c_int(n_threads))
        lib.openblas_set_num_threads(c_int(n_threads))
    
    x_ptr = x.ctypes.data_as(POINTER(c_double))
    y_ptr = y.ctypes.data_as(POINTER(c_double))
    c = np.empty((m,n), dtype=np.float64, order='F')
    c_ptr = c.ctypes.data_as(POINTER(c_double))

    lib.dgemm_(byref(c_char('N')),
               byref(c_char('N')),
               byref(c_int(m)),
               byref(c_int(n)),
               byref(c_int(k)),
               byref(c_double(1)),
               x_ptr,
               byref(c_int(m)),
               y_ptr,
               byref(c_int(k)),
               byref(c_double(0)),
               c_ptr,
               byref(c_int(m)))
    
    return c

from numba import int64, float32, float64, boolean, guvectorize, jit, vectorize

@guvectorize(['void(float32[:], float32[:,:], float32[:])'],
             '(n),(n,p)->(p)', nopython=True, target='parallel')
def mat_minplus(A, B, C):
    n, p = B.shape
    for j in range(p):
        curr_min = 10000000
        for k in range(n):
            D = A[k] + B[k, j]
            if D < curr_min:
                curr_min = D
        C[j] = curr_min
        
@guvectorize(['void(float32[:], float32[:,:], float32[:])'],
             '(n),(n,p)->(p)', nopython=True, target='parallel')
def mat_maxplus(A, B, C):
    n, p = B.shape
    for j in range(p):
        curr_min = -10000000
        for k in range(n):
            D = A[k] + B[k, j]
            if D > curr_min:
                curr_min = D
        C[j] = curr_min
        
@guvectorize(['void(float32[:], float32[:,:], float32[:])'],
             '(n),(n,p)->(p)', nopython=True, target='parallel')
def mat_maxmult(A, B, C):
    n, p = B.shape
    for j in range(p):
        curr_max = -10000000
        for k in range(n):
            D = A[k] * B[k, j]
            if D > curr_max:
                curr_max = D
        C[j] = curr_max
        
@guvectorize(['void(float32[:], float32[:,:], float32[:])'],
             '(n),(n,p)->(p)', nopython=True, target='parallel')
def mat_minmult(A, B, C):
    n, p = B.shape
    for j in range(p):
        curr = 10000000.
        for k in range(n):
            D = A[k] * B[k, j]
            if D < curr:
                curr = D
        C[j] = curr


@guvectorize(['void(float32[:], float32[:,:], float32[:])'],
             '(n),(n,p)->(p)', nopython=True, target='parallel')
def jaccard_gu(A, B, C):
    n, p = B.shape
    for j in range(p):
        min_sum = 0
        for k in range(n):
            min_sum += min(A[k], B[k, j])
        C[j] = min_sum
     
    for j in range(p):
        max_sum = 0
        for k in range(n):
            max_sum += max(A[k], B[k, j])
        C[j] = C[j] / max_sum


# In[4]:

## Xor, Min-plus implementation. To be used with sgemm_kernel_16x4_sandy.S_maxplus3

# def sgemm_openblas(x, y, n_threads=None, minimize=False):
# #     assert np.isfortran(x) and np.isfortran(y)
#     assert x.flags['F_CONTIGUOUS'] and y.flags['F_CONTIGUOUS']
    
#     m, k = x.shape
#     n = y.shape[1]
#     assert k==y.shape[0]
#     assert x.dtype==np.float32 and y.dtype==np.float32

#     if minimize:
# #         max_elt = np.max([np.max(x), np.max(y)])
# #         max_elt += 1.
#         max_elt = 10000.
#         x = max_elt - np.abs(x)
#         y = max_elt - np.abs(y)
    
#     print 'x:'
#     print x[0,:]
#     print 'y:'
#     print y[:,0]
    
#     print 'max:'
#     i, j = 0, 0
#     tmp = (x[i,:]+y[:,j]) * np.int64((x[i,:]>0) != (y[:,j]>0))
#     print tmp
#     if (tmp!=0).sum()>0:
#         print 'max elt:', tmp[tmp != 0].max()
#     else:
#         print "all 0's"
        
#     if n_threads is not None:
#         lib.goto_set_num_threads(c_int(n_threads))
#         lib.openblas_set_num_threads(c_int(n_threads))
    
#     x_ptr = x.ctypes.data_as(POINTER(c_float))
#     y_ptr = y.ctypes.data_as(POINTER(c_float))
#     c = np.empty((m,n), dtype=np.float32, order='F')
#     c_ptr = c.ctypes.data_as(POINTER(c_float))

#     start = time.time()
#     lib.sgemm_(byref(c_char('N')),
#                byref(c_char('N')),
#                byref(c_int(m)),
#                byref(c_int(n)),
#                byref(c_int(k)),
#                byref(c_float(1)),
#                x_ptr,
#                byref(c_int(m)),
#                y_ptr,
#                byref(c_int(k)),
#                byref(c_float(0)),
#                c_ptr,
#                byref(c_int(m)))
#     print 'kernel time:', time.time() - start
    
#     print 'c before:'
#     print c
    
#     if minimize:
#         c = max_elt - c
#         c[c==max_elt] = 0

#     return c


def split_indices_chunk(n, k):
    from math import ceil

    try:
        iter(n)
        n = len(n)
    except TypeError, te:
        assert isinstance(n, int) or isinstance(n, long)

    return [(k*i, min(k*(i+1), n)) for i in range(int(ceil(float(n) / k)))]

def blockwise_dot(x, y,
                  x_row_max, y_col_max,
                  dot_op=np.dot,
                  dense_op=True,
                  chunk=None,
                  nnz_rows=False,
                  x_order='C', y_order='F',
                  verbose=True):
    
    import time, sys
    total_time = time.time()
    dot_time = 0
    convert_time = 0
    
    x_rows = split_indices_chunk(x.shape[0], x_row_max)
    y_cols = split_indices_chunk(y.shape[1], y_col_max)
    
    if verbose:
        print 'x_rows:', x_rows
        print 'y_cols:', y_cols
        print x.shape, len(x_rows)
        print y.shape, len(y_cols)

    assert x.shape[1] == y.shape[0]
    assert x.dtype == np.float64 or x.dtype==np.float32
    assert x.dtype == y.dtype
    
    prod = np.zeros((x.shape[0], y.shape[1]), x.dtype)
    
    is_sparse = scipy.sparse.issparse(x)
    assert is_sparse == scipy.sparse.issparse(y)
    
    for x_a, x_b in x_rows:            
        if verbose: print x_a, x_b, round(total_time - time.time(), 3)
        sys.stdout.flush()

        x_tmp = x[x_a:x_b, :]

        start = time.time()
        if nnz_rows:
            nonzero_rows = np.unique(x.nonzero()[1])            
            x_tmp = x_tmp[:, nonzero_rows]
        if is_sparse and dense_op:
            x_tmp = x_tmp.toarray(order='C')
        convert_time += time.time() - start
        
        if x_order=='F':
            x_tmp = np.asfortranarray(x_tmp)
        else:
            x_tmp = np.ascontiguousarray(x_tmp)
#         print x_tmp.flags['F_CONTIGUOUS']
#         0 / asdf        
        
        for y_c, y_d in y_cols:                

            start = time.time()
            if nnz_rows:  y_tmp = y[nonzero_rows, y_c:y_d]
            else:         y_tmp = y[:, y_c:y_d]
            if is_sparse and dense_op:
                y_tmp = y_tmp.toarray(order='F')
            convert_time += time.time() - start

            start = time.time()
            if chunk is not None and chunk < x_tmp.shape[1]:
                split_at = [a[1] for a in split_indices_chunk(x_tmp.shape[1], chunk)[:-1]]
                chunk_iter = zip(np.split(x_tmp, split_at, axis=1),
                                 np.split(y_tmp, split_at, axis=0))
            else:
                chunk_iter = [(x_tmp, y_tmp)]
            
            tmp2 = np.zeros((x_tmp.shape[0], y_tmp.shape[1]), dtype=prod.dtype)
            
            for x_chunk, y_chunk in chunk_iter:
                if dot_op=='scipy.sparse.dot':
                    tmp2 += x_chunk.dot(y_chunk)
                else:
#                     print x_a, x_b, y_c, y_d
#                     print x_chunk.shape, y_chunk.shape
#                     print x_tmp.shape, y_tmp.shape

#                     tmp2 += dot_op(x_chunk, y_chunk)
                    tmp2 = dot_op(x_chunk, y_chunk)
                    assert np.allclose(tmp2, mat_maxplus(x_chunk, y_chunk), rtol=1e-3)
            dot_time += time.time() - start
        
            prod[x_a:x_b, y_c:y_d] = tmp2
                
    print 'Total dot time:', dot_time
    print 'Total convert time:', convert_time
    print 'Total time:', time.time() - total_time
    return prod


# In[6]:

# @guvectorize(['void(float32[:], float32[:,:], float32[:])'], '(n),(n,p)->(p)', nopython=True, target='parallel')
# def mat_minplus_nnz(A, B, C):
#     n, p = B.shape
#     for j in range(p):
#         C[j] = 1000
#         for k in range(n):
#             is_set = (A[k]>0) != (B[k, j]>0)
#             C[j] = C[j] * (not is_set) + min((A[k] + B[k, j]), C[j]) * is_set
#         if not is_set:
#             C[j] = 0
            
@guvectorize(['void(float32[:], float32[:,:], float32[:])'], '(n),(n,p)->(p)', nopython=True, target='parallel')
def mat_minplus_nnz(A, B, C):
    n, p = B.shape
    for j in range(p):
        is_set = False
        C[j] = 1000
        for k in range(n):
            if (A[k]>0) != (B[k, j]>0):
                C[j] = min(A[k] + B[k, j], C[j])
                is_set = True
        if not is_set:
            C[j] = 0
            
@guvectorize(['void(float32[:], float32[:,:], float32[:])'], '(n),(n,p)->(p)', nopython=True, target='parallel')
def mat_minplus_nnz_abs(A, B, C):
    n, p = B.shape
    for j in range(p):
        is_set = False
        C[j] = 1000.
        for k in range(n):
            if (A[k]>0) != (B[k,j]>0):
                C[j] = min(np.abs(A[k]) + np.abs(B[k, j]), C[j])
                is_set = True
        if not is_set:
            C[j] = 0
            
@guvectorize(['void(float32[:], float32[:,:], float32[:])'], '(n),(n,p)->(p)', nopython=True, target='parallel')
def mat_maxplus_nnz(A, B, C):
    n, p = B.shape
    for j in range(p):
        is_set = False
        C[j] = -1000
        for k in range(n):
            if (A[k]==0) != (B[k, j]==0):
                C[j] = max(A[k] + B[k, j], C[j])
                is_set = True
        if not is_set:
            C[j] = 0


# In[15]:

max_elt = 10000.
@vectorize([float32(float32)], nopython=True, target='parallel')
def pre_normalize(x):
    x_mask = x <= 0
#     x = max_elt - np.abs(x)
#     x[x_mask] *= -1
    return x_mask * (-1.0 * max_elt - x) + (not x_mask) * (max_elt - x)

@vectorize([float32(float32)], nopython=True, target='parallel')
def post_normalize(c):
    return (c!=0) * (2*max_elt - c)
#     c[c!=0] = 2*max_elt - c[c!=0]
#     return c


# In[16]:

def sgemm_openblas(x, y, trans_x=False, n_threads=None, minimize=False):
#     assert np.isfortran(x) and np.isfortran(y)
#     assert x.flags['F_CONTIGUOUS'] and y.flags['F_CONTIGUOUS']
    
    m, k_x = x.shape
    m, k_x = (k_x, m) if trans_x else (m, k_x)
#     m = x.shape[1] if trans_x else x.shape[0]
    
    k, n = y.shape
#     assert k==y.shape[0]
    assert x.dtype==np.float32 and y.dtype==np.float32

    start = time.time()
    x = pre_normalize(x)
    y = pre_normalize(y)

#     x_mask = x<0
#     y_mask = y<0
    
#     if minimize:
# #         max_elt = np.max([np.max(x), np.max(y)])
# #         max_elt += 1.
#         max_elt = 10000.
#         x = max_elt - np.abs(x)
#         y = max_elt - np.abs(y)

#     x[x_mask] *= -1
#     y[y_mask] *= -1

    print 'Prenormalize time:', time.time() - start
    
#     print 'x:'
#     print x[0,:]
#     print 'y:'
#     print y[:,0]
    
#     print 'max:'
#     i, j = 0, 0
#     tmp = (x[i,:]+y[:,j]) * np.int64((x[i,:]>0) != (y[:,j]>0))
#     print tmp
#     if (tmp!=0).sum()>0:
#         print 'max elt:', tmp[tmp != 0].max()
#     else:
#         print "all 0's"
        
    if n_threads is not None:
        lib.goto_set_num_threads(c_int(n_threads))
        lib.openblas_set_num_threads(c_int(n_threads))
    
    x_ptr = x.ctypes.data_as(POINTER(c_float))
    y_ptr = y.ctypes.data_as(POINTER(c_float))
    c = np.empty((m,n), dtype=np.float32, order='F')
    c_ptr = c.ctypes.data_as(POINTER(c_float))

    start = time.time()
    lib.sgemm_(byref(c_char('T')) if trans_x else byref(c_char('N')),
               byref(c_char('N')),
               byref(c_int(m)),
               byref(c_int(n)),
               byref(c_int(k)),
               byref(c_float(1)),
               x_ptr,
               byref(c_int(m)),
               y_ptr,
               byref(c_int(k)),
               byref(c_float(0)),
               c_ptr,
               byref(c_int(m)))
    print 'kernel time:', time.time() - start
    
#     print 'c before:'
#     print c
    
    start = time.time()
    c = post_normalize(c)
    
#     if minimize:
#         c[c!=0] = 2*max_elt - c[c!=0]

    print 'Postnormalize time:', time.time() - start
    
    return c


@guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'], '(m,n),(n,p)->(m,p)', nopython=True)
def mat_minplus_nnz_4(A, B, C):
#     C += np.dot(A, B)
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            tmp = 0
            for k in range(n):
                tmp += A[i, k] * B[k, j]
            C[i,j] += tmp
#     for j in range(0, n, third_block):
#         mat_minplus_nnz_4(A, B[:,j*third_block:(j+1)*third_block], C)

# @guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'], '(m,n),(n,p)->(m,p)', nopython=True)
# def mat_minplus_nnz_3(A, B, C):
#     m, n = A.shape
#     n, p = B.shape
#     for j in range(0, n, third_block):
#         mat_minplus_nnz_4(A, B[:,j*third_block:(j+1)*third_block], C)
        


# @guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'], '(m,n),(n,p)->(m,p)', nopython=True)
# def mat_minplus_nnz_2(A, B, C):
#     m, n = A.shape
#     n, p = B.shape
#     for j in range(0, m, second_block):
#         0
# #         mat_minplus_nnz_3(A[j*second_block:(j+1)*second_block, :], B, C)

# @guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'], '(m,n),(n,p)->(m,p)', nopython=True)
# def mat_minplus_nnz(A, B, C):
#     m, n = A.shape
#     n, p = B.shape
#     for k in range(0, p, first_block):
#         mat_minplus_nnz_2(A[:,k*first_block:(k+1)*first_block], B[k*first_block:(k+1)*first_block,:], C)
                        
# #     for j in range(p):
# #         C[j] = 1000
# #         for k in range(n):
# #             is_set = (A[k]>0) != (B[k, j]>0)
# #             C[j] = C[j] * (not is_set) + min((A[k] + B[k, j]), C[j]) * is_set
# #         if not is_set:
# #             C[j] = 0


# In[3]:

# %time c = sgemm_openblas2(x,y,minimize=True)
    # # %time c = blockwise_dot(x, y, x.shape[0], 384, sgemm_openblas, x_order='F', verbose=False)
    # # %time c = blockwise_dot(x, y, x.shape[0], y.shape[1], sgemm_openblas, x_order='F', verbose=True)
    
    # %time ref_dot = mat_maxplus(x,y)
    # %time ref_dot = mat_maxplus(x,y)
    # %time ref_dot = mat_maxplus(2000 - x, 2000 - y)
    # # %time ref_dot = mat_maxmult(x,y)


# @jit(nopython=True)
def sgemm_openblas2(x,y,minimize=True):
    chunk = 384
    if minimize:
        max_elt = 2000.
        x = max_elt - x
        y = max_elt - y
    x_blocks = np.split(x, [a[1] for a in split_indices_chunk(x.shape[1], chunk)[:-1]], axis=1)
    y_blocks = np.split(y, [a[1] for a in split_indices_chunk(y.shape[0], chunk)[:-1]], axis=0)
    y_blocks = [np.asfortranarray(a) for a in y_blocks]    
    start = time.time()
    res_list = [sgemm_openblas(a,b,minimize=False) for a, b in zip(x_blocks, y_blocks)]
    print 'subroutine time:', time.time() - start
    start = time.time()
    c = reduce(np.maximum, res_list)
    if minimize:
        c = -1 * (c - 2 * max_elt)
    print 'reduce:', time.time() - start
    return c



# In[17]:

@jit
def fast_min_scalar(X, Y):
    return min(X, Y)

@vectorize([float32(float32,float32)])
def fast_min_vec(X, Y):
#     return min(X, Y)
    return fast_min_scalar(X, Y)

@guvectorize([(float32[:], float32[:], float32[:])], '(n),()->(n)')
def fast_min_guvec(X, Y, Z):
    Z = np.minimum(X, Y)

@jit
def fast_min(X, Y):
    minim = np.empty(X.shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            minim[i,j] = min(X[i,j], Y[i,j])
    return minim


