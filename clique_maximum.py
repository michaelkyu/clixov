import cPickle, time, argparse, os, pickle, datetime, gzip
import subprocess, StringIO, random, sys, tempfile, shutil, igraph, multiprocessing, glob
from itertools import combinations, chain, groupby, compress, permutations, product
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import scipy, scipy.sparse
from scipy.sparse import isspmatrix_csc, isspmatrix_csr, issparse, isspmatrix, csc_matrix, csr_matrix, coo_matrix, _sparsetools

from ctypes import pointer, Structure, POINTER,c_void_p,c_int,c_char,c_double,byref,cdll, c_long, c_float, c_int64
import ctypes

import numba
from numba import int64, float32, float64, boolean, guvectorize, jit, vectorize

import clixov_utils
import clique_maximal
from clixov_utils import *
from clique_atomic import *
from constants import cache
from degeneracy import get_degeneracy
from color import color_nodes

@jit(nopython=True, cache=cache)
def MC(R_buff, R_end, PX, pos,
       G_start, G_end, G_indices,
       PXbuf, PXbuf2, depth,
       max_cover,
       cliques, cliques_indptr, cliques_n, core_nums, core_nums2,
       tree_size, offset=0, verbose=True):
    
    tree_size, curr_tree_size = update_tree_size(tree_size, depth, max_cover)
    sep = PX.size
    old_sep = sep        
    
    for v in range(PX.size):
#         if verbose:  print '-----------------------\nTesting v:', v
        
        if (core_nums[v] + 1) >= (max_cover + offset):
            new_sep, P = update_P(G_indices, G_start, G_end, PX, old_sep, sep, pos, v)
#             print 'updated P:', P
            
            if P.size > 0:                
#                 print 'core_nums[P].max()', core_nums[P].max()
                if (2 + core_nums[P].max()) >= (max_cover + offset):
                    G_end_prev, P_copy = G_end[P], P.copy()
                    
                    # Push down G_indices
                    PXbuf[P] = True
                    for u in P:
                        u_degree, G_end[u] = move_PX_fast_bool(G_indices, G_start, G_end, PXbuf, u)
                    PXbuf[P] = False

                    colors, branches = color_nodes(G_indices, G_start, G_end, np.sort(P.copy())[::-1])
                    if (1 + colors.max()) >= (max_cover + offset):
                        tree_size = update_tree_size_branch(tree_size, curr_tree_size, v, 1 + colors.max(), new_sep)
                        R_buff[R_end] = v            
                        
#                         if verbose:  print 'Branching at v:', v, 'depth:', depth
                        cliques, cliques_indptr, cliques_n, tree_size, sub_cover = MC_Branch(
                            R_buff, R_end + 1, PX, new_sep, pos,
                            G_start, G_end, G_indices,
                            PXbuf, PXbuf2, depth + 1,
                            max_cover,
                            cliques, cliques_indptr, cliques_n, core_nums, core_nums2,
                            tree_size, offset=offset, verbose=verbose)

                        max_cover = max(max_cover, sub_cover)
                    
                    G_end[P_copy] = G_end_prev
            else:                
                tree_size = update_tree_size_branch(tree_size, curr_tree_size, v, 1, new_sep)
                tree_size[0, 0] += 1
                R_buff[0] = v
                cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R_buff[:1])
                max_cover = 1

        # Remove v from P
        sep -= 1
        swap_pos(PX, pos, v, sep)

    cliques, cliques_indptr, cliques_n = trim_cliques(cliques, cliques_indptr, cliques_n)
    return cliques, cliques_indptr, cliques_n, tree_size, max_cover

@jit(nopython=True, cache=cache)
def MC_Branch(R_buff, R_end, PX, sep, pos,
              G_start, G_end, G_indices,
              PXbuf, PXbuf2, depth,
              max_cover,
              cliques, cliques_indptr, cliques_n, core_nums, core_nums2,
              tree_size, offset=0, verbose=True):
    P = PX[ : sep]

    tree_size, curr_tree_size = update_tree_size(tree_size, depth, max_cover)
    
#     if verbose:
#         indent = '\t' * depth
#         print indent, '---------MC_branch------'
#         print indent, 'DEPTH:', depth, 'max_cover:', max_cover
#         print indent, 'R:', R_buff[:R_end]
#         print indent, 'P:', P

    # Check bound on size of P
    if (R_end + P.size) < (max_cover + offset):
        tree_size[6, curr_tree_size] = 0
        return cliques, cliques_indptr, cliques_n, tree_size, 0
    elif P.size==0:
        tree_size[6, curr_tree_size] = 1
        cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R_buff[:R_end])
        return cliques, cliques_indptr, cliques_n, tree_size, R_end
        
    # Push down G_indices
    max_degree = 0
    PXbuf[P] = True
    for u in P:
        u_degree, G_end[u] = move_PX_fast_bool(G_indices, G_start, G_end, PXbuf, u)
        max_degree = max(u_degree, max_degree)
    PXbuf[P] = False
    
    # Check bound on max degree
    if (R_end + max_degree + 1) < (max_cover + offset):
        tree_size[6, curr_tree_size] = 2
        return cliques, cliques_indptr, cliques_n, tree_size, 0
    
    color_order = np.sort(P.copy())[::-1]
    
#     if depth==1:
#         color_order, core_num_vals = get_degeneracy(G_indices, G_start, G_end, P)
# #           # If change core_num here, then must change core_num_bound at reduce_G
# #         core_num = np.empty(PX.size, np.int32)
# #         core_num[color_order] = core_num_vals
# #         color_order = color_order[::-1]
#     else:
#         color_order = np.sort(P.copy())[::-1]
            
    ## Color
    colors, branches = color_nodes(G_indices, G_start, G_end, color_order)
    old_sep = sep
    
    for v_i in range(branches.size -1, -1, -1):
        v = branches[v_i]
        if (R_end + colors[v_i]) >= (max_cover + offset):
    #         if verbose:  print indent, 'Branching at v:', v, 'depth:', depth

            R_buff[R_end] = v
            new_sep, P = update_P(G_indices, G_start, G_end, PX, old_sep, sep, pos, v)
            G_end_prev, P_copy = G_end[P], P.copy()

            tree_size = update_tree_size_branch(tree_size, curr_tree_size, v, R_end + colors[v_i], new_sep)

            cliques, cliques_indptr, cliques_n, tree_size, sub_cover = MC_Branch(
                R_buff, R_end + 1, PX, new_sep, pos,
                G_start, G_end, G_indices,
                PXbuf, PXbuf2, depth + 1,
                max_cover,
                cliques, cliques_indptr, cliques_n, core_nums, core_nums2,
                tree_size, offset=offset, verbose=verbose)
            G_end[P_copy] = G_end_prev

            if sub_cover > max_cover:
                max_cover = sub_cover
                sep, P = reduce_G(G_indices, G_start, G_end, PXbuf, core_nums2, max_cover + offset - 1, PX, pos, sep)

        # Remove v from P
        sep -= 1
        swap_pos(PX, pos, v, sep)
        
    tree_size[6, curr_tree_size] = 5
    return cliques, cliques_indptr, cliques_n, tree_size, max_cover

def MC_py(G, max_cover=0, offset=0, verbose=False):
    
    ## Degeneracy
    G_start, G_end, G_indices = G.indptr[:-1].copy(), G.indptr[1:].copy(), G.indices.copy()
    degen_order, core_num = get_degeneracy(G_indices, G_start, G_end,
                                           np.arange(G.shape[0]).astype(np.int32))
    if verbose:
        print 'degen_order/core_num:', zip(degen_order, core_num)
    
    ## Identify nodes with insufficient core number
    sep = ((core_num + 1) >= max_cover).nonzero()[0]
    if sep.size > 0:
        sep = sep[0]
        keep = degen_order[sep : ]
        if verbose:
            print 'keep:', keep
            print 'Reduced_size:', keep.size
    else:
        keep = degen_order    
    
    ## Remove nodes
    G = G[:,keep][keep,:]
    G_start, G_end, G_indices = G.indptr[:-1].copy(), G.indptr[1:].copy(), G.indices.copy()
    k = G.shape[0]
    PX = np.arange(k).astype(np.int32)

    if verbose:
        print_sparse(G)
        
    core_num = core_num[np.argsort(degen_order)][keep]

    ## Initialize data structures
    pos = np.arange(PX.size)
    R_buff = np.empty(PX.size, np.int32)
    R_end = 0
    PXbuf = np.zeros(PX.size, np.bool)
    PXbuf2 = np.ones(PX.size, np.bool)
    cliques, cliques_indptr, cliques_n = np.empty(PX.size, np.int32), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0
    depth = 0
                       
    tree_size = np.asfortranarray(np.zeros((14, 100000), np.int32))
    tree_size.fill(-1)
    tree_size[:, :2] = np.array([0,0])

    core_num2 = core_num.copy()
    for i in range(1, core_num2.size):
        core_num2[i] = max(core_num2[i], core_num2[i-1])
    if verbose:
        print 'core_num:', core_num
        print 'core_num2:', core_num2
    
    start_time = time.time()    
    cliques, cliques_indptr, cliques_n, tree_size, max_cover = MC(
        R_buff, R_end, PX, pos,
        G_start, G_end, G_indices,
        PXbuf, PXbuf2, depth,
        max_cover,
        cliques, cliques_indptr, cliques_n, core_num, core_num2,
        tree_size, offset=offset, verbose=verbose)
    
    if verbose:
        print 'Time:', time.time() - start_time
    
    tree_size = tree_size[:,:tree_size[0,0]+1]
    
    cliques = keep[cliques]
    
    if verbose: print 'tree_size:', tree_size[0, :3]
    if verbose: print 'Clique sizes (%s):' % cliques_n, Counter(cliques_indptr[1:] - cliques_indptr[:-1])
    if verbose: print 'Cliques indices:', cliques.size

    tmp_cliques = [tuple(sorted(cliques[cliques_indptr[i]:cliques_indptr[i+1]])) for i in range(cliques_n)]        
    max_size = max([len(x) for x in tmp_cliques])
    if verbose: print 'Cliques:', tmp_cliques
    cliques = tuples_to_csc(tmp_cliques, degen_order.size)
    if verbose: print 'Found cliques:', cliques.shape[1], as_dense_flat(cliques.sum(0))
    cliques, max_clique = get_largest_cliques(cliques)
    if verbose: print 'After filtering for largest cliques:', cliques.shape[1]
    
    return cliques, keep, tree_size
