import time, os
from collections import Counter

import numpy as np
import pandas as pd
import scipy, scipy.sparse
from scipy.sparse import isspmatrix_csc, isspmatrix_csr, issparse, isspmatrix, csc_matrix, csr_matrix, coo_matrix

from numba import jit

import clixov_utils
import clique_maximal
from clixov_utils import *
from clique_atomic import *
from constants import cache
from degeneracy import get_degeneracy
from color import color_nodes

@jit(nopython=True, cache=cache)
def MC(Rbuf, RE, PX, pos,
       GS, GE, GI,
       Fbuf, Tbuf, depth,
       max_cover,
       C, CP, CN, core_nums, core_nums2,
       tree, offset=0, verbose=True):
    
    tree, curr_tree = update_tree_size(tree, depth, max_cover)
    sep = PX.size
    old_sep = sep        
    
    for v in range(PX.size):
#         if verbose:  print '-----------------------\nTesting v:', v
        
        if (core_nums[v] + 1) >= (max_cover + offset):
            new_sep, P = update_P(GI, GS, GE, PX, old_sep, sep, pos, v)
            
            if P.size > 0:                
#                 print 'core_nums[P].max()', core_nums[P].max()
                if (2 + core_nums[P].max()) >= (max_cover + offset):
                    GE_prev, P_copy = GE[P], P.copy()
                    
                    # Push down GI
                    Fbuf[P] = True
                    for u in P:
                        u_degree, GE[u] = move_PX_fast_bool(GI, GS, GE, Fbuf, u)
                    Fbuf[P] = False

                    colors, branches = color_nodes(GI, GS, GE, np.sort(P.copy())[::-1])
                    if (1 + colors.max()) >= (max_cover + offset):
                        tree = update_tree_size_branch(tree, curr_tree, v, 1 + colors.max(), new_sep)
                        Rbuf[RE] = v            
                        
#                         if verbose:  print 'Branching at v:', v, 'depth:', depth
                        C, CP, CN, tree, sub_cover = MC_branch(
                            Rbuf, RE + 1, PX, new_sep, pos,
                            GS, GE, GI,
                            Fbuf, Tbuf, depth + 1,
                            max_cover,
                            C, CP, CN, core_nums, core_nums2,
                            tree, offset=offset, verbose=verbose)

                        max_cover = max(max_cover, sub_cover)
                    
                    GE[P_copy] = GE_prev
            else:                
                tree = update_tree_size_branch(tree, curr_tree, v, 1, new_sep)
                tree[0, 0] += 1
                Rbuf[0] = v
                C, CP, CN = update_cliques(C, CP, CN, Rbuf[:1])
                max_cover = 1

        # Remove v from P
        sep -= 1
        swap_pos(PX, pos, v, sep)

    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN, tree, max_cover

@jit(nopython=True, cache=cache)
def MC_branch(Rbuf, RE, PX, sep, pos,
              GS, GE, GI,
              Fbuf, Tbuf, depth,
              max_cover,
              C, CP, CN, core_nums, core_nums2,
              tree, offset=0, verbose=True):
    P = PX[ : sep]

    tree, curr_tree = update_tree_size(tree, depth, max_cover)
    
#     if verbose:
#         indent = '\t' * depth
#         print indent, '---------MC_branch------'
#         print indent, 'DEPTH:', depth, 'max_cover:', max_cover
#         print indent, 'R:', Rbuf[:RE]
#         print indent, 'P:', P

    # Check bound on size of P
    if (RE + P.size) < (max_cover + offset):
        tree[6, curr_tree] = 0
        return C, CP, CN, tree, 0
    elif P.size==0:
        tree[6, curr_tree] = 1
        C, CP, CN = update_cliques(C, CP, CN, Rbuf[:RE])
        return C, CP, CN, tree, RE
        
    # Push down GI
    max_degree = 0
    Fbuf[P] = True
    for u in P:
        u_degree, GE[u] = move_PX_fast_bool(GI, GS, GE, Fbuf, u)
        max_degree = max(u_degree, max_degree)
    Fbuf[P] = False
    
    # Check bound on max degree
    if (RE + max_degree + 1) < (max_cover + offset):
        tree[6, curr_tree] = 2
        return C, CP, CN, tree, 0
    
    color_order = np.sort(P.copy())[::-1]
    
#     if depth==1:
#         color_order, core_num_vals = get_degeneracy(GI, GS, GE, P)
# #           # If change core_num here, then must change core_num_bound at reduce_G
# #         core_num = np.empty(PX.size, np.int32)
# #         core_num[color_order] = core_num_vals
# #         color_order = color_order[::-1]
#     else:
#         color_order = np.sort(P.copy())[::-1]
            
    ## Color
    colors, branches = color_nodes(GI, GS, GE, color_order)
    old_sep = sep
    
    for v_i in range(branches.size -1, -1, -1):
        v = branches[v_i]
        if (RE + colors[v_i]) >= (max_cover + offset):
    #         if verbose:  print indent, 'Branching at v:', v, 'depth:', depth

            Rbuf[RE] = v
            new_sep, P = update_P(GI, GS, GE, PX, old_sep, sep, pos, v)
            GE_prev, P_copy = GE[P], P.copy()

            tree = update_tree_size_branch(tree, curr_tree, v, RE + colors[v_i], new_sep)

            C, CP, CN, tree, sub_cover = MC_branch(
                Rbuf, RE + 1, PX, new_sep, pos,
                GS, GE, GI,
                Fbuf, Tbuf, depth + 1,
                max_cover,
                C, CP, CN, core_nums, core_nums2,
                tree, offset=offset, verbose=verbose)
            GE[P_copy] = GE_prev

            if sub_cover > max_cover:
                max_cover = sub_cover
                sep, P = reduce_G(GI, GS, GE, Fbuf, core_nums2, max_cover + offset - 1, PX, pos, sep)

        # Remove v from P
        sep -= 1
        swap_pos(PX, pos, v, sep)
        
    tree[6, curr_tree] = 5
    return C, CP, CN, tree, max_cover

def MC_py(G, max_cover=0, offset=0, verbose=False):
    
    ## Degeneracy
    GS, GE, GI = G.indptr[:-1].copy(), G.indptr[1:].copy(), G.indices.copy()
    degen_order, core_num = get_degeneracy(GI, GS, GE,
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
    GS, GE, GI = G.indptr[:-1].copy(), G.indptr[1:].copy(), G.indices.copy()
    k = G.shape[0]

    if verbose: print_sparse(G)
        
    core_num = core_num[np.argsort(degen_order)][keep]

    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k)
        
    tree = np.asfortranarray(np.zeros((14, 100000), np.int32))
    tree.fill(-1)
    tree[:, :2] = np.array([0,0])

    core_num2 = core_num.copy()
    for i in range(1, core_num2.size):
        core_num2[i] = max(core_num2[i], core_num2[i-1])
    if verbose:
        print 'core_num:', core_num
        print 'core_num2:', core_num2
    
    start_time = time.time()    
    C, CP, CN, tree, max_cover = MC(
        R, RE, PX, pos,
        GS, GE, GI,
        Fbuf, Tbuf, 0,
        max_cover,
        C, CP, CN, core_num, core_num2,
        tree, offset=offset, verbose=verbose)
    
    if verbose:
        print 'Time:', time.time() - start_time
    
    tree = tree[:,:tree[0,0]+1]
    
    C = keep[C]
    
    if verbose: print 'tree:', tree[0, :3]
    if verbose: print 'Clique sizes (%s):' % CN, Counter(CP[1:] - CP[:-1])

    tmp_cliques = [tuple(sorted(C[CP[i]:CP[i+1]])) for i in range(CN)]        
    max_size = max([len(x) for x in tmp_cliques])
    if verbose: print 'Cliques:', tmp_cliques
    cliques = tuples_to_csc(tmp_cliques, degen_order.size)
    if verbose: print 'Found cliques:', cliques.shape[1], as_dense_flat(cliques.sum(0))
    cliques, max_clique = get_largest_cliques(cliques)
    if verbose: print 'After filtering for largest cliques:', cliques.shape[1]
    
    return cliques, keep, tree
