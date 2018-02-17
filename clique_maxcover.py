import cPickle, time
import argparse, os
import pickle, datetime
import gzip, subprocess
import StringIO, random
import sys, tempfile
import shutil, multiprocessing
import glob, sys
from itertools import combinations, chain, groupby, compress, permutations, product
import itertools
from collections import Counter

import numpy as np
import pandas as pd

import igraph
import scipy, scipy.sparse
from scipy.sparse import isspmatrix_csc, isspmatrix_csr, issparse, isspmatrix, csc_matrix, csr_matrix, coo_matrix

from ctypes import pointer, Structure, POINTER,c_void_p,c_int,c_char,c_double,byref,cdll, c_long, c_float, c_int64
import ctypes

import clixov_utils
from clixov_utils import trim_cliques
import clique_maximal
from clique_atomic import *
from color import set_color, get_branch_sizes, get_branch_sizes_vw

from numba import int64, float32, float64, boolean, guvectorize, jit, vectorize

verbose = False

@jit(nopython=True)
def expand_2d_arr(arr):
    arr2 = np.zeros((2*arr.shape[1], arr.shape[0]), arr.dtype).T
    arr2[:,:arr.shape[1]] = arr
    return arr2

def BKPivotSparse2_Gnew_cover_wrapper(Gold, Gnew, PX=None, degeneracy=None, max_branch_depth=100000):
    if PX is None:
        k = Gold.shape[0]
        PX = np.arange(k).astype(np.int32)
        pos = np.empty(PX.size, np.int32)
        pos[PX] = np.arange(PX.size)
    else:
        k = Gold.shape[0]
        pos = np.empty(k, np.int32)
        pos[:] = -1
        pos[PX] = np.arange(PX.size)
    R = np.zeros(PX.size, np.int32)
    R_end = np.int32(0)    
    sep = PX.size
    PXbuf = np.zeros(k, np.bool_)
    PXbuf2 = np.ones(k, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])
    between_new = np.zeros(PX.size, np.bool_)
    between_stack = np.arange(PX.size).astype(np.int32)
    between_end = 0
    
    cliques, cliques_indptr, cliques_n = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0
   
    Gnew_indices = np.empty(2 * Gnew.indices.size, np.int32)    
    Gnew_indices[1::2] = 0
    
    if degeneracy in ['min', 'max']:
        assert k == Gnew.shape[0]
#         G = Gold + Gnew
        if degeneracy=='max':
            degen_order, degen_deg = get_degeneracy_max(Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices)
        if degeneracy=='min':
#             degen_order, degen_deg = get_degeneracy_min(G.indptr[:-1], G.indptr[1:], G.indices)
            degen_order, degen_deg = get_degeneracy_min(Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices)
        degen_pos = np.empty(k, np.int32)
        degen_pos[degen_order] = np.arange(k).astype(np.int32)
        Gnew_indices[::2] = degen_pos[Gnew.indices]
        Gnew_start = 2 * Gnew.indptr[:-1][degen_order]
        Gnew_end = 2 * Gnew.indptr[1:][degen_order]
        
        Gold_indices = degen_pos[Gold.indices]
        Gold_start, Gold_end = Gold.indptr[:-1][degen_order], Gold.indptr[1:][degen_order]
    else:
        Gnew_indices[::2] = Gnew.indices
        Gnew_start = 2 * Gnew.indptr[:-1]
        Gnew_end = 2 * Gnew.indptr[1:]        
        Gold_start, Gold_end, Gold_indices = Gold.indptr[:-1], Gold.indptr[1:], Gold.indices
    
    pot_indices = np.empty(3 * Gnew.indices.size, np.int32)
    pot_indices[:] = 0
    pot_start = Gnew_start.copy() * 3 / 2
    pot_end = Gnew_start.copy() * 3 / 2
        
    pot_min = np.zeros(PX.size, np.int32)
    pot_min.fill(1000000)
    
    min_cover, max_cover, min_cover_between = 0, 0, 0
    
    max_possible = PX.size
    
    tree_size = np.asfortranarray(np.zeros((22, 100000), np.int32))
    tree_size.fill(-1)
    tree_size[0, :2] = np.array([0,0])
    
    cliques, cliques_indptr, cliques_n, max_cover, tree_size = BKPivotSparse2_Gnew_cover(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        Gold_start, Gold_end, Gold_indices,
        Gnew_start, Gnew_end, Gnew_indices,
        PXbuf2, 0, between_new, between_stack, between_end,
        pot_indices, pot_start, pot_end, pot_min, max_possible,
        min_cover, max_cover, min_cover_between, max_branch_depth,
        cliques, cliques_indptr, cliques_n, tree_size)

    cliques, cliques_indptr, cliques_n = trim_cliques(cliques, cliques_indptr, cliques_n)
    
    if degeneracy in ['min', 'max']:
        cliques = degen_order[cliques]
        
    tree_size = tree_size[:, : 4 + tree_size[0,0]]

    return cliques, cliques_indptr, cliques_n, tree_size

@jit(nopython=True, cache=cache)
def BKPivotSparse2_Gsep_cover(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                              G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
                              PXbuf, depth,
                              pot_indices, pot_start, pot_end, pot_min, prev_max_possible,
                              min_cover, max_cover, min_cover_between, max_branch_depth,
                              cliques, cliques_indptr, cliques_n, tree_size):

    orig_size = 4 + tree_size[0, 0]
    curr_tree_size = 4 + tree_size[0, 0]    
    if tree_size.shape[1] <= curr_tree_size:
        tree_size = expand_2d_arr(tree_size)    
    tree_size[0, 0] += 1
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]

    # if verbose:
    #     indent = '\t' * depth
    #     print indent, '---------Gsep------'
    #     print indent, 'DEPTH:', depth
    #     print indent, 'PX:', PX
    #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    #     print indent, 'R:', R, 'P:', P, 'X:', X
    #     print indent, 'min_cover:', min_cover, 'min_cover_between:', min_cover_between, 'prev_max_possible:', prev_max_possible
    #     print indent, 'pot_start/end/indices:', [(i,pot_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(pot_start, pot_end))]
    #     print indent, 'Gold:', [(i, x, y, G_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(G_start, G_end))]
    #     print indent, 'Gnew:', [(i, x, y, Gnew_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(Gnew_start, Gnew_end))]
        
    if P.size==0:
        tree_size[0, 1] += 1
        if X.size==0:
            tree_size[12, curr_tree_size] = 1
            R_size = R.size

            if verbose: print((1111111, R[0], curr_tree_size, depth, min_cover))
            cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R)            
            return cliques, cliques_indptr, cliques_n, R.size, tree_size        
        else:
            tree_size[12, curr_tree_size] = 2
            return cliques, cliques_indptr, cliques_n, 0, tree_size
    
    default_max = 1000000
    max_possible = R.size + sep - PS
    min_cover_within = default_max
    min_cover_within_P = np.empty(P.size, np.int32)
    min_cover_within_P[:] = default_max
    
    if depth > 0: prev_r = R[R_end - 1]
    else:         prev_r = -1
    
    u = -1
    max_degree = -1
    P_degree = np.zeros(sep - PS, np.int32)
    P_degree_new = np.zeros(sep - PS, np.int32)
    for v_i in range(XE-1, PS-1, -1):
        v = PX[v_i]
        v_degree, curr = move_PX(G_indices, G_start, G_end,
                                 pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])

        # Move P and X to the bottom
        v_degree_new, curr_new = 0, Gnew_start[v]
        for w_i in range(Gnew_start[v], Gnew_end[v], 2):
            w = Gnew_indices[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_new += w_pos < sep
                if v_i < sep and w_pos < sep:
                    min_cover_within_P[v_i - PS] = min(min_cover_within_P[v_i - PS], Gnew_indices[w_i+1])
                Gnew_indices[curr_new], Gnew_indices[w_i] = w, Gnew_indices[curr_new]
                Gnew_indices[curr_new+1], Gnew_indices[w_i+1] = Gnew_indices[w_i+1], Gnew_indices[curr_new+1]
                curr_new += 2

            # Accounts for the case when curr_new was incremented
            if (prev_max_possible >= Gnew_indices[w_i+1]) and (v_i < sep) and Gnew_indices[w_i] == prev_r:
                pot_indices[pot_end[v]-1] = w_i + 1
                
        v_degree += v_degree_new
        
        if v_degree > max_degree:
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new        
        if v_i < sep:
            min_cover_within = min(min_cover_within, min_cover_within_P[v_i - PS])
            P_degree[v_i - PS] = v_degree
            P_degree_new[v_i - PS] = v_degree_new
    
    if min(prev_max_possible, R.size + 1 + P_degree.max()) < min(min_cover, min_cover_between, min_cover_within):
        return cliques, cliques_indptr, cliques_n, 0, tree_size    
    
    branches = P.copy()

    # Color the nodes
    colors = np.zeros(sep - PS, np.int32)
    nei_bool = np.zeros(sep - PS + 1, np.bool_)
    nei_list = np.empty(sep - PS, np.int32)
    max_colors = 0   
    
    for v_i in np.argsort(-1 * P_degree):
        max_colors = set_color(Gnew_indices, Gnew_start, Gnew_end,
                               G_indices, G_start, G_end,
                               pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)    
        
    max_possible = R.size + max_colors    
    if max_possible < min(min_cover, min_cover_within, min_cover_between):
        return cliques, cliques_indptr, cliques_n, 0, tree_size
    
    # Get bound based on X
    tmp_sizes = get_branch_sizes(Gnew_indices, Gnew_start, Gnew_end,
                                 G_indices, G_start, G_end,
                                 pos, sep, PS, XE, colors, nei_bool, nei_list, P)
    tmp_sizes_argsort = np.argsort(-1 * tmp_sizes)    
    best = 10000
    for v in PX[sep:XE]:
        X_keep = np.ones(sep - PS, np.bool_)
        for w in G_indices[G_start[v] : G_end[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif w_pos < sep:
                X_keep[w_pos - PS] = False
        for w in Gnew_indices[Gnew_start[v] : Gnew_end[v] : 2]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif w_pos < sep:
                X_keep[w_pos - PS] = False
        for w_i in tmp_sizes_argsort:
            if X_keep[w_i]:
                best = min(best, tmp_sizes[w_i])
                break

    if R.size + 1 + best < min(min_cover, min_cover_within, min_cover_between):
        return cliques, cliques_indptr, cliques_n, 0, tree_size
    
    see_purpose = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
    if see_purpose:
        within_purpose = (R.size + 1 + tmp_sizes) >= min_cover_within_P
        between_purpose = np.zeros(P.size, np.bool_)
        tmp_between_purpose = (pot_end[P] > pot_start[P]).nonzero()[0]
        between_purpose[tmp_between_purpose[pot_indices[pot_end[P[tmp_between_purpose]] - 2] <= (R.size + 1 + tmp_sizes[tmp_between_purpose])]] = True
        P_purpose = within_purpose | between_purpose
        if not np.any(P_purpose):
            return cliques, cliques_indptr, cliques_n, 0, tree_size
        
        branches = P[P_purpose]
        branches_degree = P_degree[P_purpose]
    else:
        branches_degree = P_degree
    
#     print indent, 'branches:', branches
#     print indent, 'colors:', colors[pos[branches] - PS]
#     print indent, 'branches_sizes:', branches_sizes
#     print indent, 'min(min_cover, min_cover_within, min_cover_between):', min(min_cover, min_cover_within, min_cover_between)
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'branches:', branches
    
    # Initialize max_cover
    max_cover = 0
    init_cliques_n = cliques_n

    used = np.empty(branches.size, np.int32)
    used_count = 0
    
    colors_v = np.empty(PX.size, np.int32)
    colors_v[P] = colors
       
    v_i = -1
    while True:
        v_i += 1        
        if v_i==branches.size:
            break
        
        v = branches[v_i]
    
        min_newly_capt = 1000000
        sub_min_cover_between = 1000000
        new_PS, new_XE = sep, sep
        
        tmp = get_branch_sizes_vw(Gnew_indices, Gnew_start, Gnew_end,
                                  G_indices, G_start, G_end,
                                  pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
        sub_max_possible = R.size + 1 + tmp
        if sub_max_possible >= min(min_cover, min_cover_within, min_cover_between):
            used[used_count] = v
            used_count += 1
        else:
            continue
            
        for w_i in range(G_start[v], G_end[v]):
            w = G_indices[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)
                
                if pot_end[w] > pot_start[w]:
                    sub_min_cover_between = min(sub_min_cover_between, pot_indices[pot_end[w]-2])
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            else:
                break
                
        for w_i in range(Gnew_start[v], Gnew_end[v], 2):
            w = Gnew_indices[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)
                if sub_max_possible >= Gnew_indices[w_i+1]:
                    pot_indices[pot_end[w]] = Gnew_indices[w_i+1]
                    if pot_end[w] > pot_start[w]:
                        pot_indices[pot_end[w]+1] = min(pot_indices[pot_end[w]-2], Gnew_indices[w_i+1])
                    else:
                        pot_indices[pot_end[w]+1] = Gnew_indices[w_i+1]
                    pot_end[w] += 3
                
                if pot_end[w] > pot_start[w]:
                    sub_min_cover_between = min(sub_min_cover_between, pot_indices[pot_end[w]-2])
    
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < PS or w_pos >= XE:
                break
  
        newly_capt = pot_indices[pot_start[v] : pot_end[v]]
        if newly_capt.size > 0:
            min_newly_capt = newly_capt[newly_capt.size - 2]        

        # Not necessarily true as branches are moved to X
        if sep - new_PS == 0:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_between)
        else:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_between)
        
        if do_subbranch:
            sub_tree_size = 4 + tree_size[0,0]
            if tree_size.shape[1] <= sub_tree_size:
                tree_size = expand_2d_arr(tree_size)

        if do_subbranch:
            R_buff[R_end] = v
            sub_min_cover = min(min_cover, min_newly_capt)

            prev_cliques_n = cliques_n
            
            cliques, cliques_indptr, cliques_n, sub_max_cover, tree_size = BKPivotSparse2_Gsep_cover(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                G_start, G_end, G_indices,
                Gnew_start, Gnew_end, Gnew_indices,
                PXbuf, depth+1,
                pot_indices, pot_start, pot_end, pot_min, sub_max_possible,
                sub_min_cover, max_cover, sub_min_cover_between, max_branch_depth,
                cliques, cliques_indptr, cliques_n, tree_size)
                
            # Update min_cover, max_cover
            max_cover = max(max_cover, sub_max_cover)
            min_cover = max(min_cover, sub_max_cover)
            
            if sub_max_cover > min_newly_capt:
                for w_i in range(0, newly_capt.size, 3):
                    newly_capt[w_i] = max(newly_capt[w_i], sub_max_cover)
                    newly_capt[w_i+1] = max(newly_capt[w_i+1], sub_max_cover)

        # Restore pot[w] for all new neighbors of v
        for w_i in range(Gnew_start[v], Gnew_end[v], 2):
            w = Gnew_indices[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                if sub_max_possible >= Gnew_indices[w_i+1]:
                    if do_subbranch:
#                         assert Gnew_indices[w_i+1] == Gnew_indices[pot_indices[pot_end[w]-1]]
                        Gnew_indices[w_i+1] = max(Gnew_indices[w_i+1], pot_indices[pot_end[w]-3])
                        Gnew_indices[pot_indices[pot_end[w]-1]] = max(Gnew_indices[pot_indices[pot_end[w]-1]], pot_indices[pot_end[w]-3])
            
                    pot_end[w] -= 3
            elif w_pos < PS or w_pos >= XE:
                break

        # Recompute min_cover_within
        if do_subbranch and sub_max_cover > min_cover_within:
            min_cover_within = 1000000
            for x_i in range(sep-1, PS-1, -1):
                x = PX[x_i]
                # Move P and X to the bottom
                for w_i in range(Gnew_start[x], Gnew_end[x], 2):
                    w = Gnew_indices[w_i]
                    w_pos = pos[w]
                    if w_pos < PS or w_pos >= XE:
                        break
                    elif PS <= w_pos and w_pos < sep:
                        min_cover_within = min(min_cover_within, Gnew_indices[w_i+1])
    
        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)
        
        # Don't branch anymore after this
        if depth > max_branch_depth and used_count==1:
            break

    for v in used[:used_count][::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1
        
#     print indent, 'Returning::::'
#     print indent, 'R:', R, 'P:', P, 'X:', X
#     print indent, 'min/max_cover:', min_cover, max_cover
#     print indent, 'pot_start/end/indices:', [(i,pot_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(pot_start, pot_end))]
#     print indent, 'Gnew:', [(i, x, y, Gnew_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(Gnew_start, Gnew_end))]  
#     assert np.all(pot_end >= pot_start)

    return cliques, cliques_indptr, cliques_n, max_cover, tree_size

@jit(nopython=True, cache=cache)
def BKPivotSparse2_Gnew_cover(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                              G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
                              PXbuf, depth,
                              between_new, between_stack, between_end,
                              pot_indices, pot_start, pot_end, pot_min, prev_max_possible,
                              min_cover, max_cover, min_cover_between, max_branch_depth,
                              cliques, cliques_indptr, cliques_n, tree_size):
    """
    between_new[v]: Is there a new edge crossing between R and node v in P?
    """
    curr_tree_size = 4 + tree_size[0, 0]
    if tree_size.shape[1] <= curr_tree_size:
        tree_size = expand_2d_arr(tree_size)    
    tree_size[10, curr_tree_size] = 1
    tree_size[0, curr_tree_size] = depth
    tree_size[0, 0] += 1
    
    if verbose and (curr_tree_size % 25000) == 0:
        print((3333333, curr_tree_size))
    
    if curr_tree_size >= 90000:
        return cliques, cliques_indptr, cliques_n, max_cover, tree_size
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]

    ## When this is depth==2, considering forcing the next v's (the ones that will occupy R[2]) to have a new edge with R[1].

    # if verbose:
    #     indent = '\t' * depth
    #     print indent, '--------Gnew-------'
    #     print indent, 'DEPTH:', depth
    #     print indent, 'PX:', PX
    #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    #     print indent, 'R:', R, 'P:', P, 'X:', X
    #     print indent, 'min_cover:', min_cover, 'min_cover_between:', min_cover_between, 'prev_max_possible:', prev_max_possible
    #     print indent, 'between_new:', between_new.astype(np.int32)
    #     print indent, 'pot_min:', pot_min
    #     print indent, 'pot_start/end/indices:', [(i,pot_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(pot_start, pot_end))]
    #     print indent, 'Gold:', [(i, x, y, G_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(G_start, G_end))]
    #     print indent, 'Gnew:', [(i, x, y, Gnew_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(Gnew_start, Gnew_end))]
    
    if P.size==0:
        tree_size[12, curr_tree_size] = 2
        tree_size[0, 1] += 1
        return cliques, cliques_indptr, cliques_n, 0, tree_size

    incd = np.empty(sep - PS, np.int32)
    incd_degree = np.empty(sep - PS, np.int32)
    incd_count = 0
    X_incd = np.empty(XE - sep, np.int32)
    X_incd_count = 0
    
    default_max = 1000000
    max_possible = R.size + (sep - PS)
    min_cover_within = default_max
    min_cover_within_incd = np.empty(P.size, np.int32)
    min_cover_within_incd[:] = default_max
    
    if depth > 0: prev_r = R[R_end - 1]
    else:         prev_r = -1
    
    u = -1
    max_degree = -1
    P_degree = np.zeros(sep - PS, np.int32)
    P_degree_new = np.zeros(sep - PS, np.int32)
            
    for v_i in range(XE-1, PS-1, -1):
        v, inP = PX[v_i], v_i < sep        
        
        min_cover_within_v = default_max
        
        # Move P and X to the bottom
        v_degree_new, curr_new = 0, Gnew_start[v]
        for w_i in range(Gnew_start[v], Gnew_end[v], 2):
            w = Gnew_indices[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_new += w_pos < sep
                if inP and w_pos < sep:
                    min_cover_within_v = min(min_cover_within_v, Gnew_indices[w_i+1])
                Gnew_indices[curr_new], Gnew_indices[w_i] = w, Gnew_indices[curr_new]
                Gnew_indices[curr_new+1], Gnew_indices[w_i+1] = Gnew_indices[w_i+1], Gnew_indices[curr_new+1]
                curr_new += 2
            
            if inP and (prev_max_possible >= Gnew_indices[w_i+1]) and Gnew_indices[w_i] == prev_r:
                pot_indices[pot_end[v]-1] = w_i + 1
                
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new, u_incd = v, curr_new, incd_count
        if (not inP) and v_degree_new > 0:
            tmp, curr = move_PX(G_indices, G_start, G_end,
                                pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
        if inP and (v_degree_new > 0 or between_new[v]):
            incd[incd_count] = v
            incd_degree[incd_count] = v_degree_new            
            min_cover_within_incd[incd_count] = min_cover_within_v            
            incd_count += 1
            
        if inP:
            P_degree[v_i - PS] += v_degree_new
            P_degree_new[v_i - PS] += v_degree_new
            min_cover_within = min(min_cover_within, min_cover_within_v)            
    u_curr = G_end[u]
    
    if min(prev_max_possible, R.size + (sep - PS)) < min(min_cover, min_cover_between, min_cover_within):
        return cliques, cliques_indptr, cliques_n, 0, tree_size
    
    if incd_count == 0:
        tree_size[12, curr_tree_size] = 4
        tree_size[0, 1] += 1
        return cliques, cliques_indptr, cliques_n, 0, tree_size
    incd = incd[:incd_count]
    min_cover_within_incd = min_cover_within_incd[:incd_count]
    incd_degree = incd_degree[:incd_count]
    X_incd = X_incd[:X_incd_count]

    PXbuf[incd] = False
    PXbuf[X_incd] = False
    new_incd = np.empty(XE - PS - incd.size, PX.dtype)
    new_incd_end = 0
    
    for v in incd:
        curr = G_start[v]
        v_degree_old = 0
        
        # Move P and X to the bottom
        for w_i in range(G_start[v], G_end[v]):
            w = G_indices[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_old += w_pos < sep
                G_indices[curr], G_indices[w_i] = w, G_indices[curr]
                curr += 1
                
                if PXbuf[w]:
                    new_incd[new_incd_end] = w
                    new_incd_end += 1
                    PXbuf[w] = False        
        P_degree[pos[v] - PS] += v_degree_old

    PXbuf[incd] = True
    PXbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    PXbuf[new_incd] = True
    
    PXbuf[incd] = False
    PXbuf[new_incd] = False
    for v in new_incd:
        curr = G_start[v]
        v_degree = 0
        for w_i in range(G_start[v], G_end[v]):
            w = G_indices[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                ## Counts only those neighbors that are in P and in (incd, new_incd)
                v_degree += (w_pos < sep) and not PXbuf[w]
                G_indices[curr], G_indices[w_i] = w, G_indices[curr]
                curr += 1
        if pos[v] < sep:
            P_degree[pos[v] - PS] += v_degree
    PXbuf[incd] = True
    PXbuf[new_incd] = True
    
    #--------------------------------#
    
    # Color the nodes
    colors = np.zeros(sep - PS, np.int32)
    nei_bool = np.zeros(sep - PS + 1, np.bool_)
    nei_list = np.empty(sep - PS, np.int32)
    max_colors = 0

    PXbuf[incd] = False
    PXbuf[new_incd] = False
    
    # Recompute P_degree    
    for v_i in range(sep, PS):
        v = PX[v_i]
        v_degree = 0
        for w in Gnew_indices[Gnew_start[v] : Gnew_end[v] : 2]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            v_degree += (w_pos < sep) and (not PXbuf[w])
        for w in G_indices[G_start[v] : G_end[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            v_degree += (w_pos < sep) and (not PXbuf[w])
        P_degree[v_i] = v_degree
                
    for v_i in np.argsort(-1 * P_degree):
        if not PXbuf[P[v_i]]:
            max_colors = set_color(Gnew_indices, Gnew_start, Gnew_end, G_indices, G_start, G_end,
                                   pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)
    PXbuf[incd] = True
    PXbuf[new_incd] = True
        
    max_possible = R.size + max_colors
    if max_possible < min(min_cover, min_cover_within, min_cover_between):
        tree_size[12, curr_tree_size] = 5
        tree_size[0, 1] += 1
        return cliques, cliques_indptr, cliques_n, 0, tree_size
        
    tmp_sizes = get_branch_sizes(Gnew_indices, Gnew_start, Gnew_end, G_indices, G_start, G_end, pos, sep, PS, XE, colors, nei_bool, nei_list, incd)
    see_purpose = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
    if see_purpose:
        within_purpose = (R.size + 1 + tmp_sizes) >= min_cover_within_incd
        between_purpose = np.zeros(incd.size, np.bool_)
        tmp_between_purpose = (pot_end[incd] > pot_start[incd]).nonzero()[0]
        between_purpose[tmp_between_purpose[pot_indices[pot_end[incd[tmp_between_purpose]] - 2] <= (R.size + 1 + tmp_sizes[tmp_between_purpose])]] = True
        branches_purpose = within_purpose | between_purpose
        if not np.any(branches_purpose):
            tree_size[12, curr_tree_size] = 8
            tree_size[0, 1] += 1
            return cliques, cliques_indptr, cliques_n, 0, tree_size
    else:
        branches_purpose = np.zeros(incd.size, np.bool_)
        
    ## Get branches
    Padj_u = G_indices[G_start[u] : u_curr]
    Padj_new_u = Gnew_indices[Gnew_start[u] : u_curr_new : 2]        
    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
    
    ### Consideration: maybe between_new[P] should only be included if the pivot is in P?    
    # Always keep the between_new
    PXbuf[P[between_new[P]]] = True
    
    # Ensure that for any node left out of P, all of its Gnew neighbors in P are included
    for v in incd[~ PXbuf[incd]]:
        if not PXbuf[v]:
            # By construction, this will only iterate over w's that are in incd
            for w in Gnew_indices[Gnew_start[v] : Gnew_end[v]: 2]:
                w_pos = pos[w]
                PXbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
    branches = incd[PXbuf[incd]]
    if see_purpose:
        extra_priority = branches_purpose[PXbuf[incd]]
    else:
        extra_priority = np.zeros(branches.size, np.bool_)
    
    branches_sizes = colors[pos[branches] - PS] - 1
    
    # Update branch sizes based on nodes removed by pivoting
    distinct_bool = np.zeros(PX.size, np.bool_)
    distinct = np.empty(PX.size + 1, np.int32)
    incd_bool = np.zeros(PX.size, np.bool_)
    incd_bool[incd] = True    
    for v_i in range(branches.size):
        v = branches[v_i]
        v_color = colors[pos[v] - PS]
        distinct_count = 0
        for w in G_indices[G_start[v] : G_end[v]]:
            w_pos = pos[w]
            if PS <= w_pos and w_pos < sep and incd_bool[w] and (not PXbuf[w]) and colors[w_pos - PS] > v_color:
                w_col = colors[w_pos - PS]
                if not distinct_bool[w_col]:
                    distinct_bool[w_col] = True
                    distinct[distinct_count] = w_col
                    distinct_count += 1
            elif PS > w_pos or w_pos >= XE:
                break
        for w in Gnew_indices[Gnew_start[v] : Gnew_end[v] : 2]:
            w_pos = pos[w]
            if PS <= w_pos and w_pos < sep and incd_bool[w] and (not PXbuf[w]) and colors[w_pos - PS] > v_color:
                w_col = colors[w_pos - PS]
                if not distinct_bool[w_col]:
                    distinct_bool[w_col] = True
                    distinct[distinct_count] = w_col
                    distinct_count += 1
            elif PS > w_pos or w_pos >= XE:
                break
        distinct_bool[distinct[:distinct_count]] = False
        branches_sizes[v_i] += distinct_count
        
    PXbuf[Padj_new_u] = True
    PXbuf[Padj_u] = True

    #####
    ## Strongly prioritize branching off between_new first
#     branches_order = np.argsort(-1 * (branches + (1000000 * branches_sizes)))
    branches_order = np.argsort(-1 * (branches_sizes + (1000000 * extra_priority)))
    branches = branches[branches_order]
    branches_sizes = branches_sizes[branches_order]
    
#     print indent, 'Gnew:', [(i, Gnew_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(Gnew_start, Gnew_end))]    
#     print indent, 'min_cover_within:', min_cover_within    
#     print indent, 'incd:', incd[:incd_count]
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'Padj_new_u:', Padj_new_u
#     print indent, 'incd:', incd
#     print indent, 'new_incd:', new_incd
#     print indent, 'branches:', branches
        
    # Initialize max_cover
    max_cover = 0
    init_cliques_n = cliques_n
    
    last_tree_size = tree_size[0,0]
    
    used = np.empty(branches.size, np.int32)
    used_count = 0
    
    colors_v = np.empty(PX.size, np.int32)
    colors_v[P] = colors    
    
    tree_size[15, curr_tree_size] = 0
    
    v_i = -1
    check_swap = depth == 0
    if check_swap:     
        delayed = np.empty(branches.size, np.int32)
        delayed_sizes = np.empty(branches.size, np.int32)
        delayed_count = 0
        fill = 0
    while True:
        v_i += 1
        if check_swap:
            if v_i==branches.size:                
                if delayed_count > 0:
                    v_i = fill
                    branches[fill:] = delayed[:delayed_count]
                    branches_sizes[fill:] = delayed_sizes[:delayed_count]
                    check_swap = False
                    if verbose and depth == 0:
                        print('DONE SWAPPING, DEPTH:')
                        print((depth))
                else:
                    break
            else:
                v = branches[v_i]
                
                found = False
                for w_i in range(Gnew_start[v], Gnew_end[v], 2):
                    w = Gnew_indices[w_i]
                    w_pos = pos[w]
                    if (PS <= w_pos) and (w_pos < sep):
                        if Gnew_indices[w_i+1]==0:
                            found = True
                            break
                    elif w_pos < PS or w_pos >= XE:
                        break

                if not found:
                    delayed[delayed_count] = v
                    delayed_sizes[delayed_count] = branches_sizes[v_i]
                    delayed_count += 1
                    continue
                else:
                    branches[fill] = v
                    fill += 1
        else:
            if v_i==branches.size:
                break
            
        v = branches[v_i]
#         print indent, 'branching at:', v
#         print indent, 'pot_min:', pot_min
#         print indent, 'min/max_cover:', min_cover, max_cover        
#         print indent, 'pot_start/end/indices:', [(i,pot_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(pot_start, pot_end))]
#         print indent, 'pot_start/end:', zip(pot_start, pot_end)
#         print indent, 'Gnew:', [(i, x, y, Gnew_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(Gnew_start, Gnew_end))]
#         print indent, 'sub_max_possible:', sub_max_possible, 'min(min_cover, min_cover_within, min_cover_between):', min(min_cover, min_cover_within, min_cover_between)
#         sub_max_possible = R.size + 1 + branches_sizes[v_i]

        tmp = get_branch_sizes_vw(Gnew_indices, Gnew_start, Gnew_end,
                                  G_indices, G_start, G_end,
                                  pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
        sub_max_possible = R.size + 1 + tmp

        if sub_max_possible >= min(min_cover, min_cover_within, min_cover_between):
            used[used_count] = v
            used_count += 1
        else:
            continue
            
        min_newly_capt = 1000000
        sub_min_cover_between = 1000000
        sub_min_cover_within = 1000000
        between_added = 0
        new_PS, new_XE = sep, sep
        
        sub_tree_size = 4 + tree_size[0,0]
    
        for w_i in range(G_start[v], G_end[v]):            
            w = G_indices[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)
        
                if pot_end[w] > pot_start[w]:
                    sub_min_cover_between = min(sub_min_cover_between, pot_indices[pot_end[w]-2])                
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < PS or w_pos >= XE:
                break
        
        for w_i in range(Gnew_start[v], Gnew_end[v], 2):
            w = Gnew_indices[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)               
                if not between_new[w]:
                    between_stack[between_end + between_added] = w
                    between_added += 1
                    between_new[w] = True

                if sub_max_possible >= Gnew_indices[w_i+1]:
                    pot_indices[pot_end[w]] = Gnew_indices[w_i+1]
                    if pot_end[w] > pot_start[w]:
                        pot_indices[pot_end[w]+1] = min(pot_indices[pot_end[w]-2], Gnew_indices[w_i+1])
                    else:
                        pot_indices[pot_end[w]+1] = Gnew_indices[w_i+1]
                    pot_end[w] += 3
                if pot_end[w] > pot_start[w]:
                    sub_min_cover_between = min(sub_min_cover_between, pot_indices[pot_end[w]-2])
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < PS or w_pos >= XE:
                break
        
        # Update min cover based on r4: r1,r2,r3
        newly_capt = pot_indices[pot_start[v] : pot_end[v]]
        if newly_capt.size > 0:
            min_newly_capt = newly_capt[newly_capt.size - 2]
        
        ## Don't need to apply this criterion yet (since Gnew assumes no new edge yet)
        ## However, if you do, then do it on a min_below computes by PX-by-PX looping above
        if sep - new_PS == 0:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_between)
        else:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_between)
        
        if do_subbranch:
            sub_tree_size = 4 + tree_size[0, 0]
            if tree_size.shape[1] <= sub_tree_size:
                tree_size = expand_2d_arr(tree_size)

        if do_subbranch:
            R_buff[R_end] = v
            sub_min_cover = min(min_cover, min_newly_capt)
            
            prev_cliques_n = cliques_n
            
            if between_new[v]:
                cliques, cliques_indptr, cliques_n, sub_max_cover, tree_size = BKPivotSparse2_Gsep_cover(
                    R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                    G_start, G_end, G_indices,
                    Gnew_start, Gnew_end, Gnew_indices,
                    PXbuf, depth+1,
                    pot_indices, pot_start, pot_end, pot_min, sub_max_possible,
                    sub_min_cover, max_cover, sub_min_cover_between, max_branch_depth,
                    cliques, cliques_indptr, cliques_n, tree_size)
            else:
                cliques, cliques_indptr, cliques_n, sub_max_cover, tree_size = BKPivotSparse2_Gnew_cover(
                    R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                    G_start, G_end, G_indices,
                    Gnew_start, Gnew_end, Gnew_indices,
                    PXbuf, depth+1,
                    between_new, between_stack, between_end + between_added,
                    pot_indices, pot_start, pot_end, pot_min, sub_max_possible,
                    sub_min_cover, max_cover, sub_min_cover_between, max_branch_depth,
                    cliques, cliques_indptr, cliques_n, tree_size)

            # Update min_cover, max_cover
            max_cover = max(max_cover, sub_max_cover)
            min_cover = max(min_cover, sub_max_cover)
            
            if sub_max_cover > min_newly_capt:
                # Update max cover of (r4,r1), (r4,r2), (r4,r3), which we will move back to potential, and restore captured.
                # Fresh updates to max cover are moved to accum r1:...r4, r2:...r4, r3:...r4
                for w_i in range(0, newly_capt.size, 3):
                    newly_capt[w_i] = max(newly_capt[w_i], sub_max_cover)
                    newly_capt[w_i+1] = max(newly_capt[w_i+1], sub_max_cover)

        # Restore pot[w] for all new neighbors of v
        for w_i in range(Gnew_start[v], Gnew_end[v], 2):
            w = Gnew_indices[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                if sub_max_possible >= Gnew_indices[w_i+1]:
                    if do_subbranch:
#                         assert Gnew_indices[w_i+1] == Gnew_indices[pot_indices[pot_end[w]-1]]
                        Gnew_indices[w_i+1] = max(Gnew_indices[w_i+1], pot_indices[pot_end[w]-3])
                        Gnew_indices[pot_indices[pot_end[w]-1]] = max(Gnew_indices[pot_indices[pot_end[w]-1]], pot_indices[pot_end[w]-3])                
                    pot_end[w] -= 3
            elif w_pos < PS or w_pos >= XE:
                break
        
        # Recompute min_cover_within  
        if do_subbranch and sub_max_cover > min_cover_within:
            min_cover_within = 1000000
            for x_i in range(sep-1, PS-1, -1):
                x = PX[x_i]
                # Move P and X to the bottom
                for w_i in range(Gnew_start[x], Gnew_end[x], 2):
                    w = Gnew_indices[w_i]
                    w_pos = pos[w]
                    if w_pos < PS or w_pos >= XE:
                        break
                    elif PS <= w_pos and w_pos < sep:
                        min_cover_within = min(min_cover_within, Gnew_indices[w_i+1])
        
        # Reset the between_new
        between_new[between_stack[between_end : between_end + between_added]] = False

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)
        
        if depth==0:
            if verbose: print((v, sub_max_possible, sub_max_cover, do_subbranch, last_tree_size - tree_size[0,0]))
            last_tree_size = tree_size[0,0]
            
        # Don't branch anymore after this
        if depth > max_branch_depth and used_count==1:
            break

    for v in used[:used_count][::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1
    
#     print indent, 'Returning::::'
#     print indent, 'R:', R, 'P:', P, 'X:', X
#     print indent, 'pot_min:', pot_min
#     print indent, 'min/max_cover:', min_cover, max_cover
#     print indent, 'pot_start/end/indices:', [(i,pot_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(pot_start, pot_end))]    
#     print indent, 'Gnew:', [(i, x, y, Gnew_indices[x:y].tolist()) for i, (x, y) in enumerate(zip(Gnew_start, Gnew_end))]
    
    return cliques, cliques_indptr, cliques_n, max_cover, tree_size

def max_clique_cover(to_cover, G, verbose=False, pmc=False):
    """Call's PMC maximum clique on each new edge"""

    import clique_maximum
    
    if isspmatrix_csc(G):
        G = csr_matrix(G)
        assert isspmatrix_csr(G)
    else:
        raise Exception()
    
    clique_list = []
    clique_set = set([])
    
    # Unexplained edges
    e_i, e_j = to_cover.nonzero()
    tmp = e_i < e_j
    e_i, e_j = e_i[tmp], e_j[tmp]
    edges = set(zip(e_i, e_j))

    it = 0
    start = time.time()
        
    # Iterate until there are no more unexplained edges
    while len(edges) > 0:
        # if verbose:
        #     print 'edges:', edges

        it += 1
        i, j = edges.pop()
        
        # nonzero()[1] assumes that G is sparse. Otherwise do nonzero()[0]
        if verbose:
            print i, j, G[i, :].nonzero()[1], G[j, :].nonzero()[1]
        P = np.intersect1d(G[i, :].nonzero()[1], G[j, :].nonzero()[1])
        P.sort()

        if P.size > 1:
            G_P = G[P,:][:,P]

            if pmc:
                clique = pmc.pmc(G_P, threads=48, verbose=False)            
                if clique.size==0:
                    # Pick a random elt from P to make the clique
                    clique = (P[0], i, j)
                else:
                    clique = tuple([P[k] for k in clique] + [i,j])
                cliques = [clique]
            else:
                cliques, _, _ = clique_maximum.MC_py(G_P, verbose=False)
                cliques = [tuple([P[k] for k in c] + [i,j]) for c in clixov_utils.csc_to_cliques_list(cliques)]

        elif P.size==1:
            cliques = [(P[0], i, j)]
        else:
            cliques = [(i,j)]

        cliques = [tuple(sorted(c)) for c in cliques]

        if verbose:
            print 'c:', cliques, 'P:', P, 'i,j:', (i,j)
            
        clique_list.extend(cliques)

        # for c in cliques:
        #     assert c not in clique_set
        #     clique_set.add(c)

        # # Remove edges that were just explained
        # for c in cliques:
        #     edges -= set(itertools.combinations(sorted(c), 2))

    clique_list = sorted(set(clique_list))
    
    print 'Augment iterations:', it, 'time:', time.time() - start
    
    return clique_list
    
def print_tree(tree_size):
    
    # In[45]:

    print tree_size[0,:4]
    step = 30
    width = 6
    marker_types = ['o', 'x']
    marker = marker_types[0]
    # for i in range(min(tree_size[0,0] / step + 1, 100)):
    for i in range(52, 100):
        print step*i, step*(i+1) 
        print '-------', ''.join([str(x).ljust(width) for x in range(step*i, step*(i+1))])
        print '-------', ''.join(['-----'.ljust(width) for x in range(step*i, step*(i+1))])  # ---------------------
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[0, step*i:step*(i+1)]])  # depth
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[7, step*i:step*(i+1)]])  # sub_max_possible
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[6, step*i:step*(i+1)]])  # P.size
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[18, step*i:step*(i+1)]])  # original branches_sizes for this branch
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[19, step*i:step*(i+1)]])  # recomputed branches_sizes for this branch (influenced by earlier branches)
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[20, step*i:step*(i+1)]]) # best size estimate among all branches
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[17, step*i:step*(i+1)]])  # best possible from X
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[9, step*i:step*(i+1)]])  # v
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[10, step*i:step*(i+1)]]) # Gnew/Gsep
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[12, step*i:step*(i+1)]]) # exit status
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[13, step*i:step*(i+1)]]) # sub_max_cover
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[21, step*i:step*(i+1)]]) # pivot u
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[11, step*i:step*(i+1)]]) # branches (raw)
        print '\t', ''.join([str(x).ljust(width) for x in tree_size[15, step*i:step*(i+1)]]) # actual branches that branched

        print '\t', ''.join([str(x).ljust(width) for x in tree_size[8, step*i:step*(i+1)]]) # Parent index

        tmp = []
        for j in range(step*i, step*(i+1)):
            if tree_size[8, j] != j-1:
                marker = marker_types[1] if marker==marker_types[0] else marker_types[0]
    #             0 / asdf
            tmp.append(marker)
    #     print tmp
        print '\t', ''.join([x.ljust(width) for x in tmp])

        print '\t', ''.join([str(x).ljust(width) for x in np.minimum(9000, tree_size[1, step*i:step*(i+1)])]) # min_cover
        print '\t', ''.join([str(x).ljust(width) for x in np.minimum(9000, tree_size[2, step*i:step*(i+1)])]) # min_newly_capt
        print '\t', ''.join([str(x).ljust(width) for x in np.minimum(9000, tree_size[3, step*i:step*(i+1)])]) # min_cover_within
        print '\t', ''.join([str(x).ljust(width) for x in np.minimum(9000, tree_size[4, step*i:step*(i+1)])]) # sub_min_cover_between

