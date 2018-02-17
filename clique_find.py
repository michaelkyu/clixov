import cPickle, time, scipy, scipy.sparse, argparse, os, pickle, datetime, gzip, subprocess, StringIO, random, sys, tempfile, shutil, igraph, multiprocessing, glob
from itertools import combinations, chain, groupby, compress, permutations, product
import itertools
import numpy as np
from collections import Counter
import pandas as pd
from scipy.sparse import isspmatrix_csc, isspmatrix_csr, issparse, isspmatrix, csc_matrix, csr_matrix, coo_matrix, _sparsetools

from ctypes import pointer, Structure, POINTER,c_void_p,c_int,c_char,c_double,byref,cdll, c_long, c_float, c_int64
import ctypes

from numba import int64, float32, float64, boolean, guvectorize, jit, vectorize

orig_stdout = sys.stdout
orig_stderr = sys.stderr




# # New edge cover

# ### Wrappers

# In[98]:

def BKPivotSparse2_Gnew_max_wrapper(G_indices, Gold_start, Gold_end, Gnew_indices, Gnew_start, Gnew_end, PX=None,
                                    degeneracy='blah', method=None, max_cover=0, verbose=True):
    if PX is None:
        k = Gold_start.size
        PX = np.arange(k).astype(np.int32)
        pos = np.empty(PX.size, np.int32)
        pos[PX] = np.arange(PX.size)
    else:
        k = Gold_start.size
        pos = np.empty(k, np.int32)
        pos[:] = -1
        pos[PX] = np.arange(PX.size)
        
        initialize_PX(G_indices, Gold_start, Gold_end, pos, PX)
        initialize_PX(G_indices, Gnew_start, Gnew_end, pos, PX)
        
    R = np.zeros(PX.size, np.int32)
    R_end = np.int32(0)
    PXbuf = np.ones(k, np.bool_)
    PXbuf2 = np.zeros(k, np.bool_)
    PXbuf_int = np.empty(k, np.int32)
    PS, sep, XE = np.int32([0, PX.size, PX.size])

#     print 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(Gnew_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(PX.size) if Gnew_end[v] > Gnew_start[v]]
    
#     if degeneracy=='min':
#         start = time.time()
#         degen_order, degen_deg = get_degeneracy_min_new(Gold_start, Gold_end, G_indices, Gnew_start, Gnew_end, G_indices)
#         degen_pos = np.empty(k, np.int32)
#         degen_pos[degen_order] = np.arange(k).astype(np.int32)
#         print 'degen_order:', degen_order
#         print 'degen_pos:', degen_pos
#         G_indices = degen_pos[G_indices]
#         Gnew_start = Gnew_start[degen_order]
#         Gnew_end = Gnew_end[degen_order]
#         Gold_start = Gold_start[degen_order]
#         Gold_end = Gold_end[degen_order]
#         if verbose: print 'Degeneracy time:', time.time() - start
        
    cliques, cliques_indptr, cliques_n = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0
    
    Gnew_diff = np.cumsum(Gnew_end - Gnew_start).astype(np.int32)
    X_indices = np.empty(Gnew_diff[-1], np.int32)
    X_start = np.concatenate((np.array([0], np.int32), Gnew_diff[:-1]))
    X_end = X_start.copy()
    
    tree_size = np.asfortranarray(np.zeros((14, 100000), np.int32))
    tree_size.fill(-1)
    tree_size[0, :2] = np.array([0,0])
    
    0 / asdf
    
    cliques, cliques_indptr, cliques_n, tree_size, max_cover =             BKPivotSparse2_Gnew_max_cover(R, R_end, PX, PS, sep, XE, PS, XE, pos,
                                    Gold_start, Gold_end, G_indices,
                                    Gnew_start, Gnew_end, Gnew_indices,
                                    PXbuf, PXbuf2, PXbuf_int, 0,max_cover,
                                    X_indices, X_start, X_end,
                                     cliques, cliques_indptr, cliques_n, tree_size, verbose=verbose)
        
    cliques, cliques_indptr = cliques[:cliques_indptr[cliques_n]], cliques_indptr[:cliques_n+1]
    
    if degeneracy in ['min', 'max']:
#         print 'cliques:', cliques
        cliques = degen_order[cliques]
        tmp = tree_size[7,:] != -1
        tree_size[7, tmp] = degen_order[tree_size[7, tmp]]
        
    tree_size = tree_size[:, : 4 + tree_size[0,0]]
    return cliques, cliques_indptr, cliques_n, tree_size


# In[96]:

def BKPivotSparse2_Gnew_cover_wrapper(Gold, Gnew, verbose=True):
    if verbose: print 'Beginning wrapper', time.time()
    
    k = Gold.shape[0]
    PX = np.arange(k).astype(np.int32)
    assert Gnew[:,PX][PX,:].sum() > 0
    
    Gold_start, Gold_end, Gold_indices = Gold.indptr[:-1], Gold.indptr[1:], Gold.indices
    Gnew_start, Gnew_end, Gnew_indices = Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices
    
    # Remove nodes that are not itself incident on a new edge nor is adjacent to such a node.
    print 'Removing non incd'
    Gold_start, Gold_end, Gold_indices, Gnew_start, Gnew_end, Gnew_indices, subset = remove_non_incd(Gold_start, Gold_end, Gold_indices, Gnew_start, Gnew_end, Gnew_indices)
    PX = np.arange(Gold_end.size).astype(np.int32)
    
    
#     ## Degeneracy
#     degen_order, core_num = get_degeneracy_min_new_P_Gnew2(Gnew_start, Gnew_end, Gnew_indices, G_start, G_curr, G_indices, P)
    
    
    print 'subset:', subset
    
    print 'Gold:', [(v, Gold_start[v], Gold_end[v], list(Gold_indices[Gold_start[v] : Gold_end[v]])) for v in np.arange(PX.size) if Gold_end[v] > Gold_start[v]]
    print 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(Gnew_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(PX.size) if Gnew_end[v] > Gnew_start[v]]
    
    if verbose: print 'Total new edges:', (Gnew_end - Gnew_start).sum()
        
    # Double the size of Gnew_indices
    tmp = np.empty(2 * Gnew_indices.size, np.int32)
    tmp[1::2] = 0
    tmp[::2] = Gnew_indices
    Gnew_indices = tmp
    Gnew_start = 2 * Gnew_start
    Gnew_end = 2 * Gnew_end

    cliques_list = []    
    start_iteration_time = time.time()
    
    cliques, cliques_indptr, cliques_n, tree_size = BKPivotSparse2_Gnew_max_wrapper(Gold_indices, Gold_start, Gold_end, Gnew_indices, Gnew_start, Gnew_end, PX=PX.copy(), verbose=verbose)
    if verbose: print 'tree_size:', tree_size[0, :3]
    if verbose: print 'Clique sizes (%s):' % cliques_n, Counter(cliques_indptr[1:] - cliques_indptr[:-1])
    if verbose: print 'Cliques indices:', cliques.size

    tmp_cliques = [tuple(cliques[cliques_indptr[i]:cliques_indptr[i+1]]) for i in range(cliques_n)]        
    max_size = max([len(x) for x in tmp_cliques])
    print 'Cliques:', tmp_cliques
#         print 'Cliques:', [x for x in tmp_cliques if len(x) == max_size]
    cliques = tuples_to_csc(tmp_cliques, k)
#     cliques = cliques[:,get_largest_cliques(cliques)]
    cliques, _, _ = get_unique_cols(cliques, get_column_hashes(cliques))
    curr_size = cliques.sum(0)[0,0]
    cliques_list.append(cliques)
    print 'Iteration time:', time.time() - start_iteration_time

    cliques = scipy.sparse.hstack(cliques_list)
    covers = get_largest_clique_covers(cliques, Gnew)
    covers = csc_to_cliques_list(cliques[:, covers])
#     if verbose: print len(covers), 'covers', Counter([len(x) for x in covers])
    print 'Covers:', len(covers)
    print 'Covers by frequency:', Counter([len(x) for x in covers])
    print 'Covers by size:', sorted(Counter([len(x) for x in covers]).items(), key=lambda x:x[0], reverse=True)
        
    # assert len(covers) == cliques.shape[1]
    print 'My covers:', len(covers), covers
        
    return covers


# In[95]:

@jit(nopython=True)
def remove_non_incd(G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices):
    ## Remove nodes that are not itself incident on a new edge nor is adjacent to such a node.
    
    # incd : the set of nodes incident on new edges
    incd = (Gnew_end - Gnew_start).nonzero()[0]
    assert incd.size > 0
    
    # new_incd : the set of nodes that are adjacent to nodes in incd
    new_incd_bool = np.zeros(PX.size, np.bool_)
    for v in incd:
        for w in G_indices[G_start[v] : G_end[v]]:
            new_incd_bool[w] = True
    new_incd_bool[incd] = False
    new_incd = new_incd_bool.nonzero()[0]
    
    print 'incd:', incd
    print 'new_incd:', new_incd
    
    # Nodes to keep
    keep = np.concatenate([incd, new_incd])
    
    # reorder[v] = the new index of v after non-incident nodes are removed
    reorder = np.empty(PX.size, np.int32)
    reorder[keep] = np.arange(keep.size)
    
    # Initialize new data structures
    sub_G_start = np.empty(keep.size, G_start.dtype)
    sub_G_end = np.empty(keep.size, G_start.dtype)
    sub_G_indices = np.empty(np.sum(G_end[keep] - G_start[keep]), G_indices.dtype)
    sub_Gnew_start = np.empty(keep.size, Gnew_start.dtype)
    sub_Gnew_end = np.empty(keep.size, Gnew_start.dtype)
    sub_Gnew_indices = np.empty(np.sum(Gnew_end[keep] - Gnew_start[keep]), Gnew_indices.dtype)    
    
    # Populate data structures
    for v_i in range(keep.size):
        v = keep[v_i]
        
        # Update Gnew structure
        if v_i > 0:
            sub_Gnew_start[v_i] = sub_Gnew_end[v_i - 1]
        else:
            sub_Gnew_start[v_i] = 0
        sub_Gnew_end[v_i] = sub_Gnew_start[v_i] + (Gnew_end[v] - Gnew_start[v])
        sub_Gnew_indices[sub_Gnew_start[v_i] : sub_Gnew_end[v_i]] = reorder[Gnew_indices[Gnew_start[v] : Gnew_end[v]]]
        
        # Update Gold structure
        if v_i > 0:
            sub_G_start[v_i] = sub_G_end[v_i - 1]
        else:
            sub_G_start[v_i] = 0
        sub_G_end[v_i] = sub_G_start[v_i] + (G_end[v] - G_start[v])
        sub_G_indices[sub_G_start[v_i] : sub_G_end[v_i]] = reorder[G_indices[G_start[v] : G_end[v]]]    
    
    return sub_G_start, sub_G_end, sub_G_indices, sub_Gnew_start, sub_Gnew_end, sub_Gnew_indices, keep


# ### Gsep7

# In[52]:

@jit(nopython=True)
def BKPivotSparse2_Gsep_max_cover(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                                  G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
                                  PXbuf, PXbuf2, PXbuf_int, in_P, colors_v, depth, max_cover, Gnew_curr_R,
                                  cliques, cliques_indptr, cliques_n, tree_size, verbose=True):
    curr_tree_size = 4 + tree_size[0, 0]
    if tree_size.shape[1] <= curr_tree_size:
        tree_size = expand_2d_arr(tree_size)
    tree_size[0, curr_tree_size] = depth
    tree_size[12, curr_tree_size] = R_end
    tree_size[1, curr_tree_size] = 2
    tree_size[2, curr_tree_size] = max_cover   
    tree_size[0, 0] += 1
    
    R = R_buff[:R_end]
    P = PX[PS:sep]
    XE = sep
    
    last_r = R_buff[R_end]

    if verbose and (curr_tree_size % 50000)==0:
        print((3333333, curr_tree_size))
    
    indent = '\t' * depth
    print indent, '---------Gsep------'
    print indent, 'DEPTH:', depth
    print indent, 'PX:', PX
    print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    print indent, 'R:', R, 'P:', P
    print indent, 'in_P:', in_P.astype(np.int32)
    print indent, 'Gold:', [(v, G_start[v], G_end[v], list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(PX.size) if G_end[v] > G_start[v]]
    print indent, 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(Gnew_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(PX.size) if Gnew_end[v] > Gnew_start[v]]
#     print indent, 'max_cover:', max_cover

#     assert np.all(np.sort(in_P.nonzero()[0]) == np.sort(P))
#     assert np.all(colors_v[P] == 0)
        
    if P.size==0:
        if R.size >= max_cover:
            tree_size[3, curr_tree_size] = 1
            if verbose:
                if R.size >= 4:
                    print((111111, R[0], R[1], R[2], R[3], 5555, curr_tree_size, R.size, max_cover))
                else:
                    print((111111, R[0], 5555, curr_tree_size, R.size, max_cover))
            sub_max_cover = R.size
            cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R)
        else:
            sub_max_cover = 0
        return cliques, cliques_indptr, cliques_n, tree_size, sub_max_cover
    
    Gnew_end_prev, G_end_prev = Gnew_end[P], G_end[P]
    P_copy = P.copy()
        
    max_degree = -1
    P_degree = np.empty(P.size, np.int32)
    for v_i in range(P.size):
        v = P[v_i]
        v_degree_new, Gnew_end[v] = move_PX_fast_bool_Gnew2(Gnew_indices, Gnew_start, Gnew_end, in_P, v)
        v_degree_old, G_end[v] = move_PX_fast_bool(G_indices, G_start, G_end, in_P, v)
        v_degree = v_degree_old + v_degree_new
        P_degree[v_i] = v_degree
        max_degree = max(max_degree, v_degree)
    
    tree_size[10, curr_tree_size] = P.size
    tree_size[11, curr_tree_size] = max_degree
    
    branches, limit = P, P.size - 1
    currPS, currXE = PS, XE
    # Require that max degree pivot is in P and that, to reduce overhead, when P.size > 1
    if max_degree==limit and P.size > 2:
        # Add as many of the P into R as possible
        for v_i in range(branches.size):
            if P_degree[v_i]==limit:
                v = branches[v_i]                
                R_buff[R_end] = v
                R_end += 1
                
                a, b = pos[v]-PS, currPS-PS
                P_degree[a], P_degree[b] = P_degree[b], P_degree[a]
                swap_pos(PX, pos, v, currPS)
                currPS += 1
                print indent, 'Swapped v:', v, 'PX:', PX
        R = R_buff[:R_end]
        
        if sep - currPS == 0:
            # Return clique
            if R.size >= max_cover:
                tree_size[3, curr_tree_size] = 1
                if verbose:
                    if R.size >= 4:
                        print((111111, R[0], R[1], R[2], R[3], 5555, curr_tree_size, R.size, max_cover))
                    else:
                        print((111111, R[0], 5555, curr_tree_size, R.size, max_cover))
                sub_max_cover = R.size
                cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R_buff[:R_end])
            else:
                sub_max_cover = 0
            Gnew_end[P_copy], G_end[P_copy] = Gnew_end_prev, G_end_prev
            return cliques, cliques_indptr, cliques_n, tree_size, sub_max_cover
        
    if currPS > PS:
        P_degree = P_degree[currPS - PS :]
        P_degree[:] -= currPS - PS
        P = PX[currPS : sep]
        in_P[PX[PS : currPS]] = False
        
#     print indent, 'later', 'R:', R, 'P:', P
#     print indent, 'currPS/sep:', currPS, sep
#     print indent, 'in_P', in_P
#     print indent, 'PX:', PX
    
#     assert np.all(np.sort(in_P.nonzero()[0]) == np.sort(P))
    
    # Color the nodes
    nei_bool = PXbuf2[:P.size + 1]
    nei_list = PXbuf_int[:P.size]
    max_colors = 0

    for v_i in np.argsort(-1 * (P + (10000 * P_degree))):
        v = P[v_i]
        v_color = set_color_Gnew2_fast(Gnew_indices, Gnew_start, Gnew_end,
                                       G_indices, G_start, G_end,
                                       P, v, colors_v)
        max_colors = max(max_colors, v_color)

    if R.size + max_colors < max_cover:
        Gnew_end[P_copy], G_end[P_copy] = Gnew_end_prev, G_end_prev
        in_P[PX[PS : currPS]] = True
        tree_size[3, curr_tree_size] = 2
        return cliques, cliques_indptr, cliques_n, tree_size, 0
    
    P_sizes = np.empty(P.size, np.int32)
    for v_i in range(P_sizes.size):
        P_sizes[v_i] = get_branch_sizes_vw_Gnew2_fast(Gnew_indices, Gnew_start, Gnew_end,
                                                      G_indices, G_start, G_end,
                                                      colors_v, nei_bool, nei_list, P[v_i])
    tmp_sort = np.argsort(P_sizes)
    branches_curr_sizes = P_sizes[tmp_sort] + R.size + 1
    branches_curr_sizes_v = P[tmp_sort]
    branches_start = 0
    
    keep = R.size + colors_v[P] >= max_cover
    branches = P[keep]
    branches_sizes = R.size + colors_v[branches]
    branches_degrees = P_degree[keep]
    branches_order = np.argsort(-1 * (branches_degrees + (10000 * branches_sizes)))
    branches = branches[branches_order]
    branches_sizes = branches_sizes[branches_order]    
    
    print indent, 'branches/sizes', zip(branches, branches_sizes)
    print indent, 'branches_curr_sizes_v/branches_curr_sizes:', zip(branches_curr_sizes_v, branches_curr_sizes)
    
    sub_max_cover = 0
    for v_i in range(branches.size):
        v = branches[v_i]
        print indent, 'branching at v:', v, 'v_size:', branches_sizes[v_i], 'max_cover:', max_cover
        
        if branches_sizes[v_i] < max_cover:
            branches = branches[:v_i]
            break
            
        print indent, 'Actual branching v:', v
        
        new_PS, nei_count = update_PX_bool_color_Gnew2(G_indices, G_start, G_end,
                                                       Gnew_indices, Gnew_start, Gnew_end,
                                                       nei_bool, colors_v, in_P, pos, PX, sep, v)
        assert nei_count <= (sep - new_PS)

        print indent, 'nei_count:', nei_count, 'new_PS/sep:', new_PS, sep, 'PX:', PX        
        
        if R.size + 1 + nei_count >= max_cover:
            print indent, 'Branching v:', v
            
            R_buff[R_end] = v

            sub_tree_size = 4 + tree_size[0, 0]
            if tree_size.shape[1] <= sub_tree_size:  tree_size = expand_2d_arr(tree_size)
            tree_size[5, sub_tree_size] = branches_sizes[v_i]
            tree_size[6, sub_tree_size] = curr_tree_size
            tree_size[7, sub_tree_size] = v
            tree_size[13, sub_tree_size] = nei_count
            
            sub_P = PX[new_PS : sep].copy()
            orig_colors_sub_P = colors_v[sub_P]
            colors_v[sub_P] = 0
            in_P[PX[currPS : new_PS]] = False
            cliques, cliques_indptr, cliques_n, tree_size, tmp_cover =                     BKPivotSparse2_Gsep_max_cover(R_buff, R_end + 1, PX, new_PS, sep, sep, PS, XE, pos,
                                                    G_start, G_end, G_indices,
                                                    Gnew_start, Gnew_end, Gnew_indices,
                                                    PXbuf, PXbuf2, PXbuf_int, in_P, colors_v,
                                                    depth+1, max_cover, Gnew_curr_R, 
                                                    cliques, cliques_indptr, cliques_n, tree_size, verbose=verbose)     
            sub_max_cover = max(sub_max_cover, tmp_cover)      
            in_P[PX[currPS : new_PS]] = True
            colors_v[sub_P] = orig_colors_sub_P

            if tmp_cover > max_cover:
                max_cover = tmp_cover
                while (branches_start < branches_curr_sizes.size) and (branches_curr_sizes[branches_start] < max_cover):
#                     print indent, 'Removing', branches_curr_sizes_v[branches_start], 'max_cover:', max_cover, 'branches_curr_sizes[branches_start]', branches_curr_sizes[branches_start], 'sep:', sep
                    sep -= 1
                    swap_pos(PX, pos, branches_curr_sizes_v[branches_start], sep)        
                    in_P[branches_curr_sizes_v[branches_start]] = False
                    branches_start += 1
            
#             # Restore pot[w] for all new neighbors of v
#             for w_i in range(Gnew_start[v], Gnew_end[v], 2):
#                 w = Gnew_indices[w_i]
#                 if in_P[w]:
#                     Gnew_indices[w_i+1] = max(Gnew_indices[w_i+1], pot_indices[pot_end[w]-3])
#                     Gnew_indices[pot_indices[pot_end[w]-1]] = max(Gnew_indices[pot_indices[pot_end[w]-1]], pot_indices[pot_end[w]-3])
#                     pot_end[w] -= 3
               
        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)
        in_P[v] = False
        
    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1
        in_P[v] = True
        
    for v in branches_curr_sizes_v[:branches_start]:
        swap_pos(PX, pos, v, sep)
        sep += 1    
        
    in_P[PX[PS : currPS]] = True
    in_P[branches_curr_sizes_v[:branches_start]] = True

    # Update upwards
    update_Gnew2_cover(Gnew_indices, Gnew_end, Gnew_curr_R, last_r, sub_max_cover)
    for v in PX[PS : currPS]:
        update_Gnew2_cover(Gnew_indices, Gnew_end, Gnew_curr_R, v, sub_max_cover)
    
    # Update downwards
    end_tmp = Gnew_end.copy()
    end_tmp[P_copy] = Gnew_end_prev
    in_triage = np.zeros(PX.size, np.bool_)
    in_triage[PX[PS : currPS]] = True
    for v in P_copy:
        for r_i in range(Gnew_end[v], end_tmp[v], 2):
            r = Gnew_indices[r_i]
            if r==last_r or in_triage[r]:
                Gnew_indices[r_i + 1] = max(Gnew_indices[r_i + 1], sub_max_cover)                
        
    Gnew_end[P_copy], G_end[P_copy] = Gnew_end_prev, G_end_prev
    
    tree_size[3, curr_tree_size] = 3
    tree_size[4, curr_tree_size] = branches.size
    tree_size[8, curr_tree_size] = sub_max_cover
    return cliques, cliques_indptr, cliques_n, tree_size, sub_max_cover


# ### Gnew7

# In[56]:

@jit(nopython=True)
def update_exclude_queue(Gnew_indices, Gnew_start, Gnew_curr,
                         X_indices, X_start, X_end,
                         PXbuf, queue):
    for x in queue[:q_end]:
        X_x = X_indices[X_start[x] : X_end[x]]
        PXbuf[X_x] = False
        for w in Gnew_indices[Gnew_start[x] : Gnew_curr[x]]:
            if PXbuf[w]:
                update_exclude_uv(X_indices, X_end, x, w)
        PXbuf[X_x] = True
        

@jit(nopython=True)
def append_P_nei(sep, PX, pos, u, colors_v, nei_count, nei_bool):
    sep -= 1
    swap_pos(PX, pos, u, sep)
    u_color = colors_v[u]
    nei_count += not nei_bool[u_color]
    nei_bool[u_color] = True
    return sep, nei_count


# In[100]:

@jit(nopython=True)
def MC_Gnew(R_buff, R_end, PX, PS, oldPS, oldXE, pos,
          G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
          PXbuf, PXbuf2, PXbuf_int, depth,
          max_cover,
          X_indices, X_start, X_end,
          cliques, cliques_indptr, cliques_n, tree_size, verbose=True):
    P = PX    
    
    indent = '\t' * depth
    print indent, '---------Gnew------'
    print indent, 'DEPTH:', depth
    print indent, 'PX:', PX
    print indent, 'R:', R, 'P:', P
    print indent, 'Gold:', [(v, G_start[v], G_end[v], list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(PX.size) if G_end[v] > G_start[v]]
    print indent, 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(Gnew_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(PX.size) if Gnew_end[v] > Gnew_start[v]]

    incd = (Gnew_end - Gnew_start).nonzero()[0]
    
    Gnew_curr = Gnew_end.copy()
    
    ## Color
    colors, branches = color_nodes_Gnew2(Gnew_indices, Gnew_start, Gnew_end, G_indices, G_start, G_curr, P[::-1])
    
    for v in incd:
        indent = '\t' * (depth+1)
        print indent, '-----------------------'
        print indent, 'Branching at v:', v
        
        sep = PX.size
        
        # Create P_v = neighbors of v to continue considering

        ## Add all neighbors that are connected to v by an old edge
        for u in G_indices[G_start[v] : G_curr[v]]:
            sep -= 1
            swap_pos(PX, pos, u, sep)
        sep_old = sep
        
        ## For each neighbor u connected to v by a new edge,
        ## -- add u to P if the max cover already found for (v,u) is less than the upper bound for the cover if we branch on u (based on coloring)
        ## -- also keep track of the max cover already found in <max_found>
        max_found = np.zeros(PX.size, np.int32)
        X_v = X_indices[X_start[v] : X_end[v]]
        PXbuf[X_v] = False
        for u_i in range(Gnew_start[v], Gnew_curr[v], 2):
            u = Gnew_indices[u_i]
            if PXbuf[u]:
                max_found[Gnew_indices[u_i]] = Gnew_indices[u_i + 1]
                update_exclude_uv(X_indices, X_end, u, v)
                if (1 + core_num[u]) >= Gnew_indices[u_i + 1]:                    
                    sep -= 1
                    swap_pos(PX, pos, u, sep)
        PXbuf[X_v] = True        
        
        # Copy P_v into its own array
        P_v = PX[sep : ].copy()  # The copy is necessary, because the order of PX[new_PS_v : sep] will change in sub-calls.
        P_v_new = P_v[sep_old : ] # The portion of P that contains the neighbors by new edges
        
        # Push down update of G_indices and Gnew_indices
        # Before pushing, save the current end
        Gnew_end_prev, G_end_prev = Gnew_curr[P_v], G_curr[P_v]
        PXbuf2[P_v] = True
        for u in P_v:
            u_degree_new, G_curr[u] = move_PX_fast_bool(G_indices, G_start, G_curr, PXbuf2, u)
            u_degree_old, Gnew_curr[u] = move_PX_fast_bool_X_Gnew2(Gnew_indices, Gnew_start, Gnew_curr,
                                                                   X_indices[X_start[u] : X_end[u]], PXbuf, PXbuf2, u)
        PXbuf2[P_v] = False
        
        ## Recolor P_v
        colors_v, branches_v = color_nodes_Gnew2(Gnew_indices, Gnew_start, Gnew_end, G_indices, G_start, G_curr, P_v[::-1])
        
        in_P_v[P_v] = True
        
        for w in P_v_new:
            indent = '\t' * (depth+2)
            print indent, 'Branching w:', w
            
            # branch_size = blah
            
            if (1 + colors_P_v[w]) >= max_found[w]:
                new_PS, nei_count_P_v = update_PX_bool_color_Gnew2(G_indices, G_start, G_curr,
                                                                   Gnew_indices, Gnew_start, Gnew_curr,
                                                                   nei_bool, colors_P_v, in_P_v, pos, PX, sep, w)
                assert nei_count_P_v <= (sep - new_PS)

                assert 2 + nei_count_P_v >= max_found[w]:                              
                R_buff[:2] = [v, w]

                Gnew_curr_R[PX[new_PS : sep]] = Gnew_curr[PX[new_PS : sep]]

                in_P_sep[PX[new_PS : sep]] = True
                colors_sep_v[PX[new_PS : sep]] = 0
                cliques, cliques_indptr, cliques_n, tree_size, tmp_cover =                         MC_Gsep(R_buff, R_end + 2, PX, new_PS, sep, sep, new_PS_v, new_XE_v, pos,
                              G_start, G_curr, G_indices,
                              Gnew_start, Gnew_curr, Gnew_indices,
                              PXbuf, PXbuf2, PXbuf_int, in_P_sep, colors_sep_v,
                              depth+2, max_cover, Gnew_curr_R,
                              cliques, cliques_indptr, cliques_n, tree_size, verbose=verbose)
                max_cover = max(max_cover, tmp_cover)
                in_P_sep[PX[new_PS : sep]] = False

            # Swap w to the end of P, and then decrement separator
            sep -= 1
            swap_pos(PX, pos, w, sep)
            
            in_P_v[w] = False
        in_P_v[P_v] = False
        
        for w in used[:used_count][::-1]:
            # Move v to the beginning of X and increment separator
            swap_pos(PX, pos, w, sep)
            sep += 1
    
        Gnew_curr[P_v], G_curr[P_v] = Gnew_end_prev, G_end_prev


# In[ ]:

### @jit(nopython=True)
def BKPivotSparse2_Gnew_max_cover(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                                G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
                                PXbuf, PXbuf2, PXbuf_int, depth,
                                max_cover,
                                X_indices, X_start, X_end,
                                cliques, cliques_indptr, cliques_n, tree_size, verbose=True):
    
    curr_tree_size = 4 + tree_size[0, 0]
    if tree_size.shape[1] <= curr_tree_size:
        tree_size = expand_2d_arr(tree_size)
    tree_size[0, curr_tree_size] = depth
    tree_size[1, curr_tree_size] = 1
    tree_size[2, curr_tree_size] = max_cover
    tree_size[0, 0] += 1
    
    R = R_buff[:R_end]
    P = PX[PS:sep]
    XE = sep
    
    indent = '\t' * depth
    print indent, '---------Gnew------'
    print indent, 'DEPTH:', depth
    print indent, 'PX:', PX
    print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    print indent, 'R:', R, 'P:', P
    print indent, 'Gold:', [(v, G_start[v], G_end[v], list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(PX.size) if G_end[v] > G_start[v]]
    print indent, 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(Gnew_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(PX.size) if Gnew_end[v] > Gnew_start[v]]
#     print indent, 'Gold:', [(v, list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(k)]
#     print indent, 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(G_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(k)]
#     print (Gnew_end - Gnew_start).sum(), 'New edges'
#     do_break = (Gnew_end - Gnew_start).sum() == 30
#     do_break = False
    
    if P.size==0:
        tree_size[1] += 1
        return cliques, cliques_indptr, cliques_n, tree_size, 0
    
    default_max = 100000
    
    G_curr = G_end.copy()
    Gnew_curr = Gnew_end.copy()
    Gnew_curr_R = Gnew_end.copy()
    
    # incd : the set of nodes incident on new edges
    P_degree = (Gnew_end - Gnew_start) / 2
    incd = P_degree.nonzero()[0]
    P_degree[incd] += G_end[incd] - G_start[incd]

    if incd.size == 0:
        tree_size[3, curr_tree_size] = 4
        return cliques, cliques_indptr, cliques_n, tree_size, 0
    
    # new_incd : the set of nodes that are adjacent to nodes in incd
    new_incd_bool = np.zeros(PX.size, np.bool_)
    for v in incd:
        for w in G_indices[G_start[v] : G_end[v] : 2]:
            new_incd_bool[w] = True
    new_incd_bool[incd] = False
    new_incd = new_incd_bool.nonzero()[0]

    PXbuf[incd] = False
    PXbuf[new_incd] = False
    for v in PX[PXbuf]:
        sep -= 1
        a, b = pos[v]-PS, sep-PS
        P_degree[a], P_degree[b] = P_degree[b], P_degree[a]
        swap_pos(PX, pos, v, sep)
    P = PX[PS:sep]
    P_degree = P_degree[:sep-PS]
    XE = sep
    PXbuf[incd] = True
    PXbuf[new_incd] = True
    
    PXbuf2[P] = True
    for v in new_incd:
        P_degree[pos[v]-PS], G_curr[v] = move_PX_fast_bool(G_indices, G_start, G_end, PXbuf2, v)
    PXbuf2[P] = False
    
    PXbuf3 = np.ones(PX.size, np.bool_) # False if node has been excluded because of branches_curr_sizes
    queue = np.empty(PX.size, np.int32)
    
    in_P_v = np.zeros(PX.size, np.bool_)
    in_P_sep = np.zeros(PX.size, np.bool_)
    colors_sep_v = np.zeros(PX.size, np.int32)

    # Color the nodes
    colors_v = np.zeros(PX.size, np.int32)
    nei_bool = np.zeros(P.size + 1, np.bool_)
    nei_list = np.empty(P.size, np.int32)
    max_colors = 0
    
    degen_order, degen_deg = get_degeneracy_min_new_P_Gnew2(Gnew_start, Gnew_end, Gnew_indices, G_start, G_curr, G_indices, P)
    
    for v in degen_order[::-1]:
        v_color = set_color_Gnew2_fast(Gnew_indices, Gnew_start, Gnew_end, G_indices, G_start, G_curr, P, v, colors_v)    

    P_sizes = np.empty(P.size, np.int32)
    for v_i in range(P_sizes.size):
        P_sizes[v_i] = get_branch_sizes_vw_Gnew2_fast(Gnew_indices, Gnew_start, Gnew_end, G_indices, G_start, G_curr,
                                                      colors_v, nei_bool, nei_list, P[v_i])
    print 'P:', P
    print 'P/colors/sizes:', zip(P, colors_v[P], P_sizes)

    branches = P.copy()
    branches_sizes = P_sizes[pos[branches] - PS]
    branches_degrees = P_degree[pos[branches] - PS]
    branches_colors = colors_v[branches]    
    branches_order = np.argsort(-1 * (branches_sizes + (10000 * branches_colors)))
    
    branches = branches[branches_order]
    branches_colors = branches_colors[branches_order]
    branches_sizes = branches_sizes[branches_order]
    branches_pos = np.empty(PX.size, np.int32)
    branches_pos[branches] = np.arange(branches.size)
    
    print 'branches:', branches
    print 'branches_pos:', branches_pos
    
    max_colors = branches_colors[0]
    cc = np.zeros((PX.size, max_colors + 1), np.int32)
    for v in P:
        for w in G_indices[G_start[v] : G_curr[v]]:
            cc[v, colors_v[w]] += 1
        for w in Gnew_indices[Gnew_start[v] : Gnew_curr[v]]:
            cc[v, colors_v[w]] += 1
    branches_curr_sizes = branches_sizes.copy()
    print 'branches/sizes/curr_sizes/colors:', zip(branches, branches_sizes, branches_curr_sizes, branches_colors)
    
    cc_v = np.zeros((PX.size, max_colors + 1), np.int32)

    v_used = 0
    
    is_incd, is_new_incd = np.zeros(PX.size, np.bool_), np.zeros(PX.size, np.bool_)
    is_incd[incd] = True
    is_new_incd[new_incd] = True
    P_old_count = 0
    
    print 'incd:', incd.size
    print 'new_incd:', new_incd.size    
    indent = '\t' * (depth + 1)

    for v_i in range(branches.size):
        v = branches[v_i]
        v_col = colors_v[v]

        print indent, '-----------------------'
        print indent, 'Branching at v:', v, 'max_cover:', max_cover, 'v_col:', v_col, 'curr_size:', branches_curr_sizes[branches_pos[v]]
        print indent, 'branches/sizes/curr_sizes/colors:', sorted(zip(branches, branches_sizes, branches_curr_sizes, branches_colors), key=lambda x:x[0])
        print indent, 'X_indices:', [(u, X_start[u], X_end[u], list(X_indices[X_start[u] : X_end[u]])) for u in np.arange(PX.size)]

        if is_new_incd[v]:  continue        
        assert branches_curr_sizes[v_i] <= branches_sizes[v_i]
        
#         # Calculate max_cover for this v
#         X_v = X_indices[X_start[v] : X_end[v]]
#         PXbuf[X_v] = False
#         max_cover = default_max
#         for u_i in range(Gnew_start[v], Gnew_curr[v], 2):
#             if PXbuf[Gnew_indices[u_i]]:
#                 max_cover = min(max_cover, Gnew_indices[u_i + 1])
#         PXbuf[X_v] = True
        
#         if max_cover==default_max or (1 + branches_curr_sizes[v_i]) < max_cover:
# #             print((2222222, v, max_cover, 1 + branches_sizes[v_i]))
#             continue
    
#         nei_count = 0
#         new_PS_v, new_XE_v = sep, sep
#         for u in G_indices[G_start[v] : G_curr[v]]:
#             print indent, 'u:', u,
#             print indent, 'branches_pos[u]:', branches_pos[u]
#             print indent, 'branches_curr_sizes[branches_pos[u]]:', branches_curr_sizes[branches_pos[u]]
#             if (1 + branches_curr_sizes[branches_pos[u]] >= max_cover):
#                 new_PS_v, nei_count = append_P_nei(new_PS_v, PX, pos, u, colors_v, nei_count, nei_bool)
#         new_PS_v_old = new_PS_v
              
# @jit(nopython=True)
# def append_P_nei(sep, PX, pos, u, colors_v, nei_count, nei_bool):
#     sep -= 1
#     swap_pos(PX, pos, u, sep)
#     u_color = colors_v[u]
#     nei_count += not nei_bool[u_color]
#     nei_bool[u_color] = True
#     return sep, nei_count

        X_v = X_indices[X_start[v] : X_end[v]]
        PXbuf[X_v] = False
        for u_i in range(Gnew_start[v], Gnew_curr[v], 2):
            u = Gnew_indices[u_i]
            if PXbuf[u]:
                update_exclude_uv(X_indices, X_end, u, v)
                update_cc(u, colors_v[u], v, v_col, cc, branches_curr_sizes, branches_pos)
 
                if (1 + branches_curr_sizes[branches_pos[u]] >= min(max_cover, Gnew_indices[u_i + 1])):
                    new_PS_v, nei_count = append_P_nei(new_PS_v, PX, pos, u, colors_v, nei_count, nei_bool)
        PXbuf[X_v] = True
        
        P_v = PX[new_PS_v : sep].copy()  # The copy is necessary, because the order of PX[new_PS_v : sep] will change in sub-calls.
        P_v_new = P_v[:new_PS_v_old - new_PS_v]
        nei_bool[colors_v[P_v]] = False
        assert nei_count <= P_v.size
        
        if (P_v_new.size==0) or (1 + nei_count < max_cover):
#             print((777777, v, max_cover, 1 + branches_sizes[v_i]))
            continue
        
        Gnew_end_prev, G_end_prev = Gnew_curr[P_v], G_curr[P_v]

        P_v_degree = np.empty(P_v.size, np.int32)
        PXbuf2[P_v] = True
        for u_i in range(P_v.size):
            u = P_v[u_i]
            u_degree_new, G_curr[u] = move_PX_fast_bool(G_indices, G_start, G_curr, PXbuf2, u)
            u_degree_old, Gnew_curr[u] = move_PX_fast_bool_X_Gnew2(Gnew_indices, Gnew_start, Gnew_curr,
                                                                   X_indices[X_start[u] : X_end[u]], PXbuf, PXbuf2, u)
            P_v_degree[u_i] = u_degree_new + u_degree_old
        PXbuf2[P_v] = False
        
        colors_P_v_vec = np.zeros(PX.size, np.int32)
        max_colors = 0        
        for u_i in np.argsort(-1 * (P_v + (10000 * P_v_degree))):
            u_color = set_color_Gnew2_fast(Gnew_indices, Gnew_start, Gnew_curr,
                                           G_indices, G_start, G_curr,
                                           P_v, P_v[u_i], colors_P_v_vec)
            max_colors = max(max_colors, u_color)
        print 'P_v_new:', P_v_new
        
        if 1 + max_colors < max_cover:
            Gnew_curr[P_v], G_curr[P_v] = Gnew_end_prev, G_end_prev
#             print((3333333, v, max_cover, 1 + branches_sizes[v_i], 1 + max_colors, new_PS_v, new_PS_v_old, sep))
            continue

        used = np.empty(P_v.size, np.int32)
        used_count = 0
        sub_max_cover = 0
        actual_branch = 0
        
        in_P_v[P_v] = True
        
        print 'v:', v, 'P_v:', P_v
        
        P_v_old_count = 0
        w_i_order = np.argsort(-1 * colors_P_v_vec[P_v])
        for w_ii in range(w_i_order.size):
            w_i = w_i_order[w_ii]

            if w_i >= P_v_new.size:
                P_v_old_count += 1
                continue
                
            print '\t w:', P_v[w_i], 'w_i:', w_i, 'w in P_v_new:', w_i < P_v_new.size, 'P_v_old_count:', P_v_old_count, 'max_cover:', max_cover, 'colors_P_v_vec[w]:', colors_P_v_vec[w]

            w = P_v[w_i]
            if (1 + colors_P_v_vec[w] + P_v_old_count) < max_cover:
                break

            new_PS, nei_count_P_v = update_PX_bool_color_Gnew2(G_indices, G_start, G_curr,
                                                               Gnew_indices, Gnew_start, Gnew_curr,
                                                               nei_bool, colors_P_v_vec, in_P_v, pos, PX, sep, w)
            assert nei_count_P_v <= (sep - new_PS)

            if 2 + nei_count_P_v >= max_cover:
                print 'Branching w:', w
                
                actual_branch += 1
                R_buff[:2] = [v, w]

                Gnew_curr_R[PX[new_PS : sep]] = Gnew_curr[PX[new_PS : sep]]
            
                in_P_sep[PX[new_PS : sep]] = True
                colors_sep_v[PX[new_PS : sep]] = 0
                cliques, cliques_indptr, cliques_n, tree_size, tmp_cover =                         BKPivotSparse2_Gsep_max_cover(R_buff, R_end + 2, PX, new_PS, sep, sep, new_PS_v, new_XE_v, pos,
                                                      G_start, G_curr, G_indices,
                                                      Gnew_start, Gnew_curr, Gnew_indices,
                                                      PXbuf, PXbuf2, PXbuf_int, in_P_sep, colors_sep_v,
                                                      depth+2, max_cover, Gnew_curr_R,
                                                      cliques, cliques_indptr, cliques_n, tree_size, verbose=verbose)
                sub_max_cover = max(sub_max_cover, tmp_cover)
                max_cover = max(max_cover, tmp_cover)
                in_P_sep[PX[new_PS : sep]] = False
                
#                 update_Gnew2_cover(Gnew_indices, Gnew_start, Gnew_curr, w, tmp_cover)
                
            used[used_count] = w
            used_count += 1
            
            # Swap w to the end of P, and then decrement separator
            sep -= 1
            swap_pos(PX, pos, w, sep)
            
            in_P_v[w] = False
        in_P_v[P_v] = False
        
        for w in used[:used_count][::-1]:
            # Move v to the beginning of X and increment separator
            swap_pos(PX, pos, w, sep)
            sep += 1
    
        Gnew_curr[P_v], G_curr[P_v] = Gnew_end_prev, G_end_prev

#         print((4444444, v, v_i, max_cover, 1 + branches_curr_sizes[v_i], 1 + branches_sizes[v_i], branches_colors[v_i], sub_max_cover, 55555,
#                used_count, actual_branch, new_PS_v, new_PS_v_old, sep, sep - new_PS_v, nei_count))
        v_used += 1
    
    if verbose:
        print 'v_considered:', v_i
        print 'v_used:', v_used

    tree_size[10, curr_tree_size] = P.size
    tree_size[11, curr_tree_size] = P_degree.max()
    tree_size[3, curr_tree_size] = 3
#     tree_size[4, curr_tree_size] = branches.size
    tree_size[4, curr_tree_size] = used_count
    tree_size[8, curr_tree_size] = sub_max_cover
    return cliques, cliques_indptr, cliques_n, tree_size, sub_max_cover


# ### Test7

# In[99]:

# for x in range(100):
k = 8
r = 6
s = 1

# dosim = True
dosim = False

# if dosim:
#     Gold = np.zeros((k,k), dtype=np.int32, order='F')
#     old_clusters = [np.random.randint(0, k, r) for i in range(s)]
#     for c in old_clusters:
#         Gold[np.ix_(c,c)] = True
#     old = Gold.copy()
#     np.fill_diagonal(Gold, 0)
#     new_clusters = [np.random.randint(0, k, r) for i in range(s)]
#     Gnew = np.zeros((k,k), dtype=np.int32, order='F')
#     for c in new_clusters:
#         Gnew[np.ix_(c,c)] = True
#     np.fill_diagonal(Gnew, 0)
#     Gnew -= (Gnew * Gold)
#     Gnew_dense = Gnew
#     assert (Gnew * Gold).sum() == 0
#     Gold, Gnew = csc_matrix(Gold), csc_matrix(Gnew)
#     G1 = Gnew + Gold
# G = csc_matrix(G1)

if dosim:
    G1 = np.zeros((k,k), dtype=np.bool, order='F')
    G1[np.random.randint(0, k, k * r), np.random.randint(0, k, k * r)] = True
    G1 = np.logical_or(G1, G1.T)
    np.fill_diagonal(G1, 0)
    G1 = G1.astype(np.bool, order='F')
G = csc_matrix(G1)

if dosim:
    # Randomly keep some edges as Gnew
    old = np.random.random(G.data.size) < 0.5

Gnew = G.copy()
Gnew.data[old] = 0
Gnew.eliminate_zeros()
Gnew = csc_matrix((Gnew.T.toarray() * Gnew.toarray()))
Gnew_dense = Gnew.toarray('F')
Gold = csc_matrix(G.toarray() & ~ Gnew.toarray())
assert Gnew.sum() > 0

# # ### Pickle
# import cPickle
# # 10, 30, 40, 102, 103, 119, 130, 160, 166, 239, 245, 246, 247
# # Recently hard: 140, 236, 323, 352, 353, 354
# dt_iter = 2
# with open('/cellar/users/mikeyu/clixov/dt_iters/%s.pickle' % dt_iter, 'rb') as f:
# # with open('/cellar/users/mikeyu/clixov/go_dt_iters/%s.pickle' % dt_iter, 'rb') as f:
#     tmp = cPickle.load(f)
# Gold, Gnew = tmp['G_sp'], tmp['dG_sp']
# G = (Gold + Gnew).toarray()
# k = Gold.shape[1]

# 0 / asdf

print Gnew.sum() / 2, Gold.sum() / 2, 'Gnew/old edges'
# print Gold.toarray().astype(np.int32)
print 'Gold:', sorted(set([tuple(sorted(x)) for x in zip(*Gold.nonzero())]))
# print Gnew.toarray().astype(np.int32)
print 'Gnew:', sorted(set([tuple(sorted(x)) for x in zip(*Gnew.nonzero())]))

# G_start, G_end, G_indices = Gold.indptr[:-1], Gold.indptr[1:], Gold.indices
# Gnew_start, Gnew_end, Gnew_indices = Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices
# print 'Gold:', [(v, G_start[v], G_end[v], list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(k) if G_end[v] > G_start[v]]
# print 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(G_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(k) if Gnew_end[v] > Gnew_start[v]]
    
start = time.time()
covers = BKPivotSparse2_Gnew_cover_wrapper(Gold.copy(), Gnew.copy(), verbose=True)
# covers, tree_size = BKPivotSparse2_Gnew_cover_wrapper(Gold.copy(), Gnew.copy())
print 'Total time:', time.time() - start

# # # ############
# # # ## Reference

# start = time.time()
# PX = np.arange(k).astype(np.int32)
# ref_cliques, ref_cliques_indptr, ref_cliques_n, _ = BKPivotSparse2_Gnew_wrapper(Gold, Gnew, PX=PX)
# print 'Ref cliques indices:', ref_cliques.size
# print 'Time:', time.time() - start

# ref_cliques = [tuple(sorted(ref_cliques[ref_cliques_indptr[i]:ref_cliques_indptr[i+1]])) for i in range(ref_cliques_n)]
# new_cliques = len(ref_cliques)
# total_cliques = len(ref_cliques)

# # print 'Ref cliques:', len(ref_cliques), sorted(ref_cliques)
# ref_cliques = tuples_to_csc(ref_cliques, k)
# start = time.time()
# ref_covers = get_largest_clique_covers(ref_cliques, Gnew)
# # ref_covers = get_largest_cliques(ref_cliques)
# print 'Filtering Time:', time.time() - start
# ref_covers = csc_to_cliques_list(ref_cliques[:, ref_covers])

# print 'New edges:', Gnew.sum() / 2, 'Old edges:', Gold.sum() / 2
# print 'new/total cliques:', new_cliques, total_cliques

# # print 'My covers:', len(covers), sorted(covers)
# # print 'Ref covers:', len(ref_covers), sorted(ref_covers)
# print len(covers), len(ref_covers), 'my/ref covers'
# print 'Maximum clique size:', len(covers[0]), len(ref_covers[0])
# assert sorted(covers) == sorted(ref_covers)

# # print((1111111, R[0], curr_tree_size, depth, min_cover))
# # print((4444444, v, v_i, max_cover, 1 + branches_curr_sizes[v_i], 1 + branches_sizes[v_i], branches_colors[v_i], sub_max_cover, 55555,
# #                used_count, actual_branch, new_PS_v, new_PS_v_old, sep, sep - new_PS_v, nei_count))
# # print((v, branches_sizes[v_i], branches_degrees[v_i], branches_degrees_new[v_i], R.size + 1 + (sep - new_PS), branches_colors[v_i], sub_max_cover))


# # Test DIMACS

# ## Table

# In[ ]:

print tree_size[0,:4]
step = 30
width = 6
marker_types = ['o', 'x']
marker = marker_types[0]
# for i in range(min(tree_size[0,0] / step + 1, 100)):
# for i in range(363, 390):
for i in range(0, 100):
    print step*i, step*(i+1) 
    print '-------', ''.join([str(x).ljust(width) for x in range(step*i, step*(i+1))])
    print '-------', ''.join(['-----'.ljust(width) for x in range(step*i, step*(i+1))])  # ---------------------
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[0, step*i:step*(i+1)]])  # depth
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[12, step*i:step*(i+1)]])  # R.size
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[1, step*i:step*(i+1)]])  # Gnew/Gsep
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[2, step*i:step*(i+1)]])  # max_cover at beginning
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[8, step*i:step*(i+1)]])  # sub_max_cover
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[3, step*i:step*(i+1)]])  # exit status
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[4, step*i:step*(i+1)]])  # number of branches
    print '-------', ''.join(['-----'.ljust(width) for x in range(step*i, step*(i+1))])  # ---------------------
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[5, step*i:step*(i+1)]])  # color estimate from parent
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[13 step*i:step*(i+1)]])  # nei_count
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[12, step*i:step*(i+1)]])  # R.size
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[6, step*i:step*(i+1)]])  # Parent index
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[7, step*i:step*(i+1)]])  # v
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[10, step*i:step*(i+1)]])  # P.size
    print '\t', ''.join([str(x).ljust(width) for x in tree_size[11, step*i:step*(i+1)]])  # P_degree.max()
    
    tmp = []
    for j in range(step*i, step*(i+1)):
        if tree_size[6, j] != j-1:
            marker = marker_types[1] if marker==marker_types[0] else marker_types[0]
        tmp.append(marker)
    print '\t', ''.join([x.ljust(width) for x in tmp])
                
# tree_size[0,0:500]


# ### End Test
