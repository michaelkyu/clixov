import time, os
import sys

import numpy as np
import pandas as pd
from numba import jit

import scipy, scipy.sparse
from scipy.sparse import isspmatrix_csc, isspmatrix_csr, issparse, isspmatrix, csc_matrix, csr_matrix, coo_matrix

import clixov_utils
from clixov_utils import trim_cliques
import clique_maximal
import clique_maximum
from clique_atomic import *
from color import set_color, get_branch_sizes, get_branch_sizes_vw

verbose = False

def BK_Gnew_cover_wrapper(Gold, Gnew, PX=None, degeneracy=None, max_branch_depth=100000):
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
    btw_new = np.zeros(PX.size, np.bool_)
    btw_stack = np.arange(PX.size).astype(np.int32)
    btw_end = 0
    
    C, CP, CN = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    CP[:2] = 0
   
    GI_new = np.empty(2 * Gnew.indices.size, np.int32)    
    GI_new[1::2] = 0
    
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
        GI_new[::2] = degen_pos[Gnew.indices]
        GS_new = 2 * Gnew.indptr[:-1][degen_order]
        GE_end = 2 * Gnew.indptr[1:][degen_order]
        
        Gold_indices = degen_pos[Gold.indices]
        Gold_start, Gold_end = Gold.indptr[:-1][degen_order], Gold.indptr[1:][degen_order]
    else:
        GI_new[::2] = Gnew.indices
        GS_new = 2 * Gnew.indptr[:-1]
        GE_end = 2 * Gnew.indptr[1:]        
        Gold_start, Gold_end, Gold_indices = Gold.indptr[:-1], Gold.indptr[1:], Gold.indices
    
    potI = np.empty(3 * Gnew.indices.size, np.int32)
    potI[:] = 0
    potS = GS_new.copy() * 3 / 2
    potE = GS_new.copy() * 3 / 2
        
    pot_min = np.zeros(PX.size, np.int32)
    pot_min.fill(1000000)
    
    min_cover, max_cover, min_cover_btw = 0, 0, 0
    
    max_possible = PX.size
    
    tree = np.asfortranarray(np.zeros((22, 100000), np.int32))
    tree.fill(-1)
    tree[0, :2] = np.array([0,0])
    
    C, CP, CN, max_cover, tree = BK_Gnew_cover(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        Gold_start, Gold_end, Gold_indices,
        GS_new, GE_end, GI_new,
        PXbuf2, 0, btw_new, btw_stack, btw_end,
        potI, potS, potE, pot_min, max_possible,
        min_cover, max_cover, min_cover_btw, max_branch_depth,
        C, CP, CN, tree)

    C, CP, CN = trim_cliques(C, CP, CN)
    
    if degeneracy in ['min', 'max']:
        C = degen_order[C]
        
    tree = tree[:, : 4 + tree[0,0]]

    return C, CP, CN, tree

@jit(nopython=True, cache=cache)
def BK_Gsep_cover(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                              GS, GE, GI, GS_new, GE_end, GI_new,
                              PXbuf, depth,
                              potI, potS, potE, pot_min, prev_max_possible,
                              min_cover, max_cover, min_cover_btw, max_branch_depth,
                              C, CP, CN, tree):

    orig_size = 4 + tree[0, 0]
    curr_tree = 4 + tree[0, 0]    
    if tree.shape[1] <= curr_tree:
        tree = expand_2d_arr(tree)    
    tree[0, 0] += 1
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]

    # if verbose:
    #     indent = '\t' * depth
    #     print indent, '---------Gsep------'
    #     print indent, 'DEPTH:', depth
    #     print indent, 'PX:', PX
    #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    #     print indent, 'R:', R, 'P:', P, 'X:', X
    #     print indent, 'min_cover:', min_cover, 'min_cover_btw:', min_cover_btw, 'prev_max_possible:', prev_max_possible
    #     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
    #     print indent, 'Gold:', [(i, x, y, GI[x:y].tolist()) for i, (x, y) in enumerate(zip(GS, GE))]
    #     print indent, 'Gnew:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
        
    if P.size==0:
        tree[0, 1] += 1
        if X.size==0:
            tree[12, curr_tree] = 1
            R_size = R.size

            if verbose: print((1111111, R[0], curr_tree, depth, min_cover))
            C, CP, CN = update_cliques2(C, CP, CN, R)            
            return C, CP, CN, R.size, tree        
        else:
            tree[12, curr_tree] = 2
            return C, CP, CN, 0, tree
    
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
        v_degree, curr = move_PX(GI, GS, GE,
                                 pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])

        # Move P and X to the bottom
        v_degree_new, curr_new = 0, GS_new[v]
        for w_i in range(GS_new[v], GE_end[v], 2):
            w = GI_new[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_new += w_pos < sep
                if v_i < sep and w_pos < sep:
                    min_cover_within_P[v_i - PS] = min(min_cover_within_P[v_i - PS], GI_new[w_i+1])
                GI_new[curr_new], GI_new[w_i] = w, GI_new[curr_new]
                GI_new[curr_new+1], GI_new[w_i+1] = GI_new[w_i+1], GI_new[curr_new+1]
                curr_new += 2

            # Accounts for the case when curr_new was incremented
            if (prev_max_possible >= GI_new[w_i+1]) and (v_i < sep) and GI_new[w_i] == prev_r:
                potI[potE[v]-1] = w_i + 1
                
        v_degree += v_degree_new
        
        if v_degree > max_degree:
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new        
        if v_i < sep:
            min_cover_within = min(min_cover_within, min_cover_within_P[v_i - PS])
            P_degree[v_i - PS] = v_degree
            P_degree_new[v_i - PS] = v_degree_new
    
    if min(prev_max_possible, R.size + 1 + P_degree.max()) < min(min_cover, min_cover_btw, min_cover_within):
        return C, CP, CN, 0, tree    
    
    branches = P.copy()

    # Color the nodes
    colors = np.zeros(sep - PS, np.int32)
    nei_bool = np.zeros(sep - PS + 1, np.bool_)
    nei_list = np.empty(sep - PS, np.int32)
    max_colors = 0   
    
    for v_i in np.argsort(-1 * P_degree):
        max_colors = set_color(GI_new, GS_new, GE_end,
                               GI, GS, GE,
                               pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)    
        
    max_possible = R.size + max_colors    
    if max_possible < min(min_cover, min_cover_within, min_cover_btw):
        return C, CP, CN, 0, tree
    
    # Get bound based on X
    tmp_sizes = get_branch_sizes(GI_new, GS_new, GE_end,
                                 GI, GS, GE,
                                 pos, sep, PS, XE, colors, nei_bool, nei_list, P)
    tmp_sizes_argsort = np.argsort(-1 * tmp_sizes)    
    best = 10000
    for v in PX[sep:XE]:
        X_keep = np.ones(sep - PS, np.bool_)
        for w in GI[GS[v] : GE[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif w_pos < sep:
                X_keep[w_pos - PS] = False
        for w in GI_new[GS_new[v] : GE_end[v] : 2]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif w_pos < sep:
                X_keep[w_pos - PS] = False
        for w_i in tmp_sizes_argsort:
            if X_keep[w_i]:
                best = min(best, tmp_sizes[w_i])
                break

    if R.size + 1 + best < min(min_cover, min_cover_within, min_cover_btw):
        return C, CP, CN, 0, tree
    
    see_purpose = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
    if see_purpose:
        within_purpose = (R.size + 1 + tmp_sizes) >= min_cover_within_P
        btw_purpose = np.zeros(P.size, np.bool_)
        tmp_btw_purpose = (potE[P] > potS[P]).nonzero()[0]
        btw_purpose[tmp_btw_purpose[potI[potE[P[tmp_btw_purpose]] - 2] <= (R.size + 1 + tmp_sizes[tmp_btw_purpose])]] = True
        P_purpose = within_purpose | btw_purpose
        if not np.any(P_purpose):
            return C, CP, CN, 0, tree
        
        branches = P[P_purpose]
        branches_degree = P_degree[P_purpose]
    else:
        branches_degree = P_degree
    
#     print indent, 'branches:', branches
#     print indent, 'colors:', colors[pos[branches] - PS]
#     print indent, 'branches_sizes:', branches_sizes
#     print indent, 'min(min_cover, min_cover_within, min_cover_btw):', min(min_cover, min_cover_within, min_cover_btw)
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'branches:', branches
    
    # Initialize max_cover
    max_cover = 0
    init_CN = CN

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
        sub_min_cover_btw = 1000000
        new_PS, new_XE = sep, sep
        
        tmp = get_branch_sizes_vw(GI_new, GS_new, GE_end,
                                  GI, GS, GE,
                                  pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
        sub_max_possible = R.size + 1 + tmp
        if sub_max_possible >= min(min_cover, min_cover_within, min_cover_btw):
            used[used_count] = v
            used_count += 1
        else:
            continue
            
        for w_i in range(GS[v], GE[v]):
            w = GI[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)
                
                if potE[w] > potS[w]:
                    sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            else:
                break
                
        for w_i in range(GS_new[v], GE_end[v], 2):
            w = GI_new[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)
                if sub_max_possible >= GI_new[w_i+1]:
                    potI[potE[w]] = GI_new[w_i+1]
                    if potE[w] > potS[w]:
                        potI[potE[w]+1] = min(potI[potE[w]-2], GI_new[w_i+1])
                    else:
                        potI[potE[w]+1] = GI_new[w_i+1]
                    potE[w] += 3
                
                if potE[w] > potS[w]:
                    sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])
    
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < PS or w_pos >= XE:
                break
  
        newly_capt = potI[potS[v] : potE[v]]
        if newly_capt.size > 0:
            min_newly_capt = newly_capt[newly_capt.size - 2]        

        # Not necessarily true as branches are moved to X
        if sep - new_PS == 0:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_btw)
        else:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_btw)
        
        if do_subbranch:
            sub_tree = 4 + tree[0,0]
            if tree.shape[1] <= sub_tree:
                tree = expand_2d_arr(tree)

        if do_subbranch:
            R_buff[R_end] = v
            sub_min_cover = min(min_cover, min_newly_capt)

            prev_CN = CN
            
            C, CP, CN, sub_max_cover, tree = BK_Gsep_cover(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_end, GI_new,
                PXbuf, depth+1,
                potI, potS, potE, pot_min, sub_max_possible,
                sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
                C, CP, CN, tree)
                
            # Update min_cover, max_cover
            max_cover = max(max_cover, sub_max_cover)
            min_cover = max(min_cover, sub_max_cover)
            
            if sub_max_cover > min_newly_capt:
                for w_i in range(0, newly_capt.size, 3):
                    newly_capt[w_i] = max(newly_capt[w_i], sub_max_cover)
                    newly_capt[w_i+1] = max(newly_capt[w_i+1], sub_max_cover)

        # Restore pot[w] for all new neighbors of v
        for w_i in range(GS_new[v], GE_end[v], 2):
            w = GI_new[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                if sub_max_possible >= GI_new[w_i+1]:
                    if do_subbranch:
#                         assert GI_new[w_i+1] == GI_new[potI[potE[w]-1]]
                        GI_new[w_i+1] = max(GI_new[w_i+1], potI[potE[w]-3])
                        GI_new[potI[potE[w]-1]] = max(GI_new[potI[potE[w]-1]], potI[potE[w]-3])
            
                    potE[w] -= 3
            elif w_pos < PS or w_pos >= XE:
                break

        # Recompute min_cover_within
        if do_subbranch and sub_max_cover > min_cover_within:
            min_cover_within = 1000000
            for x_i in range(sep-1, PS-1, -1):
                x = PX[x_i]
                # Move P and X to the bottom
                for w_i in range(GS_new[x], GE_end[x], 2):
                    w = GI_new[w_i]
                    w_pos = pos[w]
                    if w_pos < PS or w_pos >= XE:
                        break
                    elif PS <= w_pos and w_pos < sep:
                        min_cover_within = min(min_cover_within, GI_new[w_i+1])
    
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
#     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
#     print indent, 'Gnew:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]  
#     assert np.all(potE >= potS)

    return C, CP, CN, max_cover, tree

@jit(nopython=True, cache=cache)
def BK_Gnew_cover(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                              GS, GE, GI, GS_new, GE_end, GI_new,
                              PXbuf, depth,
                              btw_new, btw_stack, btw_end,
                              potI, potS, potE, pot_min, prev_max_possible,
                              min_cover, max_cover, min_cover_btw, max_branch_depth,
                              C, CP, CN, tree):
    """
    btw_new[v]: Is there a new edge crossing btw R and node v in P?
    """
    curr_tree = 4 + tree[0, 0]
    if tree.shape[1] <= curr_tree:
        tree = expand_2d_arr(tree)    
    tree[10, curr_tree] = 1
    tree[0, curr_tree] = depth
    tree[0, 0] += 1
    
    if verbose and (curr_tree % 25000) == 0:
        print((3333333, curr_tree))
    
    if curr_tree >= 90000:
        return C, CP, CN, max_cover, tree
    
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
    #     print indent, 'min_cover:', min_cover, 'min_cover_btw:', min_cover_btw, 'prev_max_possible:', prev_max_possible
    #     print indent, 'btw_new:', btw_new.astype(np.int32)
    #     print indent, 'pot_min:', pot_min
    #     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
    #     print indent, 'Gold:', [(i, x, y, GI[x:y].tolist()) for i, (x, y) in enumerate(zip(GS, GE))]
    #     print indent, 'Gnew:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
    
    if P.size==0:
        tree[12, curr_tree] = 2
        tree[0, 1] += 1
        return C, CP, CN, 0, tree

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
        v_degree_new, curr_new = 0, GS_new[v]
        for w_i in range(GS_new[v], GE_end[v], 2):
            w = GI_new[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_new += w_pos < sep
                if inP and w_pos < sep:
                    min_cover_within_v = min(min_cover_within_v, GI_new[w_i+1])
                GI_new[curr_new], GI_new[w_i] = w, GI_new[curr_new]
                GI_new[curr_new+1], GI_new[w_i+1] = GI_new[w_i+1], GI_new[curr_new+1]
                curr_new += 2
            
            if inP and (prev_max_possible >= GI_new[w_i+1]) and GI_new[w_i] == prev_r:
                potI[potE[v]-1] = w_i + 1
                
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new, u_incd = v, curr_new, incd_count
        if (not inP) and v_degree_new > 0:
            tmp, curr = move_PX(GI, GS, GE,
                                pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
        if inP and (v_degree_new > 0 or btw_new[v]):
            incd[incd_count] = v
            incd_degree[incd_count] = v_degree_new            
            min_cover_within_incd[incd_count] = min_cover_within_v            
            incd_count += 1
            
        if inP:
            P_degree[v_i - PS] += v_degree_new
            P_degree_new[v_i - PS] += v_degree_new
            min_cover_within = min(min_cover_within, min_cover_within_v)            
    u_curr = GE[u]
    
    if min(prev_max_possible, R.size + (sep - PS)) < min(min_cover, min_cover_btw, min_cover_within):
        return C, CP, CN, 0, tree
    
    if incd_count == 0:
        tree[12, curr_tree] = 4
        tree[0, 1] += 1
        return C, CP, CN, 0, tree
    incd = incd[:incd_count]
    min_cover_within_incd = min_cover_within_incd[:incd_count]
    incd_degree = incd_degree[:incd_count]
    X_incd = X_incd[:X_incd_count]

    PXbuf[incd] = False
    PXbuf[X_incd] = False
    new_incd = np.empty(XE - PS - incd.size, PX.dtype)
    new_incd_end = 0
    
    for v in incd:
        curr = GS[v]
        v_degree_old = 0
        
        # Move P and X to the bottom
        for w_i in range(GS[v], GE[v]):
            w = GI[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_old += w_pos < sep
                GI[curr], GI[w_i] = w, GI[curr]
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
        curr = GS[v]
        v_degree = 0
        for w_i in range(GS[v], GE[v]):
            w = GI[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                ## Counts only those neighbors that are in P and in (incd, new_incd)
                v_degree += (w_pos < sep) and not PXbuf[w]
                GI[curr], GI[w_i] = w, GI[curr]
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
        for w in GI_new[GS_new[v] : GE_end[v] : 2]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            v_degree += (w_pos < sep) and (not PXbuf[w])
        for w in GI[GS[v] : GE[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            v_degree += (w_pos < sep) and (not PXbuf[w])
        P_degree[v_i] = v_degree
                
    for v_i in np.argsort(-1 * P_degree):
        if not PXbuf[P[v_i]]:
            max_colors = set_color(GI_new, GS_new, GE_end, GI, GS, GE,
                                   pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)
    PXbuf[incd] = True
    PXbuf[new_incd] = True
        
    max_possible = R.size + max_colors
    if max_possible < min(min_cover, min_cover_within, min_cover_btw):
        tree[12, curr_tree] = 5
        tree[0, 1] += 1
        return C, CP, CN, 0, tree
        
    tmp_sizes = get_branch_sizes(GI_new, GS_new, GE_end, GI, GS, GE, pos, sep, PS, XE, colors, nei_bool, nei_list, incd)
    see_purpose = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
    if see_purpose:
        within_purpose = (R.size + 1 + tmp_sizes) >= min_cover_within_incd
        btw_purpose = np.zeros(incd.size, np.bool_)
        tmp_btw_purpose = (potE[incd] > potS[incd]).nonzero()[0]
        btw_purpose[tmp_btw_purpose[potI[potE[incd[tmp_btw_purpose]] - 2] <= (R.size + 1 + tmp_sizes[tmp_btw_purpose])]] = True
        branches_purpose = within_purpose | btw_purpose
        if not np.any(branches_purpose):
            tree[12, curr_tree] = 8
            tree[0, 1] += 1
            return C, CP, CN, 0, tree
    else:
        branches_purpose = np.zeros(incd.size, np.bool_)
        
    ## Get branches
    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = GI_new[GS_new[u] : u_curr_new : 2]        
    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
    
    ### Consideration: maybe btw_new[P] should only be included if the pivot is in P?    
    # Always keep the btw_new
    PXbuf[P[btw_new[P]]] = True
    
    # Ensure that for any node left out of P, all of its Gnew neighbors in P are included
    for v in incd[~ PXbuf[incd]]:
        if not PXbuf[v]:
            # By construction, this will only iterate over w's that are in incd
            for w in GI_new[GS_new[v] : GE_end[v]: 2]:
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
        for w in GI[GS[v] : GE[v]]:
            w_pos = pos[w]
            if PS <= w_pos and w_pos < sep and incd_bool[w] and (not PXbuf[w]) and colors[w_pos - PS] > v_color:
                w_col = colors[w_pos - PS]
                if not distinct_bool[w_col]:
                    distinct_bool[w_col] = True
                    distinct[distinct_count] = w_col
                    distinct_count += 1
            elif PS > w_pos or w_pos >= XE:
                break
        for w in GI_new[GS_new[v] : GE_end[v] : 2]:
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
    ## Strongly prioritize branching off btw_new first
#     branches_order = np.argsort(-1 * (branches + (1000000 * branches_sizes)))
    branches_order = np.argsort(-1 * (branches_sizes + (1000000 * extra_priority)))
    branches = branches[branches_order]
    branches_sizes = branches_sizes[branches_order]
    
#     print indent, 'Gnew:', [(i, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]    
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
    init_CN = CN
    
    last_tree = tree[0,0]
    
    used = np.empty(branches.size, np.int32)
    used_count = 0
    
    colors_v = np.empty(PX.size, np.int32)
    colors_v[P] = colors    
    
    tree[15, curr_tree] = 0
    
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
                for w_i in range(GS_new[v], GE_end[v], 2):
                    w = GI_new[w_i]
                    w_pos = pos[w]
                    if (PS <= w_pos) and (w_pos < sep):
                        if GI_new[w_i+1]==0:
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
#         print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
#         print indent, 'potS/end:', zip(potS, potE)
#         print indent, 'Gnew:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
#         print indent, 'sub_max_possible:', sub_max_possible, 'min(min_cover, min_cover_within, min_cover_btw):', min(min_cover, min_cover_within, min_cover_btw)
#         sub_max_possible = R.size + 1 + branches_sizes[v_i]

        tmp = get_branch_sizes_vw(GI_new, GS_new, GE_end,
                                  GI, GS, GE,
                                  pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
        sub_max_possible = R.size + 1 + tmp

        if sub_max_possible >= min(min_cover, min_cover_within, min_cover_btw):
            used[used_count] = v
            used_count += 1
        else:
            continue
            
        min_newly_capt = 1000000
        sub_min_cover_btw = 1000000
        sub_min_cover_within = 1000000
        btw_added = 0
        new_PS, new_XE = sep, sep
        
        sub_tree = 4 + tree[0,0]
    
        for w_i in range(GS[v], GE[v]):            
            w = GI[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)
        
                if potE[w] > potS[w]:
                    sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])                
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < PS or w_pos >= XE:
                break
        
        for w_i in range(GS_new[v], GE_end[v], 2):
            w = GI_new[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)               
                if not btw_new[w]:
                    btw_stack[btw_end + btw_added] = w
                    btw_added += 1
                    btw_new[w] = True

                if sub_max_possible >= GI_new[w_i+1]:
                    potI[potE[w]] = GI_new[w_i+1]
                    if potE[w] > potS[w]:
                        potI[potE[w]+1] = min(potI[potE[w]-2], GI_new[w_i+1])
                    else:
                        potI[potE[w]+1] = GI_new[w_i+1]
                    potE[w] += 3
                if potE[w] > potS[w]:
                    sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < PS or w_pos >= XE:
                break
        
        # Update min cover based on r4: r1,r2,r3
        newly_capt = potI[potS[v] : potE[v]]
        if newly_capt.size > 0:
            min_newly_capt = newly_capt[newly_capt.size - 2]
        
        ## Don't need to apply this criterion yet (since Gnew assumes no new edge yet)
        ## However, if you do, then do it on a min_below computes by PX-by-PX looping above
        if sep - new_PS == 0:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_btw)
        else:
            do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_btw)
        
        if do_subbranch:
            sub_tree = 4 + tree[0, 0]
            if tree.shape[1] <= sub_tree:
                tree = expand_2d_arr(tree)

        if do_subbranch:
            R_buff[R_end] = v
            sub_min_cover = min(min_cover, min_newly_capt)
            
            prev_CN = CN
            
            if btw_new[v]:
                C, CP, CN, sub_max_cover, tree = BK_Gsep_cover(
                    R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                    GS, GE, GI,
                    GS_new, GE_end, GI_new,
                    PXbuf, depth+1,
                    potI, potS, potE, pot_min, sub_max_possible,
                    sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
                    C, CP, CN, tree)
            else:
                C, CP, CN, sub_max_cover, tree = BK_Gnew_cover(
                    R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                    GS, GE, GI,
                    GS_new, GE_end, GI_new,
                    PXbuf, depth+1,
                    btw_new, btw_stack, btw_end + btw_added,
                    potI, potS, potE, pot_min, sub_max_possible,
                    sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
                    C, CP, CN, tree)

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
        for w_i in range(GS_new[v], GE_end[v], 2):
            w = GI_new[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                if sub_max_possible >= GI_new[w_i+1]:
                    if do_subbranch:
#                         assert GI_new[w_i+1] == GI_new[potI[potE[w]-1]]
                        GI_new[w_i+1] = max(GI_new[w_i+1], potI[potE[w]-3])
                        GI_new[potI[potE[w]-1]] = max(GI_new[potI[potE[w]-1]], potI[potE[w]-3])                
                    potE[w] -= 3
            elif w_pos < PS or w_pos >= XE:
                break
        
        # Recompute min_cover_within  
        if do_subbranch and sub_max_cover > min_cover_within:
            min_cover_within = 1000000
            for x_i in range(sep-1, PS-1, -1):
                x = PX[x_i]
                # Move P and X to the bottom
                for w_i in range(GS_new[x], GE_end[x], 2):
                    w = GI_new[w_i]
                    w_pos = pos[w]
                    if w_pos < PS or w_pos >= XE:
                        break
                    elif PS <= w_pos and w_pos < sep:
                        min_cover_within = min(min_cover_within, GI_new[w_i+1])
        
        # Reset the btw_new
        btw_new[btw_stack[btw_end : btw_end + btw_added]] = False

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)
        
        if depth==0:
            if verbose: print((v, sub_max_possible, sub_max_cover, do_subbranch, last_tree - tree[0,0]))
            last_tree = tree[0,0]
            
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
#     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]    
#     print indent, 'Gnew:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
    
    return C, CP, CN, max_cover, tree

def max_clique_cover(to_cover, G, verbose=False, pmc=False):
    """Call's PMC maximum clique on each new edge"""
    
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

    clique_list = sorted(set(clique_list))
    
    print 'Augment iterations:', it, 'time:', time.time() - start
    
    return clique_list
    
