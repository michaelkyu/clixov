verbose = False

import time, os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from numba import jit

import scipy, scipy.sparse
from scipy.sparse import isspmatrix_csc, isspmatrix_csr, issparse, isspmatrix, csc_matrix, csr_matrix, coo_matrix

import clixov_utils
from clixov_utils import *
import clique_maximal
import clique_maximum
from clique_atomic import *
from clique_approx import MC_cover_approx_py
        
from color import set_color, get_branch_sizes, get_branch_sizes_vw, color_nodes, count_unique
from degeneracy import get_degeneracy

@jit(nopython=True)                    
def restore_pot(dGI, dGS, dGE, potI, potS, potE, pos, PS, sep, XE, sub_UB, v, do_subbranch, verbose=False):
    # Restore pot[w] for all new neighbors of v
    # if verbose:
    #     print 'dGS[v]/dGE[v]:', dGS[v], dGE[v]
    #     print 'dGI[dGS[v]:dGE[v]]:', dGI[dGS[v]:dGE[v]]
    for w_i in range(dGS[v], dGE[v], 2):
        w = dGI[w_i]
        # if verbose: print 'w:', w, 'w_i', w_i
        w_pos = pos[w]
        if (PS <= w_pos) and (w_pos < sep):
            if sub_UB >= dGI[w_i+1]:
                if do_subbranch:
                    # if verbose:
                    #     print 'potI[potE[w]-1]:', potI[potE[w]-1]
                    #     print 'potI[potE[w]-3]', potI[potE[w]-3]
                    #     print 'dGI[w_i+1]', dGI[w_i+1]
                        
                    # assert dGI[w_i+1] == dGI[potI[potE[w]-1]]
                    dGI[w_i+1] = max(dGI[w_i+1], potI[potE[w]-3])
                    dGI[potI[potE[w]-1]] = max(dGI[potI[potE[w]-1]], potI[potE[w]-3])
                potE[w] -= 3
        elif w_pos < PS or w_pos >= XE:
            break

@jit(nopython=True)
def recompute_min_cover_within(dGI, dGS, dGE, PX, pos, PS, sep, XE):
    min_cover_within = 1000000
    for x_i in range(sep-1, PS-1, -1):
        x = PX[x_i]
        # Move P and X to the bottom
        for w_i in range(dGS[x], dGE[x], 2):
            w = dGI[w_i]
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif PS <= w_pos and w_pos < sep:
                min_cover_within = min(min_cover_within, dGI[w_i+1])
    return min_cover_within

def BK_dG_cover_py(G, dG, PX=None, degeneracy=True, max_branch_depth=100000, approx=True):
    assert G.shape == dG.shape
    k = G.shape[0]
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, _ = initialize_structures(k, PX=PX)

    if degeneracy:
        G_total = G + dG
        degen_order, _, _ = get_degeneracy(G_total.indices, G_total.indptr[:-1], G_total.indptr[1:], np.arange(k).astype(np.int32))
        degen_pos = np.argsort(degen_order)
        G = G[degen_order,:][:,degen_order]
        dG = dG[degen_order,:][:,degen_order]
        G.sort_indices()
        dG.sort_indices()

    dGI = np.zeros(2 * dG.indices.size, np.int32)    
    dGI[::2] = dG.indices
    dGS, dGE = 2 * dG.indptr[:-1], 2 * dG.indptr[1:]
    GI, GS, GE = G.indices, G.indptr[:-1], G.indptr[1:]

    if approx:
        start = time.time()
        cliques_approx, cover_G, tree = MC_cover_approx_py(G, dG)
        print 'MC_cover_approx_py time:', time.time() - start
        dGI[1::2] = cover_G.data

    potI = np.empty(3 * dG.indices.size, np.int32)
    potI[:] = 0
    potS, potE = dGS.copy() * 3 / 2,  dGS.copy() * 3 / 2
        
    pot_min = np.zeros(PX.size, np.int32)
    pot_min.fill(1000000)
    
    tree = np.asfortranarray(np.zeros((22, 100000), np.int32))

    start = time.time()    
    C, CP, CN, max_cover, tree = BK_dG_cover(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        GS, GE, GI,
        dGS, dGE, dGI,
        Tbuf, 0, btw_new, btw_stack, btw_end,
        potI, potS, potE, pot_min, PX.size,
        0, 0, 0, max_branch_depth,
        C, CP, CN, stats, tree)
    print 'BK_dG_cover time:', time.time() - start

    C, CP, CN = trim_cliques(C, CP, CN)
    if degeneracy:
        C = degen_order[C]
    tree = tree[:,:stats[0]]

    return C, CP, CN, tree

@jit(nopython=True, cache=cache)
def BK_Gsep_cover(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
                  GS, GE, GI, dGS, dGE, dGI,
                  Tbuf, depth,
                  potI, potS, potE, pot_min, prev_UB,
                  min_cover, max_cover, min_cover_btw, max_branch_depth,
                  C, CP, CN,
                  stats, tree):
    if tree.shape[1] == stats[0] + 2:
        tree = expand_2d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]
    
    R = Rbuf[:RE]
    P, X = PX[PS:sep], PX[sep:XE]

    # if verbose:
    #     indent = '\t' * depth
    #     print indent, '---------Gsep------'
    #     print indent, 'DEPTH:', depth
    #     print indent, 'PX:', PX
    #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    #     print indent, 'R:', R, 'P:', P, 'X:', X
    #     print indent, 'min_cover:', min_cover, 'min_cover_btw:', min_cover_btw, 'prev_UB:', prev_UB
    #     print indent, 'potS/end/indices:', sparse_str_I(potI, potS, potE)
    #     print indent, 'G:', sparse_str_I(GI, GS, GE)
    #     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
        
    if P.size==0:
        if X.size==0:
            C, CP, CN = update_cliques(C, CP, CN, R)
            tree[1, curr_node] = 0
            return C, CP, CN, R.size, tree        
        else:
            tree[1, curr_node] = 1
            return C, CP, CN, 0, tree
    
    default_max = 1000000
    UB = R.size + sep - PS
    min_cover_within = default_max
    min_cover_within_P = np.empty(P.size, np.int32)
    min_cover_within_P[:] = default_max
    
    if depth > 0: prev_r = R[RE - 1]
    else:         prev_r = -1
    
    u, max_degree = -1, -1
    P_degree = np.zeros(P.size, np.int32)
    P_degree_new = np.zeros(P.size, np.int32)
    for v_i in range(XE-1, PS-1, -1):
        v = PX[v_i]
        v_degree, curr = move_PX(GI, GS, GE, pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])

        # Move P and X to the bottom
        v_degree_new, curr_new = 0, dGS[v]
        for w_i in range(dGS[v], dGE[v], 2):
            w = dGI[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:  break
            elif PS <= w_pos and w_pos < XE:
                v_degree_new += w_pos < sep
                if v_i < sep and w_pos < sep:
                    min_cover_within_P[v_i - PS] = min(min_cover_within_P[v_i - PS], dGI[w_i+1])
                dGI[curr_new], dGI[w_i] = w, dGI[curr_new]
                dGI[curr_new+1], dGI[w_i+1] = dGI[w_i+1], dGI[curr_new+1]
                curr_new += 2

            # Accounts for the case when curr_new was incremented
            if (prev_UB >= dGI[w_i+1]) and (v_i < sep) and dGI[w_i] == prev_r:
                potI[potE[v]-1] = w_i + 1
                
        v_degree += v_degree_new
        
        if v_degree > max_degree:
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new        
        if v_i < sep:
            min_cover_within = min(min_cover_within, min_cover_within_P[v_i - PS])
            P_degree[v_i - PS] = v_degree
            P_degree_new[v_i - PS] = v_degree_new
    
    if min(prev_UB, R.size + 1 + P_degree.max()) < min(min_cover, min_cover_btw, min_cover_within):
        tree[1, curr_node] = 2
        return C, CP, CN, 0, tree    
    
    # Color the nodes
    colors = np.zeros(sep - PS, np.int32)
    nei_bool = np.zeros(sep - PS + 1, np.bool_)
    nei_list = np.empty(sep - PS, np.int32)
    max_colors = 0   
    
    for v_i in np.argsort(-1 * P_degree):
        max_colors = set_color(dGI, dGS, dGE,
                               GI, GS, GE,
                               pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)    
        
    UB = R.size + max_colors    
    if UB < min(min_cover, min_cover_within, min_cover_btw):
        tree[1, curr_node] = 3
        return C, CP, CN, 0, tree
    
    # Get bound based on X
    tmp_sizes = get_branch_sizes(dGI, dGS, dGE,
                                 GI, GS, GE,
                                 pos, sep, PS, XE, colors, nei_bool, nei_list, P)
    tmp_sizes_argsort = np.argsort(-1 * tmp_sizes)    
    best = default_max
    for v in PX[sep:XE]:
        X_keep = np.ones(sep - PS, np.bool_)
        for w in GI[GS[v] : GE[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:  break
            elif w_pos < sep:              X_keep[w_pos - PS] = False
        for w in dGI[dGS[v] : dGE[v] : 2]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:  break
            elif w_pos < sep:              X_keep[w_pos - PS] = False
        for w_i in tmp_sizes_argsort:
            if X_keep[w_i]:
                best = min(best, tmp_sizes[w_i])
                break
    if R.size + 1 + best < min(min_cover, min_cover_within, min_cover_btw):
        tree[1, curr_node] = 4
        return C, CP, CN, 0, tree

    # # Identify branches for which there may be a better cover of an internal edge
    # branches = np.empty(P.size, PX.dtype)
    # branches_i = 0
    # for u_i in range(P.size):
    #     u = P[u_i]
    #     within_purp = (R.size + 1 + tmp_sizes[u_i]) >= min_cover_within_P[u_i]
    #     btw_purp = (potS[u]<potE[u]) and (potI[potE[u]-2] <= (R.size+1+tmp_sizes[u_i]))
    #     if within_purp or btw_purp:        
    #         branches[branches_i] = u
    #         branches_i += 1
    # branches = branches[:branches_i]

    # Identify branches for which there may be a better cover of an internal edge
    see_purp = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
    if see_purp:
        within_purp = (R.size + 1 + tmp_sizes) >= min_cover_within_P
        btw_purp = np.zeros(P.size, np.bool_)
        tmp_btw_purp = (potE[P] > potS[P]).nonzero()[0]
        btw_purp[tmp_btw_purp[potI[potE[P[tmp_btw_purp]] - 2] <= (R.size + 1 + tmp_sizes[tmp_btw_purp])]] = True
        P_purp = within_purp | btw_purp
        if not np.any(P_purp):
            tree[1, curr_node] = 5
            return C, CP, CN, 0, tree
        
        branches = P[P_purp]
        branches_degree = P_degree[P_purp]
    else:
        branches = P.copy()
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
    
        min_newly_capt = default_max
        sub_min_cover_btw = default_max
        new_PS, new_XE = sep, sep
        
        tmp = get_branch_sizes_vw(dGI, dGS, dGE,
                                  GI, GS, GE,
                                  pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
        sub_UB = R.size + 1 + tmp
        if sub_UB >= min(min_cover, min_cover_within, min_cover_btw):
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
                
        for w_i in range(dGS[v], dGE[v], 2):
            w = dGI[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)
                if sub_UB >= dGI[w_i+1]:
                    potI[potE[w]] = dGI[w_i+1]
                    if potE[w] > potS[w]:
                        potI[potE[w]+1] = min(potI[potE[w]-2], dGI[w_i+1])
                    else:
                        potI[potE[w]+1] = dGI[w_i+1]
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
            do_subbranch = min(sub_UB, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_btw)
        else:
            do_subbranch = min(sub_UB, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_btw)
        
        if do_subbranch:
            Rbuf[RE] = v
            sub_min_cover = min(min_cover, min_newly_capt)

            prev_CN = CN

            tree[0, stats[0]+1] = curr_node
            C, CP, CN, sub_max_cover, tree = BK_Gsep_cover(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                dGS, dGE, dGI,
                Tbuf, depth+1,
                potI, potS, potE, pot_min, sub_UB,
                sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
                C, CP, CN, stats, tree)

            # if verbose:
            #     print indent, 'Ending branching at:', v
            #     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
            #     print indent, 'G:', sparse_str_I(GI, GS, GE)
            #     print indent, 'PX:', PX, 'P:', PX[PS:sep], 'X:', PX[sep:XE], 'PS,sep,XE:', PS, sep, XE

            # Update min_cover, max_cover
            max_cover = max(max_cover, sub_max_cover)
            min_cover = max(min_cover, sub_max_cover)
            
            if sub_max_cover > min_newly_capt:
                for w_i in range(0, newly_capt.size, 3):
                    newly_capt[w_i] = max(newly_capt[w_i], sub_max_cover)
                    newly_capt[w_i+1] = max(newly_capt[w_i+1], sub_max_cover)

        restore_pot(dGI, dGS, dGE, potI, potS, potE,
                    pos, PS, sep, XE, sub_UB, v, do_subbranch)
                
        # Recompute min_cover_within
        if do_subbranch and sub_max_cover > min_cover_within:
            min_cover_within = recompute_min_cover_within(dGI, dGS, dGE, PX, pos, PS, sep, XE)
    
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

    # if verbose:
    #     print indent, 'Returning::::'
    #     # print indent, 'R:', R, 'P:', P, 'X:', X
    #     # print indent, 'pot_min:', pot_min
    #     # print indent, 'min/max_cover:', min_cover, max_cover
    #     # print indent, 'potS/end/indices:', sparse_str_I(potI, potS, potE)
    #     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
    #     print indent, 'G:', sparse_str_I(GI, GS, GE)

    tree[1, curr_node] = 6
    return C, CP, CN, max_cover, tree

@jit(nopython=True, cache=cache)
def BK_dG_cover(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
                GS, GE, GI, dGS, dGE, dGI,
                Tbuf, depth,
                btw_new, btw_stack, btw_end,
                potI, potS, potE, pot_min, prev_UB,
                min_cover, max_cover, min_cover_btw, max_branch_depth,
                C, CP, CN, stats, tree):
    """
    btw_new[v]: Is there a new edge crossing btw R and node v in P?

    Suppose R = (r1,r2,...,rn)
    For some v in P, then pot[v] has three values:
    (1) max cover found so far for (rn, v), if that's a new edge
    (2) rolling max cover found so far for among the new edges in (r1,v),(r2,v),...,(rn,v)
    (3) The index in G_i + 1

    max_cover = maximum of the per-edge maximum cliques found in a branch
    min_cover = minium of the per-edge maximum cliques found in a branch

    """
    if tree.shape[1] == stats[0] + 2:
        tree = expand_2d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]

    R = Rbuf[:RE]
    P, X = PX[PS:sep], PX[sep:XE]

    # if verbose:
    #     indent = '\t' * depth
    #     print indent, '--------dG-------'
    #     print indent, 'DEPTH:', depth
    #     print indent, 'PX:', PX
    #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    #     print indent, 'R:', R, 'P:', P, 'X:', X
    #     print indent, 'min_cover:', min_cover, 'min_cover_btw:', min_cover_btw, 'prev_UB:', prev_UB
    #     print indent, 'btw_new:', btw_new.astype(np.int32), 'pot_min:', pot_min
    #     print indent, 'potS/end/indices:', sparse_str_I(potI, potS, potE)
    #     print indent, 'G:', sparse_str_I(GI, GS, GE)
    #     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
    
    if P.size==0:
        tree[1, curr_node] = 1
        return C, CP, CN, 0, tree

    incd = np.empty(sep - PS, np.int32)
    incd_degree = np.empty(sep - PS, np.int32)
    incd_count = 0
    X_incd = np.empty(XE - sep, np.int32)
    X_incd_count = 0
    
    default_max = 1000000
    UB = R.size + (sep - PS)
    min_cover_within = default_max
    min_cover_within_incd = np.empty(P.size, np.int32)
    min_cover_within_incd[:] = default_max
    
    if depth > 0: prev_r = R[RE - 1]
    else:         prev_r = -1
    
    u = -1
    max_degree = -1
    P_degree = np.zeros(sep - PS, np.int32)
    P_degree_new = np.zeros(sep - PS, np.int32)
            
    for v_i in range(XE-1, PS-1, -1):
        v, inP = PX[v_i], v_i < sep        
        
        min_cover_within_v = default_max
        
        # Move P and X to the bottom
        v_degree_new, curr_new = 0, dGS[v]
        for w_i in range(dGS[v], dGE[v], 2):
            w = dGI[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_new += w_pos < sep
                if inP and w_pos < sep:
                    min_cover_within_v = min(min_cover_within_v, dGI[w_i+1])
                dGI[curr_new], dGI[w_i] = w, dGI[curr_new]
                dGI[curr_new+1], dGI[w_i+1] = dGI[w_i+1], dGI[curr_new+1]
                curr_new += 2
            
            if inP and (prev_UB >= dGI[w_i+1]) and dGI[w_i] == prev_r:
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
    
    if min(prev_UB, R.size + (sep - PS)) < min(min_cover, min_cover_btw, min_cover_within):
        tree[1, curr_node] = 2
        return C, CP, CN, 0, tree
    
    if incd_count == 0:
        tree[1, curr_node] = 6
        return C, CP, CN, 0, tree
    incd = incd[:incd_count]
    min_cover_within_incd = min_cover_within_incd[:incd_count]
    incd_degree = incd_degree[:incd_count]
    X_incd = X_incd[:X_incd_count]

    Tbuf[incd] = False
    Tbuf[X_incd] = False
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
                
                if Tbuf[w]:
                    new_incd[new_incd_end] = w
                    new_incd_end += 1
                    Tbuf[w] = False        
        P_degree[pos[v] - PS] += v_degree_old

    Tbuf[incd] = True
    Tbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    Tbuf[new_incd] = True
    
    Tbuf[incd] = False
    Tbuf[new_incd] = False
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
                v_degree += (w_pos < sep) and not Tbuf[w]
                GI[curr], GI[w_i] = w, GI[curr]
                curr += 1
        if pos[v] < sep:
            P_degree[pos[v] - PS] += v_degree
    Tbuf[incd] = True
    Tbuf[new_incd] = True
    
    #--------------------------------#
    
    # Color the nodes
    colors = np.zeros(sep - PS, np.int32)
    nei_bool = np.zeros(sep - PS + 1, np.bool_)
    nei_list = np.empty(sep - PS, np.int32)
    max_colors = 0

    Tbuf[incd] = False
    Tbuf[new_incd] = False
    
    # Recompute P_degree    
    for v_i in range(sep, PS):
        v = PX[v_i]
        v_degree = 0
        for w in dGI[dGS[v] : dGE[v] : 2]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            v_degree += (w_pos < sep) and (not Tbuf[w])
        for w in GI[GS[v] : GE[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            v_degree += (w_pos < sep) and (not Tbuf[w])
        P_degree[v_i] = v_degree
                
    for v_i in np.argsort(-1 * P_degree):
        if not Tbuf[P[v_i]]:
            max_colors = set_color(dGI, dGS, dGE, GI, GS, GE,
                                   pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)
    Tbuf[incd] = True
    Tbuf[new_incd] = True
        
    UB = R.size + max_colors
    if UB < min(min_cover, min_cover_within, min_cover_btw):
        tree[1, curr_node] = 3
        return C, CP, CN, 0, tree
        
    tmp_sizes = get_branch_sizes(dGI, dGS, dGE, GI, GS, GE, pos, sep, PS, XE, colors, nei_bool, nei_list, incd)
    see_purp = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
    if see_purp:
        within_purp = (R.size + 1 + tmp_sizes) >= min_cover_within_incd
        btw_purp = np.zeros(incd.size, np.bool_)
        tmp_btw_purp = (potE[incd] > potS[incd]).nonzero()[0]
        btw_purp[tmp_btw_purp[potI[potE[incd[tmp_btw_purp]] - 2] <= (R.size + 1 + tmp_sizes[tmp_btw_purp])]] = True
        branches_purp = within_purp | btw_purp
        if not np.any(branches_purp):
            tree[1, curr_node] = 4
            return C, CP, CN, 0, tree
    else:
        branches_purp = np.zeros(incd.size, np.bool_)
        
    ## Get branches
    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = dGI[dGS[u] : u_curr_new : 2]        
    Tbuf[Padj_u] = False
    Tbuf[Padj_new_u] = False
    
    ### Consideration: maybe btw_new[P] should only be included if the pivot is in P?    
    # Always keep the btw_new
    Tbuf[P[btw_new[P]]] = True
    
    # Ensure that for any node left out of P, all of its dG neighbors in P are included
    for v in incd[~ Tbuf[incd]]:
        if not Tbuf[v]:
            # By construction, this will only iterate over w's that are in incd
            for w in dGI[dGS[v] : dGE[v]: 2]:
                w_pos = pos[w]
                Tbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
    branches = incd[Tbuf[incd]]
    if see_purp:
        extra_priority = branches_purp[Tbuf[incd]]
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
            if PS <= w_pos and w_pos < sep and incd_bool[w] and (not Tbuf[w]) and colors[w_pos - PS] > v_color:
                w_col = colors[w_pos - PS]
                if not distinct_bool[w_col]:
                    distinct_bool[w_col] = True
                    distinct[distinct_count] = w_col
                    distinct_count += 1
            elif PS > w_pos or w_pos >= XE:
                break
        for w in dGI[dGS[v] : dGE[v] : 2]:
            w_pos = pos[w]
            if PS <= w_pos and w_pos < sep and incd_bool[w] and (not Tbuf[w]) and colors[w_pos - PS] > v_color:
                w_col = colors[w_pos - PS]
                if not distinct_bool[w_col]:
                    distinct_bool[w_col] = True
                    distinct[distinct_count] = w_col
                    distinct_count += 1
            elif PS > w_pos or w_pos >= XE:
                break
        distinct_bool[distinct[:distinct_count]] = False
        branches_sizes[v_i] += distinct_count
        
    Tbuf[Padj_new_u] = True
    Tbuf[Padj_u] = True

    #####
    ## Strongly prioritize branching off btw_new first
#     branches_order = np.argsort(-1 * (branches + (1000000 * branches_sizes)))
    branches_order = np.argsort(-1 * (branches_sizes + (default_max * extra_priority)))
    branches = branches[branches_order]
    branches_sizes = branches_sizes[branches_order]
    
#     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
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
    
    used = np.empty(branches.size, np.int32)
    used_count = 0
    
    colors_v = np.empty(PX.size, np.int32)
    colors_v[P] = colors    
    
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
                else:
                    break
            else:
                v = branches[v_i]
                
                found = False
                for w_i in range(dGS[v], dGE[v], 2):
                    w = dGI[w_i]
                    w_pos = pos[w]
                    if (PS <= w_pos) and (w_pos < sep):
                        if dGI[w_i+1]==0:
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
        tmp = get_branch_sizes_vw(dGI, dGS, dGE,
                                  GI, GS, GE,
                                  pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
        sub_UB = R.size + 1 + tmp

        # if verbose:
        #     print indent, 'branching at:', v
        #     print indent, 'pot_min:', pot_min
        #     print indent, 'min/max_cover:', min_cover, max_cover        
        #     print indent, 'potS/end/indices:', sparse_str_I(potI, potS, potE)
        #     print indent, 'potS/end:', zip(potS, potE)
        #     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
        #     print indent, 'G:', sparse_str_I(GI, GS, GE)
        #     #print indent, 'sub_UB:', sub_UB, 'min(min_cover, min_cover_within, min_cover_btw):', min(min_cover, min_cover_within, min_cover_btw)
        #     print indent, 'PX:', PX, 'P:', PX[PS:sep], 'X:', PX[sep:XE], 'PS,sep,XE:', PS, sep, XE

        if sub_UB >= min(min_cover, min_cover_within, min_cover_btw):
            used[used_count] = v
            used_count += 1
        else:
            continue
            
        min_newly_capt, sub_min_cover_btw, sub_min_cover_within = default_max, default_max, default_max
        btw_added = 0
        new_PS, new_XE = sep, sep
        
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
        
        for w_i in range(dGS[v], dGE[v], 2):
            w = dGI[w_i]
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)               
                if not btw_new[w]:
                    btw_stack[btw_end + btw_added] = w
                    btw_added += 1
                    btw_new[w] = True

                if sub_UB >= dGI[w_i+1]:
                    potI[potE[w]] = dGI[w_i+1]
                    if potE[w] > potS[w]:
                        potI[potE[w]+1] = min(potI[potE[w]-2], dGI[w_i+1])
                    else:
                        potI[potE[w]+1] = dGI[w_i+1]
                    potE[w] += 3
                if potE[w] > potS[w]:
                    sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < PS or w_pos >= XE:
                break
            
        # if verbose:
        #     print indent, 'PX:', PX, 'P:', PX[PS:sep], 'X:', PX[sep:XE], 'PS,sep,XE:', PS, sep, XE, 'newPS, newXE:', new_PS, new_XE
        
        # Update min cover based on r4: r1,r2,r3
        newly_capt = potI[potS[v] : potE[v]]
        if newly_capt.size > 0:
            min_newly_capt = newly_capt[newly_capt.size - 2]
        
        ## Don't need to apply this criterion yet (since dG assumes no new edge yet)
        ## However, if you do, then do it on a min_below computes by PX-by-PX looping above
        if sep - new_PS == 0:
            do_subbranch = min(sub_UB, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_btw)
        else:
            do_subbranch = min(sub_UB, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_btw)
        
        if do_subbranch:
            Rbuf[RE] = v
            sub_min_cover = min(min_cover, min_newly_capt)
            
            prev_CN = CN
            tree[0, stats[0]+1] = curr_node
            
            if btw_new[v]:
                C, CP, CN, sub_max_cover, tree = BK_Gsep_cover(
                    Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                    GS, GE, GI,
                    dGS, dGE, dGI,
                    Tbuf, depth+1,
                    potI, potS, potE, pot_min, sub_UB,
                    sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
                    C, CP, CN, stats, tree)
            else:
                C, CP, CN, sub_max_cover, tree = BK_dG_cover(
                    Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                    GS, GE, GI,
                    dGS, dGE, dGI,
                    Tbuf, depth+1,
                    btw_new, btw_stack, btw_end + btw_added,
                    potI, potS, potE, pot_min, sub_UB,
                    sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
                    C, CP, CN, stats, tree)

            # if verbose:
            #     print indent, 'Ending branching at:', v
            #     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
            #     print indent, 'G:', sparse_str_I(GI, GS, GE)
            #     print indent, 'PX:', PX, 'P:', PX[PS:sep], 'X:', PX[sep:XE], 'PS,sep,XE:', PS, sep, XE

            # Update min_cover, max_cover
            max_cover = max(max_cover, sub_max_cover)
            min_cover = max(min_cover, sub_max_cover)
            
            if sub_max_cover > min_newly_capt:
                # Update max cover of (r4,r1), (r4,r2), (r4,r3), which we will move back to potential, and restore captured.
                # Fresh updates to max cover are moved to accum r1:...r4, r2:...r4, r3:...r4
                for w_i in range(0, newly_capt.size, 3):
                    newly_capt[w_i] = max(newly_capt[w_i], sub_max_cover)
                    newly_capt[w_i+1] = max(newly_capt[w_i+1], sub_max_cover)
        
        restore_pot(dGI, dGS, dGE, potI, potS, potE,
                    pos, PS, sep, XE, sub_UB, v, do_subbranch)

        # Recompute min_cover_within
        if do_subbranch and sub_max_cover > min_cover_within:
            min_cover_within = recompute_min_cover_within(dGI, dGS, dGE, PX, pos, PS, sep, XE)
        
        # Reset the btw_new
        btw_new[btw_stack[btw_end : btw_end + btw_added]] = False

        # Swap v to the end of P, and then decrement separator
        sep -= 1 ; swap_pos(PX, pos, v, sep)
        
        # Don't branch anymore after this
        if depth > max_branch_depth and used_count==1:  break

    for v in used[:used_count][::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    # if verbose:
    #     print indent, 'Returning::::'
    #     # print indent, 'R:', R, 'P:', P, 'X:', X
    #     # print indent, 'pot_min:', pot_min
    #     # print indent, 'min/max_cover:', min_cover, max_cover
    #     # print indent, 'potS/end/indices:', sparse_str_I(potI, potS, potE)
    #     print indent, 'dG:', sparse_str_I(dGI, dGS, dGE)
    #     print indent, 'G:', sparse_str_I(GI, GS, GE)
                
    tree[1, curr_node] = 6
    return C, CP, CN, max_cover, tree



####################
# Old
####################

# def BK_dG_cover_py(G, dG, PX=None, degeneracy=None, max_branch_depth=100000):
#     k = G.shape[0]
#     PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, _ = initialize_structures(k, PX=PX)
   
#     GI_new = np.empty(2 * dG.indices.size, np.int32)    
#     GI_new[1::2] = 0
    
#     if degeneracy in ['min', 'max']:
#         assert k == dG.shape[0]
# #         G = G + dG
#         if degeneracy=='max':
#             degen_order, degen_deg = get_degeneracy_max(dG.indptr[:-1], dG.indptr[1:], dG.indices)
#         if degeneracy=='min':
# #             degen_order, degen_deg = get_degeneracy_min(G.indptr[:-1], G.indptr[1:], G.indices)
#             degen_order, degen_deg = get_degeneracy_min(dG.indptr[:-1], dG.indptr[1:], dG.indices)
#         degen_pos = np.empty(k, np.int32)
#         degen_pos[degen_order] = np.arange(k).astype(np.int32)
#         GI_new[::2] = degen_pos[dG.indices]
#         GS_new = 2 * dG.indptr[:-1][degen_order]
#         GE_end = 2 * dG.indptr[1:][degen_order]
        
#         G_indices = degen_pos[G.indices]
#         G_start, G_end = G.indptr[:-1][degen_order], G.indptr[1:][degen_order]
#     else:
#         GI_new[::2] = dG.indices
#         GS_new = 2 * dG.indptr[:-1]
#         GE_end = 2 * dG.indptr[1:]        
#         G_start, G_end, G_indices = G.indptr[:-1], G.indptr[1:], G.indices
    
#     potI = np.empty(3 * dG.indices.size, np.int32)
#     potI[:] = 0
#     potS = GS_new.copy() * 3 / 2
#     potE = GS_new.copy() * 3 / 2
        
#     pot_min = np.zeros(PX.size, np.int32)
#     pot_min.fill(1000000)
    
#     min_cover, max_cover, min_cover_btw = 0, 0, 0
    
#     max_possible = PX.size
    
#     tree = np.asfortranarray(np.zeros((22, 100000), np.int32))
#     tree.fill(-1)
#     tree[0, :2] = np.array([0,0])
    
#     C, CP, CN, max_cover, tree = BK_dG_cover(
#         R, RE, PX, PS, sep, XE, PS, XE, pos,
#         G_start, G_end, G_indices,
#         GS_new, GE_end, GI_new,
#         Tbuf, 0, btw_new, btw_stack, btw_end,
#         potI, potS, potE, pot_min, max_possible,
#         min_cover, max_cover, min_cover_btw, max_branch_depth,
#         C, CP, CN, tree)

#     C, CP, CN = trim_cliques(C, CP, CN)
    
#     if degeneracy in ['min', 'max']:
#         C = degen_order[C]
        
#     tree = tree[:, : 4 + tree[0,0]]

#     return C, CP, CN, tree

# @jit(nopython=True, cache=cache)
# def BK_Gsep_cover(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
#                   GS, GE, GI, GS_new, GE_end, GI_new,
#                   Tbuf, depth,
#                   potI, potS, potE, pot_min, prev_max_possible,
#                   min_cover, max_cover, min_cover_btw, max_branch_depth,
#                   C, CP, CN, tree):

#     orig_size = 4 + tree[0, 0]
#     curr_tree = 4 + tree[0, 0]    
#     if tree.shape[1] <= curr_tree:
#         tree = expand_2d_arr(tree)    
#     tree[0, 0] += 1
    
#     R = Rbuf[:RE]
#     P, X = PX[PS:sep], PX[sep:XE]

#     # if verbose:
#     #     indent = '\t' * depth
#     #     print indent, '---------Gsep------'
#     #     print indent, 'DEPTH:', depth
#     #     print indent, 'PX:', PX
#     #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     #     print indent, 'R:', R, 'P:', P, 'X:', X
#     #     print indent, 'min_cover:', min_cover, 'min_cover_btw:', min_cover_btw, 'prev_max_possible:', prev_max_possible
#     #     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
#     #     print indent, 'G:', [(i, x, y, GI[x:y].tolist()) for i, (x, y) in enumerate(zip(GS, GE))]
#     #     print indent, 'dG:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
        
#     if P.size==0:
#         tree[0, 1] += 1
#         if X.size==0:
#             tree[12, curr_tree] = 1
#             R_size = R.size

#             # if verbose: print((1111111, R[0], curr_tree, depth, min_cover))
#             C, CP, CN = update_cliques(C, CP, CN, R)            
#             return C, CP, CN, R.size, tree        
#         else:
#             tree[12, curr_tree] = 2
#             return C, CP, CN, 0, tree
    
#     default_max = 1000000
#     max_possible = R.size + sep - PS
#     min_cover_within = default_max
#     min_cover_within_P = np.empty(P.size, np.int32)
#     min_cover_within_P[:] = default_max
    
#     if depth > 0: prev_r = R[RE - 1]
#     else:         prev_r = -1
    
#     u = -1
#     max_degree = -1
#     P_degree = np.zeros(sep - PS, np.int32)
#     P_degree_new = np.zeros(sep - PS, np.int32)
#     for v_i in range(XE-1, PS-1, -1):
#         v = PX[v_i]
#         v_degree, curr = move_PX(GI, GS, GE, pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])

#         # Move P and X to the bottom
#         v_degree_new, curr_new = 0, GS_new[v]
#         for w_i in range(GS_new[v], GE_end[v], 2):
#             w = GI_new[w_i]
#             w_pos = pos[w]
#             if w_pos < oldPS or w_pos >= oldXE:
#                 break
#             elif PS <= w_pos and w_pos < XE:
#                 v_degree_new += w_pos < sep
#                 if v_i < sep and w_pos < sep:
#                     min_cover_within_P[v_i - PS] = min(min_cover_within_P[v_i - PS], GI_new[w_i+1])
#                 GI_new[curr_new], GI_new[w_i] = w, GI_new[curr_new]
#                 GI_new[curr_new+1], GI_new[w_i+1] = GI_new[w_i+1], GI_new[curr_new+1]
#                 curr_new += 2

#             # Accounts for the case when curr_new was incremented
#             if (prev_max_possible >= GI_new[w_i+1]) and (v_i < sep) and GI_new[w_i] == prev_r:
#                 potI[potE[v]-1] = w_i + 1
                
#         v_degree += v_degree_new
        
#         if v_degree > max_degree:
#             max_degree = v_degree
#             u, u_curr, u_curr_new = v, curr, curr_new        
#         if v_i < sep:
#             min_cover_within = min(min_cover_within, min_cover_within_P[v_i - PS])
#             P_degree[v_i - PS] = v_degree
#             P_degree_new[v_i - PS] = v_degree_new
    
#     if min(prev_max_possible, R.size + 1 + P_degree.max()) < min(min_cover, min_cover_btw, min_cover_within):
#         return C, CP, CN, 0, tree    
    
#     branches = P.copy()

#     # Color the nodes
#     colors = np.zeros(sep - PS, np.int32)
#     nei_bool = np.zeros(sep - PS + 1, np.bool_)
#     nei_list = np.empty(sep - PS, np.int32)
#     max_colors = 0   
    
#     for v_i in np.argsort(-1 * P_degree):
#         max_colors = set_color(GI_new, GS_new, GE_end,
#                                GI, GS, GE,
#                                pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)    
        
#     max_possible = R.size + max_colors    
#     if max_possible < min(min_cover, min_cover_within, min_cover_btw):
#         return C, CP, CN, 0, tree
    
#     # Get bound based on X
#     tmp_sizes = get_branch_sizes(GI_new, GS_new, GE_end,
#                                  GI, GS, GE,
#                                  pos, sep, PS, XE, colors, nei_bool, nei_list, P)
#     tmp_sizes_argsort = np.argsort(-1 * tmp_sizes)    
#     best = 10000
#     for v in PX[sep:XE]:
#         X_keep = np.ones(sep - PS, np.bool_)
#         for w in GI[GS[v] : GE[v]]:
#             w_pos = pos[w]
#             if w_pos < PS or w_pos >= XE:
#                 break
#             elif w_pos < sep:
#                 X_keep[w_pos - PS] = False
#         for w in GI_new[GS_new[v] : GE_end[v] : 2]:
#             w_pos = pos[w]
#             if w_pos < PS or w_pos >= XE:
#                 break
#             elif w_pos < sep:
#                 X_keep[w_pos - PS] = False
#         for w_i in tmp_sizes_argsort:
#             if X_keep[w_i]:
#                 best = min(best, tmp_sizes[w_i])
#                 break

#     if R.size + 1 + best < min(min_cover, min_cover_within, min_cover_btw):
#         return C, CP, CN, 0, tree
    
#     see_purpose = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
#     if see_purpose:
#         within_purpose = (R.size + 1 + tmp_sizes) >= min_cover_within_P
#         btw_purpose = np.zeros(P.size, np.bool_)
#         tmp_btw_purpose = (potE[P] > potS[P]).nonzero()[0]
#         btw_purpose[tmp_btw_purpose[potI[potE[P[tmp_btw_purpose]] - 2] <= (R.size + 1 + tmp_sizes[tmp_btw_purpose])]] = True
#         P_purpose = within_purpose | btw_purpose
#         if not np.any(P_purpose):
#             return C, CP, CN, 0, tree
        
#         branches = P[P_purpose]
#         branches_degree = P_degree[P_purpose]
#     else:
#         branches_degree = P_degree
    
# #     print indent, 'branches:', branches
# #     print indent, 'colors:', colors[pos[branches] - PS]
# #     print indent, 'branches_sizes:', branches_sizes
# #     print indent, 'min(min_cover, min_cover_within, min_cover_btw):', min(min_cover, min_cover_within, min_cover_btw)
# #     print indent, 'pivot u:', u
# #     print indent, 'Padj_u:', Padj_u
# #     print indent, 'branches:', branches
    
#     # Initialize max_cover
#     max_cover = 0
#     init_CN = CN

#     used = np.empty(branches.size, np.int32)
#     used_count = 0
    
#     colors_v = np.empty(PX.size, np.int32)
#     colors_v[P] = colors
       
#     v_i = -1
#     while True:
#         v_i += 1        
#         if v_i==branches.size:
#             break
        
#         v = branches[v_i]
    
#         min_newly_capt = 1000000
#         sub_min_cover_btw = 1000000
#         new_PS, new_XE = sep, sep
        
#         tmp = get_branch_sizes_vw(GI_new, GS_new, GE_end,
#                                   GI, GS, GE,
#                                   pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
#         sub_max_possible = R.size + 1 + tmp
#         if sub_max_possible >= min(min_cover, min_cover_within, min_cover_btw):
#             used[used_count] = v
#             used_count += 1
#         else:
#             continue
            
#         for w_i in range(GS[v], GE[v]):
#             w = GI[w_i]
#             w_pos = pos[w]
#             if (PS <= w_pos) and (w_pos < sep):
#                 new_PS -= 1
#                 swap_pos(PX, pos, w, new_PS)
                
#                 if potE[w] > potS[w]:
#                     sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])
#             elif (sep <= w_pos) and (w_pos < XE):
#                 swap_pos(PX, pos, w, new_XE)
#                 new_XE += 1
#             else:
#                 break
                
#         for w_i in range(GS_new[v], GE_end[v], 2):
#             w = GI_new[w_i]
#             w_pos = pos[w]
#             if (PS <= w_pos) and (w_pos < sep):
#                 new_PS -= 1
#                 swap_pos(PX, pos, w, new_PS)
#                 if sub_max_possible >= GI_new[w_i+1]:
#                     potI[potE[w]] = GI_new[w_i+1]
#                     if potE[w] > potS[w]:
#                         potI[potE[w]+1] = min(potI[potE[w]-2], GI_new[w_i+1])
#                     else:
#                         potI[potE[w]+1] = GI_new[w_i+1]
#                     potE[w] += 3
                
#                 if potE[w] > potS[w]:
#                     sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])
    
#             elif (sep <= w_pos) and (w_pos < XE):
#                 swap_pos(PX, pos, w, new_XE)
#                 new_XE += 1
#             elif w_pos < PS or w_pos >= XE:
#                 break
  
#         newly_capt = potI[potS[v] : potE[v]]
#         if newly_capt.size > 0:
#             min_newly_capt = newly_capt[newly_capt.size - 2]        

#         # Not necessarily true as branches are moved to X
#         if sep - new_PS == 0:
#             do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_btw)
#         else:
#             do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_btw)
        
#         if do_subbranch:
#             sub_tree = 4 + tree[0,0]
#             if tree.shape[1] <= sub_tree:
#                 tree = expand_2d_arr(tree)

#         if do_subbranch:
#             Rbuf[RE] = v
#             sub_min_cover = min(min_cover, min_newly_capt)

#             prev_CN = CN
            
#             C, CP, CN, sub_max_cover, tree = BK_Gsep_cover(
#                 Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
#                 GS, GE, GI,
#                 GS_new, GE_end, GI_new,
#                 Tbuf, depth+1,
#                 potI, potS, potE, pot_min, sub_max_possible,
#                 sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
#                 C, CP, CN, tree)
                
#             # Update min_cover, max_cover
#             max_cover = max(max_cover, sub_max_cover)
#             min_cover = max(min_cover, sub_max_cover)
            
#             if sub_max_cover > min_newly_capt:
#                 for w_i in range(0, newly_capt.size, 3):
#                     newly_capt[w_i] = max(newly_capt[w_i], sub_max_cover)
#                     newly_capt[w_i+1] = max(newly_capt[w_i+1], sub_max_cover)

#         # Restore pot[w] for all new neighbors of v
#         for w_i in range(GS_new[v], GE_end[v], 2):
#             w = GI_new[w_i]
#             w_pos = pos[w]
#             if (PS <= w_pos) and (w_pos < sep):
#                 if sub_max_possible >= GI_new[w_i+1]:
#                     if do_subbranch:
# #                         assert GI_new[w_i+1] == GI_new[potI[potE[w]-1]]
#                         GI_new[w_i+1] = max(GI_new[w_i+1], potI[potE[w]-3])
#                         GI_new[potI[potE[w]-1]] = max(GI_new[potI[potE[w]-1]], potI[potE[w]-3])
            
#                     potE[w] -= 3
#             elif w_pos < PS or w_pos >= XE:
#                 break

#         # Recompute min_cover_within
#         if do_subbranch and sub_max_cover > min_cover_within:
#             min_cover_within = 1000000
#             for x_i in range(sep-1, PS-1, -1):
#                 x = PX[x_i]
#                 # Move P and X to the bottom
#                 for w_i in range(GS_new[x], GE_end[x], 2):
#                     w = GI_new[w_i]
#                     w_pos = pos[w]
#                     if w_pos < PS or w_pos >= XE:
#                         break
#                     elif PS <= w_pos and w_pos < sep:
#                         min_cover_within = min(min_cover_within, GI_new[w_i+1])
    
#         # Swap v to the end of P, and then decrement separator
#         sep -= 1
#         swap_pos(PX, pos, v, sep)
        
#         # Don't branch anymore after this
#         if depth > max_branch_depth and used_count==1:
#             break

#     for v in used[:used_count][::-1]:
#         # Move v to the beginning of X and increment separator
#         swap_pos(PX, pos, v, sep)
#         sep += 1
        
# #     print indent, 'Returning::::'
# #     print indent, 'R:', R, 'P:', P, 'X:', X
# #     print indent, 'min/max_cover:', min_cover, max_cover
# #     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
# #     print indent, 'dG:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]  
# #     assert np.all(potE >= potS)

#     return C, CP, CN, max_cover, tree

# @jit(nopython=True, cache=cache)
# def BK_dG_cover(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
#                 GS, GE, GI, GS_new, GE_end, GI_new,
#                 Tbuf, depth,
#                 btw_new, btw_stack, btw_end,
#                 potI, potS, potE, pot_min, prev_max_possible,
#                 min_cover, max_cover, min_cover_btw, max_branch_depth,
#                 C, CP, CN, tree):
#     """
#     btw_new[v]: Is there a new edge crossing btw R and node v in P?
#     """
#     curr_tree = 4 + tree[0, 0]
#     if tree.shape[1] <= curr_tree:
#         tree = expand_2d_arr(tree)    
#     tree[10, curr_tree] = 1
#     tree[0, curr_tree] = depth
#     tree[0, 0] += 1
    
#     # if verbose and (curr_tree % 25000) == 0:
#     #     print((3333333, curr_tree))
    
#     # if curr_tree >= 90000:
#     #     return C, CP, CN, max_cover, tree
    
#     R = Rbuf[:RE]
#     P, X = PX[PS:sep], PX[sep:XE]

#     ## When this is depth==2, considering forcing the next v's (the ones that will occupy R[2]) to have a new edge with R[1].

#     # if verbose:
#     #     indent = '\t' * depth
#     #     print indent, '--------dG-------'
#     #     print indent, 'DEPTH:', depth
#     #     print indent, 'PX:', PX
#     #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     #     print indent, 'R:', R, 'P:', P, 'X:', X
#     #     print indent, 'min_cover:', min_cover, 'min_cover_btw:', min_cover_btw, 'prev_max_possible:', prev_max_possible
#     #     print indent, 'btw_new:', btw_new.astype(np.int32)
#     #     print indent, 'pot_min:', pot_min
#     #     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
#     #     print indent, 'G:', [(i, x, y, GI[x:y].tolist()) for i, (x, y) in enumerate(zip(GS, GE))]
#     #     print indent, 'dG:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
    
#     if P.size==0:
#         tree[12, curr_tree] = 2
#         tree[0, 1] += 1
#         return C, CP, CN, 0, tree

#     incd = np.empty(sep - PS, np.int32)
#     incd_degree = np.empty(sep - PS, np.int32)
#     incd_count = 0
#     X_incd = np.empty(XE - sep, np.int32)
#     X_incd_count = 0
    
#     default_max = 1000000
#     max_possible = R.size + (sep - PS)
#     min_cover_within = default_max
#     min_cover_within_incd = np.empty(P.size, np.int32)
#     min_cover_within_incd[:] = default_max
    
#     if depth > 0: prev_r = R[RE - 1]
#     else:         prev_r = -1
    
#     u = -1
#     max_degree = -1
#     P_degree = np.zeros(sep - PS, np.int32)
#     P_degree_new = np.zeros(sep - PS, np.int32)
            
#     for v_i in range(XE-1, PS-1, -1):
#         v, inP = PX[v_i], v_i < sep        
        
#         min_cover_within_v = default_max
        
#         # Move P and X to the bottom
#         v_degree_new, curr_new = 0, GS_new[v]
#         for w_i in range(GS_new[v], GE_end[v], 2):
#             w = GI_new[w_i]
#             w_pos = pos[w]
#             if w_pos < oldPS or w_pos >= oldXE:
#                 break
#             elif PS <= w_pos and w_pos < XE:
#                 v_degree_new += w_pos < sep
#                 if inP and w_pos < sep:
#                     min_cover_within_v = min(min_cover_within_v, GI_new[w_i+1])
#                 GI_new[curr_new], GI_new[w_i] = w, GI_new[curr_new]
#                 GI_new[curr_new+1], GI_new[w_i+1] = GI_new[w_i+1], GI_new[curr_new+1]
#                 curr_new += 2
            
#             if inP and (prev_max_possible >= GI_new[w_i+1]) and GI_new[w_i] == prev_r:
#                 potI[potE[v]-1] = w_i + 1
                
#         if v_degree_new > max_degree:
#             max_degree = v_degree_new
#             u, u_curr_new, u_incd = v, curr_new, incd_count
#         if (not inP) and v_degree_new > 0:
#             tmp, curr = move_PX(GI, GS, GE,
#                                 pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])
#             X_incd[X_incd_count] = v
#             X_incd_count += 1
#         if inP and (v_degree_new > 0 or btw_new[v]):
#             incd[incd_count] = v
#             incd_degree[incd_count] = v_degree_new            
#             min_cover_within_incd[incd_count] = min_cover_within_v            
#             incd_count += 1
            
#         if inP:
#             P_degree[v_i - PS] += v_degree_new
#             P_degree_new[v_i - PS] += v_degree_new
#             min_cover_within = min(min_cover_within, min_cover_within_v)            
#     u_curr = GE[u]
    
#     if min(prev_max_possible, R.size + (sep - PS)) < min(min_cover, min_cover_btw, min_cover_within):
#         return C, CP, CN, 0, tree
    
#     if incd_count == 0:
#         tree[12, curr_tree] = 4
#         tree[0, 1] += 1
#         return C, CP, CN, 0, tree
#     incd = incd[:incd_count]
#     min_cover_within_incd = min_cover_within_incd[:incd_count]
#     incd_degree = incd_degree[:incd_count]
#     X_incd = X_incd[:X_incd_count]

#     Tbuf[incd] = False
#     Tbuf[X_incd] = False
#     new_incd = np.empty(XE - PS - incd.size, PX.dtype)
#     new_incd_end = 0
    
#     for v in incd:
#         curr = GS[v]
#         v_degree_old = 0
        
#         # Move P and X to the bottom
#         for w_i in range(GS[v], GE[v]):
#             w = GI[w_i]
#             w_pos = pos[w]
#             if w_pos < oldPS or w_pos >= oldXE:
#                 break
#             elif PS <= w_pos and w_pos < XE:
#                 v_degree_old += w_pos < sep
#                 GI[curr], GI[w_i] = w, GI[curr]
#                 curr += 1
                
#                 if Tbuf[w]:
#                     new_incd[new_incd_end] = w
#                     new_incd_end += 1
#                     Tbuf[w] = False        
#         P_degree[pos[v] - PS] += v_degree_old

#     Tbuf[incd] = True
#     Tbuf[X_incd] = True
#     new_incd = new_incd[:new_incd_end]
#     Tbuf[new_incd] = True
    
#     Tbuf[incd] = False
#     Tbuf[new_incd] = False
#     for v in new_incd:
#         curr = GS[v]
#         v_degree = 0
#         for w_i in range(GS[v], GE[v]):
#             w = GI[w_i]
#             w_pos = pos[w]
#             if w_pos < oldPS or w_pos >= oldXE:
#                 break
#             elif PS <= w_pos and w_pos < XE:
#                 ## Counts only those neighbors that are in P and in (incd, new_incd)
#                 v_degree += (w_pos < sep) and not Tbuf[w]
#                 GI[curr], GI[w_i] = w, GI[curr]
#                 curr += 1
#         if pos[v] < sep:
#             P_degree[pos[v] - PS] += v_degree
#     Tbuf[incd] = True
#     Tbuf[new_incd] = True
    
#     #--------------------------------#
    
#     # Color the nodes
#     colors = np.zeros(sep - PS, np.int32)
#     nei_bool = np.zeros(sep - PS + 1, np.bool_)
#     nei_list = np.empty(sep - PS, np.int32)
#     max_colors = 0

#     Tbuf[incd] = False
#     Tbuf[new_incd] = False
    
#     # Recompute P_degree    
#     for v_i in range(sep, PS):
#         v = PX[v_i]
#         v_degree = 0
#         for w in GI_new[GS_new[v] : GE_end[v] : 2]:
#             w_pos = pos[w]
#             if w_pos < PS or w_pos >= XE:
#                 break
#             v_degree += (w_pos < sep) and (not Tbuf[w])
#         for w in GI[GS[v] : GE[v]]:
#             w_pos = pos[w]
#             if w_pos < PS or w_pos >= XE:
#                 break
#             v_degree += (w_pos < sep) and (not Tbuf[w])
#         P_degree[v_i] = v_degree
                
#     for v_i in np.argsort(-1 * P_degree):
#         if not Tbuf[P[v_i]]:
#             max_colors = set_color(GI_new, GS_new, GE_end, GI, GS, GE,
#                                    pos, P, sep, PS, XE, v_i, colors, nei_bool, nei_list, max_colors)
#     Tbuf[incd] = True
#     Tbuf[new_incd] = True
        
#     max_possible = R.size + max_colors
#     if max_possible < min(min_cover, min_cover_within, min_cover_btw):
#         tree[12, curr_tree] = 5
#         tree[0, 1] += 1
#         return C, CP, CN, 0, tree
        
#     tmp_sizes = get_branch_sizes(GI_new, GS_new, GE_end, GI, GS, GE, pos, sep, PS, XE, colors, nei_bool, nei_list, incd)
#     see_purpose = (min_cover > R.size + 1 + tmp_sizes.max()) and (max_degree < P.size)
#     if see_purpose:
#         within_purpose = (R.size + 1 + tmp_sizes) >= min_cover_within_incd
#         btw_purpose = np.zeros(incd.size, np.bool_)
#         tmp_btw_purpose = (potE[incd] > potS[incd]).nonzero()[0]
#         btw_purpose[tmp_btw_purpose[potI[potE[incd[tmp_btw_purpose]] - 2] <= (R.size + 1 + tmp_sizes[tmp_btw_purpose])]] = True
#         branches_purpose = within_purpose | btw_purpose
#         if not np.any(branches_purpose):
#             tree[12, curr_tree] = 8
#             tree[0, 1] += 1
#             return C, CP, CN, 0, tree
#     else:
#         branches_purpose = np.zeros(incd.size, np.bool_)
        
#     ## Get branches
#     Padj_u = GI[GS[u] : u_curr]
#     Padj_new_u = GI_new[GS_new[u] : u_curr_new : 2]        
#     Tbuf[Padj_u] = False
#     Tbuf[Padj_new_u] = False
    
#     ### Consideration: maybe btw_new[P] should only be included if the pivot is in P?    
#     # Always keep the btw_new
#     Tbuf[P[btw_new[P]]] = True
    
#     # Ensure that for any node left out of P, all of its dG neighbors in P are included
#     for v in incd[~ Tbuf[incd]]:
#         if not Tbuf[v]:
#             # By construction, this will only iterate over w's that are in incd
#             for w in GI_new[GS_new[v] : GE_end[v]: 2]:
#                 w_pos = pos[w]
#                 Tbuf[w] = True
#                 if w_pos < PS or w_pos >= XE:
#                     break
#     branches = incd[Tbuf[incd]]
#     if see_purpose:
#         extra_priority = branches_purpose[Tbuf[incd]]
#     else:
#         extra_priority = np.zeros(branches.size, np.bool_)
    
#     branches_sizes = colors[pos[branches] - PS] - 1
    
#     # Update branch sizes based on nodes removed by pivoting
#     distinct_bool = np.zeros(PX.size, np.bool_)
#     distinct = np.empty(PX.size + 1, np.int32)
#     incd_bool = np.zeros(PX.size, np.bool_)
#     incd_bool[incd] = True    
#     for v_i in range(branches.size):
#         v = branches[v_i]
#         v_color = colors[pos[v] - PS]
#         distinct_count = 0
#         for w in GI[GS[v] : GE[v]]:
#             w_pos = pos[w]
#             if PS <= w_pos and w_pos < sep and incd_bool[w] and (not Tbuf[w]) and colors[w_pos - PS] > v_color:
#                 w_col = colors[w_pos - PS]
#                 if not distinct_bool[w_col]:
#                     distinct_bool[w_col] = True
#                     distinct[distinct_count] = w_col
#                     distinct_count += 1
#             elif PS > w_pos or w_pos >= XE:
#                 break
#         for w in GI_new[GS_new[v] : GE_end[v] : 2]:
#             w_pos = pos[w]
#             if PS <= w_pos and w_pos < sep and incd_bool[w] and (not Tbuf[w]) and colors[w_pos - PS] > v_color:
#                 w_col = colors[w_pos - PS]
#                 if not distinct_bool[w_col]:
#                     distinct_bool[w_col] = True
#                     distinct[distinct_count] = w_col
#                     distinct_count += 1
#             elif PS > w_pos or w_pos >= XE:
#                 break
#         distinct_bool[distinct[:distinct_count]] = False
#         branches_sizes[v_i] += distinct_count
        
#     Tbuf[Padj_new_u] = True
#     Tbuf[Padj_u] = True

#     #####
#     ## Strongly prioritize branching off btw_new first
# #     branches_order = np.argsort(-1 * (branches + (1000000 * branches_sizes)))
#     branches_order = np.argsort(-1 * (branches_sizes + (1000000 * extra_priority)))
#     branches = branches[branches_order]
#     branches_sizes = branches_sizes[branches_order]
    
# #     print indent, 'dG:', [(i, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]    
# #     print indent, 'min_cover_within:', min_cover_within    
# #     print indent, 'incd:', incd[:incd_count]
# #     print indent, 'pivot u:', u
# #     print indent, 'Padj_u:', Padj_u
# #     print indent, 'Padj_new_u:', Padj_new_u
# #     print indent, 'incd:', incd
# #     print indent, 'new_incd:', new_incd
# #     print indent, 'branches:', branches
        
#     # Initialize max_cover
#     max_cover = 0
#     init_CN = CN
    
#     last_tree = tree[0,0]
    
#     used = np.empty(branches.size, np.int32)
#     used_count = 0
    
#     colors_v = np.empty(PX.size, np.int32)
#     colors_v[P] = colors    
    
#     tree[15, curr_tree] = 0
    
#     v_i = -1
#     check_swap = depth == 0
#     if check_swap:     
#         delayed = np.empty(branches.size, np.int32)
#         delayed_sizes = np.empty(branches.size, np.int32)
#         delayed_count = 0
#         fill = 0
#     while True:
#         v_i += 1
#         if check_swap:
#             if v_i==branches.size:                
#                 if delayed_count > 0:
#                     v_i = fill
#                     branches[fill:] = delayed[:delayed_count]
#                     branches_sizes[fill:] = delayed_sizes[:delayed_count]
#                     check_swap = False
#                     # if verbose and depth == 0:
#                     #     print('DONE SWAPPING, DEPTH:')
#                     #     print((depth))
#                 else:
#                     break
#             else:
#                 v = branches[v_i]
                
#                 found = False
#                 for w_i in range(GS_new[v], GE_end[v], 2):
#                     w = GI_new[w_i]
#                     w_pos = pos[w]
#                     if (PS <= w_pos) and (w_pos < sep):
#                         if GI_new[w_i+1]==0:
#                             found = True
#                             break
#                     elif w_pos < PS or w_pos >= XE:
#                         break

#                 if not found:
#                     delayed[delayed_count] = v
#                     delayed_sizes[delayed_count] = branches_sizes[v_i]
#                     delayed_count += 1
#                     continue
#                 else:
#                     branches[fill] = v
#                     fill += 1
#         else:
#             if v_i==branches.size:
#                 break
            
#         v = branches[v_i]
# #         print indent, 'branching at:', v
# #         print indent, 'pot_min:', pot_min
# #         print indent, 'min/max_cover:', min_cover, max_cover        
# #         print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]
# #         print indent, 'potS/end:', zip(potS, potE)
# #         print indent, 'dG:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
# #         print indent, 'sub_max_possible:', sub_max_possible, 'min(min_cover, min_cover_within, min_cover_btw):', min(min_cover, min_cover_within, min_cover_btw)
# #         sub_max_possible = R.size + 1 + branches_sizes[v_i]

#         tmp = get_branch_sizes_vw(GI_new, GS_new, GE_end,
#                                   GI, GS, GE,
#                                   pos, sep, PS, XE, colors_v, nei_bool, nei_list, v)
#         sub_max_possible = R.size + 1 + tmp

#         if sub_max_possible >= min(min_cover, min_cover_within, min_cover_btw):
#             used[used_count] = v
#             used_count += 1
#         else:
#             continue
            
#         min_newly_capt = 1000000
#         sub_min_cover_btw = 1000000
#         sub_min_cover_within = 1000000
#         btw_added = 0
#         new_PS, new_XE = sep, sep
        
#         sub_tree = 4 + tree[0,0]
    
#         for w_i in range(GS[v], GE[v]):            
#             w = GI[w_i]
#             w_pos = pos[w]
#             if (PS <= w_pos) and (w_pos < sep):
#                 new_PS -= 1
#                 swap_pos(PX, pos, w, new_PS)
        
#                 if potE[w] > potS[w]:
#                     sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])                
#             elif (sep <= w_pos) and (w_pos < XE):
#                 swap_pos(PX, pos, w, new_XE)
#                 new_XE += 1
#             elif w_pos < PS or w_pos >= XE:
#                 break
        
#         for w_i in range(GS_new[v], GE_end[v], 2):
#             w = GI_new[w_i]
#             w_pos = pos[w]
#             if (PS <= w_pos) and (w_pos < sep):
#                 new_PS -= 1
#                 swap_pos(PX, pos, w, new_PS)               
#                 if not btw_new[w]:
#                     btw_stack[btw_end + btw_added] = w
#                     btw_added += 1
#                     btw_new[w] = True

#                 if sub_max_possible >= GI_new[w_i+1]:
#                     potI[potE[w]] = GI_new[w_i+1]
#                     if potE[w] > potS[w]:
#                         potI[potE[w]+1] = min(potI[potE[w]-2], GI_new[w_i+1])
#                     else:
#                         potI[potE[w]+1] = GI_new[w_i+1]
#                     potE[w] += 3
#                 if potE[w] > potS[w]:
#                     sub_min_cover_btw = min(sub_min_cover_btw, potI[potE[w]-2])
#             elif (sep <= w_pos) and (w_pos < XE):
#                 swap_pos(PX, pos, w, new_XE)
#                 new_XE += 1
#             elif w_pos < PS or w_pos >= XE:
#                 break
        
#         # Update min cover based on r4: r1,r2,r3
#         newly_capt = potI[potS[v] : potE[v]]
#         if newly_capt.size > 0:
#             min_newly_capt = newly_capt[newly_capt.size - 2]
        
#         ## Don't need to apply this criterion yet (since dG assumes no new edge yet)
#         ## However, if you do, then do it on a min_below computes by PX-by-PX looping above
#         if sep - new_PS == 0:
#             do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, sub_min_cover_btw)
#         else:
#             do_subbranch = min(sub_max_possible, R.size + 1 + (sep - new_PS)) >= min(min_cover, min_newly_capt, min_cover_within, sub_min_cover_btw)
        
#         if do_subbranch:
#             sub_tree = 4 + tree[0, 0]
#             if tree.shape[1] <= sub_tree:
#                 tree = expand_2d_arr(tree)

#         if do_subbranch:
#             Rbuf[RE] = v
#             sub_min_cover = min(min_cover, min_newly_capt)
            
#             prev_CN = CN
            
#             if btw_new[v]:
#                 C, CP, CN, sub_max_cover, tree = BK_Gsep_cover(
#                     Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
#                     GS, GE, GI,
#                     GS_new, GE_end, GI_new,
#                     Tbuf, depth+1,
#                     potI, potS, potE, pot_min, sub_max_possible,
#                     sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
#                     C, CP, CN, tree)
#             else:
#                 C, CP, CN, sub_max_cover, tree = BK_dG_cover(
#                     Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
#                     GS, GE, GI,
#                     GS_new, GE_end, GI_new,
#                     Tbuf, depth+1,
#                     btw_new, btw_stack, btw_end + btw_added,
#                     potI, potS, potE, pot_min, sub_max_possible,
#                     sub_min_cover, max_cover, sub_min_cover_btw, max_branch_depth,
#                     C, CP, CN, tree)

#             # Update min_cover, max_cover
#             max_cover = max(max_cover, sub_max_cover)
#             min_cover = max(min_cover, sub_max_cover)
            
#             if sub_max_cover > min_newly_capt:
#                 # Update max cover of (r4,r1), (r4,r2), (r4,r3), which we will move back to potential, and restore captured.
#                 # Fresh updates to max cover are moved to accum r1:...r4, r2:...r4, r3:...r4
#                 for w_i in range(0, newly_capt.size, 3):
#                     newly_capt[w_i] = max(newly_capt[w_i], sub_max_cover)
#                     newly_capt[w_i+1] = max(newly_capt[w_i+1], sub_max_cover)

#         # Restore pot[w] for all new neighbors of v
#         for w_i in range(GS_new[v], GE_end[v], 2):
#             w = GI_new[w_i]
#             w_pos = pos[w]
#             if (PS <= w_pos) and (w_pos < sep):
#                 if sub_max_possible >= GI_new[w_i+1]:
#                     if do_subbranch:
# #                         assert GI_new[w_i+1] == GI_new[potI[potE[w]-1]]
#                         GI_new[w_i+1] = max(GI_new[w_i+1], potI[potE[w]-3])
#                         GI_new[potI[potE[w]-1]] = max(GI_new[potI[potE[w]-1]], potI[potE[w]-3])                
#                     potE[w] -= 3
#             elif w_pos < PS or w_pos >= XE:
#                 break
        
#         # Recompute min_cover_within  
#         if do_subbranch and sub_max_cover > min_cover_within:
#             min_cover_within = 1000000
#             for x_i in range(sep-1, PS-1, -1):
#                 x = PX[x_i]
#                 # Move P and X to the bottom
#                 for w_i in range(GS_new[x], GE_end[x], 2):
#                     w = GI_new[w_i]
#                     w_pos = pos[w]
#                     if w_pos < PS or w_pos >= XE:
#                         break
#                     elif PS <= w_pos and w_pos < sep:
#                         min_cover_within = min(min_cover_within, GI_new[w_i+1])
        
#         # Reset the btw_new
#         btw_new[btw_stack[btw_end : btw_end + btw_added]] = False

#         # Swap v to the end of P, and then decrement separator
#         sep -= 1
#         swap_pos(PX, pos, v, sep)
        
#         if depth==0:
#             # if verbose: print((v, sub_max_possible, sub_max_cover, do_subbranch, last_tree - tree[0,0]))
#             last_tree = tree[0,0]
            
#         # Don't branch anymore after this
#         if depth > max_branch_depth and used_count==1:
#             break

#     for v in used[:used_count][::-1]:
#         # Move v to the beginning of X and increment separator
#         swap_pos(PX, pos, v, sep)
#         sep += 1
    
# #     print indent, 'Returning::::'
# #     print indent, 'R:', R, 'P:', P, 'X:', X
# #     print indent, 'pot_min:', pot_min
# #     print indent, 'min/max_cover:', min_cover, max_cover
# #     print indent, 'potS/end/indices:', [(i,potI[x:y].tolist()) for i, (x, y) in enumerate(zip(potS, potE))]    
# #     print indent, 'dG:', [(i, x, y, GI_new[x:y].tolist()) for i, (x, y) in enumerate(zip(GS_new, GE_end))]
    
#     return C, CP, CN, max_cover, tree
