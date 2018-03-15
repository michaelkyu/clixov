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
from color import set_color, get_branch_sizes, get_branch_sizes_vw, color_nodes, count_unique
from degeneracy import get_degeneracy

@jit(nopython=True, cache=cache)
def MC_bf_cover(Rbuf, RE, PX, pos,
                GS, GE, GI,
                dGS, dGE, dGI,
                cover,
                Fbuf, Tbuf, depth,
                C, CP, CN, core_nums, core_nums2,
                stats, tree, offset=0, verbose=True):
    # F is a set of max cover cliques for new edges that have already
    # been iterated upon

    curr_node = stats[0]

    sep = PX.size
    old_sep = sep     

    to_branch = np.zeros(PX.size, np.bool_)
    colors_v = np.empty(PX.size, PX.dtype)

    stack = np.empty(PX.size, PX.dtype)
    used = np.zeros((PX.size, PX.size), np.bool_)
    used2 = np.zeros((PX.size, PX.size), np.bool_)

    F, FP, FN = C.copy(), CP.copy(), CN

    tri = set()
    
    for v in range(PX.size):        
        prev_FN_v = FN
        dG_v = dGI[dGS[v]:dGE[v]]
        dG_v = dG_v[dG_v > v]
        
        if verbose: indent = '\t'
        if verbose: print indent, 'v:', v, 'dG_v:', dG_v
        if verbose: print indent, 'used:', zip(*used.nonzero())
        if verbose: print indent, 'old_sep:', old_sep
        if verbose: print indent, 'GI[GS[v]:GE[v]]:', GI[GS[v]:GE[v]]

        if dG_v.size > 0:
            v_sep, P = update_P(GI, GS, GE, PX, old_sep, old_sep, pos, v)
            if verbose: print indent, 'P:', P

            to_branch[dG_v] = True
                        
            # v_sep = 0
            # for u in P:
            #     if (not to_branch[u]) or (u>v):
            #         swap_pos(PX, pos, u, v_sep)
            #         v_sep += 1
            # P = PX[:v_sep]
            # if verbose: print indent, 'P not used:', P

            stats[0] += 1
            curr_node_v = stats[0]
            tree[0, curr_node_v] = curr_node
            tree[1, curr_node_v] = v
            
            GE_prev, P_copy = GE[P], P.copy()

            # Push down GI
            Fbuf[P] = True
            for u in P:
                u_degree, GE[u] = move_PX_fast_bool(GI, GS, GE, Fbuf, u)
#                _, _ = move_PX_fast(F, FP[:FN-1], FP[1:FN], Fbuf, u)

                # # Remove all new edges that have already been max covered
                # u_degree, GE[u] = move_PX_fast_bool_unused(GI, GS, GE, Fbuf, used, u)
            Fbuf[P] = False
            # if verbose: print indent, 'G:', sparse_str_I(GI, GS, GE)
                
            w_sep = v_sep

            colors, branches = color_nodes(GI, GS, GE, np.sort(P.copy())[::-1])
            colors_v[branches] = colors
            if verbose: print indent, 'colors/branches:', zip(colors, branches)

            # Manually reset used2
            used2 = np.zeros((PX.size, PX.size), np.bool_)

            Rbuf[0] = v
            for w_i in range(branches.size-1, -1, -1):
                w = branches[w_i]
                if verbose: print indent, 'w:', w, 'to_branch[w]:', to_branch[w]
                
                if to_branch[w]:
                    if verbose: print indent, 'w:', w, 'P:', P
                    w_sep, P = update_P(GI, GS, GE, PX, v_sep, v_sep, pos, w)
                    if verbose: print indent, 'w:', w, 'P:', P
                    if verbose: print indent, 'w:', w, 'used2:', zip(*used2.nonzero())                                        

                    if verbose: print indent, 'tri:', tri
                    
                    # Only keep in P those nodes that have not
                    # iterated with w in a max cover
                    w_sep = 0
                    for u in P:
#                        if not to_branch[u] or ((not used2[u,w]) and u>v and u>w):
                        if not to_branch[u] or (not used2[u, w]):  # Works 3-9-18
#                        if not used2[u, w]:
                            if ((not v<u) or ((v, u, w) not in tri)) and ((not u<v) or ((u,v,w) not in tri)):
                                swap_pos(PX, pos, u, w_sep)
                                w_sep += 1
                    P = PX[:w_sep]
                    if verbose: print indent, 'w:', w, 'P not used2:', P

                    # sep, P = reduce_G(GI, GS, GE, Fbuf, core_nums2, max_cover - 1, PX, pos, sep)
                                    
                    cover_vw = max(cover[v,w], cover[w,v])
                    cover[v,w], cover[w,v] = cover_vw, cover_vw
                    if verbose: print indent, 'w:', w, 'cover_vw', cover_vw, 'upper bound:', 2 + P.size

                    unique_colors = count_unique(colors_v[P], Fbuf, stack)
                    if (2 + unique_colors) >= cover_vw:
                        Rbuf[1] = w
                        tree[0, stats[0]+1] = curr_node_v
                        prev_CN = CN
                        C, CP, CN, tree, max_cover, sub_cover = MC_branch_bf_cover(
                            Rbuf, 2, PX, w_sep, pos,
                            GS, GE, GI,
                            Fbuf, Tbuf, 2,
                            cover_vw, 0, cover,
                            C, CP, CN, core_nums, core_nums2,
                            stats, tree, offset=offset, verbose=verbose)
                        cover_vw = max(cover_vw, sub_cover)
                        cover[v,w], cover[w,v] = cover_vw, cover_vw

                        # Update F
                        stack_n = 0
                        if verbose: print indent, 'w:', w, 'prev_CN/CN:', prev_CN, CN
                        for r_i in range(prev_CN, CN):
                            if (CP[r_i+1] - CP[r_i]) == cover_vw:
                                r = C[CP[r_i]:CP[r_i+1]]
                                if verbose: print indent, 'Found clique:', r
                                F, FP, FN = update_cliques(F, FP, FN, r)
                                for rr in r:
                                    if not Fbuf[rr]:                                     
                                        Fbuf[rr] = True
                                        stack[stack_n] = rr
                                        stack_n += 1                                        
                        for u in stack[:stack_n]:
                            # tri.add((v,w,u))
                            used2[w, u] = True
                        used2[w, w] = False
                        Fbuf[stack[:stack_n]] = False

            # stack_n = 0
            # for r_i in range(prev_FN, FN):
            #     clique = F[FP[r_i]:FP[r_i+1]]
            #     for rr in clique[1:]:
            #         if not Fbuf[rr]:
            #             Fbuf[rr] = True
            #             stack[stack_n] = rr
            #             stack_n += 1
            # for x_i in range(stack_n):
            #     for dGI[d
                
            # Fbuf[stack[:stack_n]] = False
            
            to_branch[dG_v] = False
            GE[P_copy] = GE_prev
            
    C, CP, CN = trim_cliques(C, CP, CN)
    tree[2,curr_node] = 3
    return C, CP, CN, tree

@jit(nopython=True, cache=cache)
def MC_branch_bf_cover(Rbuf, RE, PX, sep, pos,
                       GS, GE, GI,
                       Fbuf, Tbuf, depth,
                       max_cover, sub_cover, cover,
                       C, CP, CN, core_nums, core_nums2,
                       stats, tree, offset=0, verbose=True):
    if tree.shape[1] == stats[0] + 2:
        tree = expand_2d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]
    tree[1,curr_node] = Rbuf[RE-1]

    P = PX[:sep]
    R = Rbuf[:RE]

    if verbose: indent = '\t' * depth
    if verbose: print indent, '---------MC_branch------'
    if verbose: print indent, 'DEPTH:', depth, 'max_cover:', max_cover
    if verbose: print indent, 'R:', R
    if verbose: print indent, 'P:', P

    # Check bound on size of P
    if (R.size + P.size) < max_cover:
        tree[2,curr_node] = 1
        return C, CP, CN, tree, max_cover, 0
    elif P.size==0:
        C, CP, CN = update_cliques(C, CP, CN, R)
        # if verbose: print indent, '********************'
        tree[2,curr_node] = 0
        return C, CP, CN, tree, max_cover, R.size

    GE_prev, P_copy = GE[P], P.copy()

    # Push down GI
    max_degree = 0
    Fbuf[P] = True
    for u in P:
        u_degree, GE[u] = move_PX_fast_bool(GI, GS, GE, Fbuf, u)
        max_degree = max(u_degree, max_degree)
    Fbuf[P] = False

    # if verbose: print indent, 'G:', sparse_str_I(GI, GS, GE)
                
    # Check if any H's cover all of the nodes. If so, then return

    # Filter out H's that are not part HPX, HPS, 
    
    # Check bound on max degree
    if (R.size + max_degree + 1) < max_cover:
        if verbose: print indent, 'Returning because max_degree/max_cover:', max_degree, max_cover
        GE[P_copy] = GE_prev
        tree[2,curr_node] = 2
        return C, CP, CN, tree, max_cover, 0
            
    ## Color
    color_order = np.sort(P.copy())[::-1]
    colors, branches = color_nodes(GI, GS, GE, color_order)
    old_sep = sep

    if verbose: print indent, 'colors/branches:', zip(colors, branches)
        
    ## TODO: include reduce_G
    
    for v_i in range(branches.size -1, -1, -1):
        v = branches[v_i]
        if verbose: indent = '\t' * depth
        if verbose: print indent, 'v:', v, 'color:', colors[v_i], 'max_cover:', max_cover, 'R:', R

        if (R.size + colors[v_i]) >= max_cover:
            Rbuf[RE] = v
            new_sep, P = update_P(GI, GS, GE, PX, old_sep, sep, pos, v)
            tree[0, stats[0]+1] = curr_node
            C, CP, CN, tree, max_cover, tmp_cover = MC_branch_bf_cover(
                Rbuf, RE + 1, PX, new_sep, pos,
                GS, GE, GI,
                Fbuf, Tbuf, depth + 1,
                max_cover, 0, cover,
                C, CP, CN, core_nums, core_nums2,
                stats, tree, offset=offset, verbose=verbose)
            if sub_cover > max_cover:
                max_cover = sub_cover
                sep, P = reduce_G(GI, GS, GE, Fbuf, core_nums2, max_cover - 1, PX, pos, sep)

            sub_cover = max(sub_cover, tmp_cover)
            max_cover = max(max_cover, sub_cover)            
            for r in R:
                cover[v,r] = max(cover[v,r], sub_cover)

            ## TODO: include reduce_G
            
        # Remove v from P
        sep -= 1
        swap_pos(PX, pos, v, sep)

    GE[P_copy] = GE_prev

    if verbose: print indent, '---------MC_branch End------'

    tree[2,curr_node] = 3
    return C, CP, CN, tree, max_cover, sub_cover

def MC_bf_cover_py(dG, G, offset=0, verbose=False, unique=False):
    """
    'bf' stands for a brute force algorithm to search all per-edge max
    covers by iterating through each edge separately.

    dG is the edges to cover. G is the complete graph, including edges in dG.

    Goes through each edge one-by-one to find the maximum clique containing that edge.

    """

    verbose = True

    if verbose:
        print 'G'
        print G.toarray().astype(np.int32)
        print 'dG'
        print dG.toarray().astype(np.int32)
    
    ## Degeneracy
    GS, GE, GI = G.indptr[:-1].copy(), G.indptr[1:].copy(), G.indices.copy()
    degen_order, core_num, _ = get_degeneracy(GI, GS, GE, np.arange(G.shape[0]).astype(np.int32))
    # if verbose:  print 'degen_order/core_num:', zip(degen_order, core_num)

    keep = degen_order
    if verbose:
        print 'keep:', keep
        interest = [3,6,7,9]
        print interest, np.argsort(keep)[interest]
        #print '(3, 5, 6, 8, 11):', np.argsort(keep)[[3, 5, 6, 8, 11]]
    
    ## Remove nodes
    G = G[:,keep][keep,:]
    G.sort_indices()
    if verbose:
        print 'G:', clixov_utils.sparse_str(G)
        print G.toarray().astype(np.int32)
        tmp = np.argsort(keep)[interest]
        #tmp = np.argsort(keep)[[1,2,3,4,5]]
        print G[:,tmp][tmp,:].toarray().astype(np.int32)
    GS, GE, GI = G.indptr[:-1].copy(), G.indptr[1:].copy(), G.indices.copy()
    dG = dG[:,keep][keep,:]
    dG.sort_indices()
    if verbose:
        print 'dG:', clixov_utils.sparse_str(dG)
        print dG.toarray().astype(np.int32)
        print dG[:,tmp][tmp,:].toarray().astype(np.int32)
    dGS, dGE, dGI = dG.indptr[:-1].copy(), dG.indptr[1:].copy(), dG.indices.copy()
    k = G.shape[0]
        
    core_num = core_num[np.argsort(degen_order)][keep]

    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, _ = initialize_structures(k)

    cover = np.zeros((k,k), np.int32)
    
    tree = np.asfortranarray(np.zeros((14, 100000), np.int32))
    tree.fill(-1)

    core_num2 = core_num.copy()
    for i in range(1, core_num2.size):
        core_num2[i] = max(core_num2[i], core_num2[i-1])

    #if verbose:
        # print 'degrees:', as_dense_flat(G.sum(0))
        # print 'core_num:', core_num
        # print 'core_num2:', core_num2

    start_time = time.time()    
    C, CP, CN, tree = MC_bf_cover(
        R, RE, PX, pos,
        GS, GE, GI,
        dGS, dGE, dGI,
        cover,
        Fbuf, Tbuf, 0,
        C, CP, CN, core_num, core_num2,
        stats, tree, offset=offset, verbose=verbose)

    # if verbose:
    # tmp = np.argsort(keep)[interest]
    # print cover[tmp,:][:,tmp]
        
    if verbose:
        print 'Time:', time.time() - start_time

    print 'CN:', CN
    print 'Tree nodes:', stats[0]
    tree = tree[:,:stats[0]+1]

    tmp2_cliques = [tuple(sorted(C[CP[i]:CP[i+1]])) for i in range(CN)]
    if verbose: print 'Cliques (degen indices):', tmp2_cliques
    C = keep[C]
    
    if verbose: print 'tree:', tree[0, :3]
    if verbose: print 'Clique sizes (%s):' % CN, Counter(CP[1:] - CP[:-1])

    tmp_cliques = [tuple(sorted(C[CP[i]:CP[i+1]])) for i in range(CN)]        
    max_size = max([len(x) for x in tmp_cliques])
    if verbose: print 'Cliques:', tmp_cliques
    
    if unique:
        if not len(tmp_cliques) == len(set(tmp_cliques)):
            print Counter(tmp_cliques)
            raise Exception('Non-unique cliques')
    else:
        a = len(tmp_cliques)
        tmp_cliques = list(set(tmp_cliques))
        print 'Found/non-redundant cliques:', a, len(tmp_cliques)
        
    cliques = tuples_to_csc(tmp_cliques, degen_order.size)
    if verbose: print 'Found cliques:', cliques.shape[1], as_dense_flat(cliques.sum(0))
    # cliques, max_clique = get_largest_cliques(cliques)
    if verbose: print 'After filtering for largest cliques:', cliques.shape[1]
    
    return cliques, keep, tree





