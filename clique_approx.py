import time

import numpy as np
from numba import jit

from constants import cache
import clixov_utils
from clixov_utils import get_largest_clique_covers, cliques_to_csc, sparse_str, sparse_str_I
from clique_atomic import update_cliques, trim_cliques, swap_pos, get_unique_cliques, expand_2d_arr, move_PX
from degeneracy import get_degeneracy
from color import color_nodes

verbose = True

# @jit(nopython=True, cache=cache)
# def cont(GI, GS, GE,
#          in_P, P, pos,
#          Rbuf, RE, cores,
#          C, CP, CN):
#     ## Greedily select the vertex with the largest k-core number

#     if P.size==0:
#         R = Rbuf[:RE]
#         C, CP, CN = update_cliques(C, CP, CN, R)
#         return C, CP, CN
    
#     max_core = cores[P].max()
#     for v in P:
#         if cores[v]==max_core:
#             sep, P_v, not_P_v = update_P_fast(GI, GS, GE, in_P, P, pos, v)
#             Rbuf[RE] = v
#             C, CP, CN = cont(GI, GS, GE,
#                              in_P, P_v, pos,
#                              Rbuf, RE+1, cores,
#                              C, CP, CN)
#             in_P[not_P_v] = True

#     return C, CP, CN

# @jit(nopython=True, cache=cache)
# def approx_nodes(GI, GS, GE):
#     n = GS.size
#     P = np.arange(n).astype(GS.dtype)
#     pos = P.copy()
#     in_P = np.ones(n, np.bool_)
#     Rbuf = np.empty(n, P.dtype)

#     degen_order, degen_deg, cores = get_degeneracy(GI, GS, GE, P)
#     C, CP, CN = np.empty(P.size, P.dtype), np.zeros(P.size, P.dtype), 0
    
#     for v in degen_order:
#         Rbuf[0] = v
#         sep, P_v, not_P_v = update_P_fast(GI, GS, GE, in_P, P, v)
#         C, CP, CN = cont(GI, GS, GE,
#                          in_P, P_v, pos,
#                          Rbuf, 1, cores,
#                          C, CP, CN)
#         in_P[not_P_v] = True

#     C, CP, CN = trim_cliques(C, CP, CN)
#     return C, CP, CN

# @jit(nopython=True, cache=cache)
# def update_P_fast(GI, GS, GE, in_P, P, pos, v):
#     sep = 0
#     for w in GI[GS[v]:GE[v]]:
#         if in_P[w]:
#             swap_pos(P, pos, w, sep)
#             sep += 1
#     P_v = P[:sep]
#     not_P_v = P[sep:]
#     in_P[not_P_v] = False
#     return sep, P_v, not_P_v
        
# @jit(nopython=True, cache=cache)
# def approx_edges(GI, GS, GE, dGI, dGS, dGE):
#     n = GS.size
#     P = np.arange(n).astype(GS.dtype)
#     pos = P.copy()
#     in_P = np.ones(n, np.bool_)
#     Rbuf = np.empty(n, P.dtype)
#     in_dG_v = np.zeros(n, np.bool_)
    
#     degen_order, degen_deg, cores = get_degeneracy(GI, GS, GE, P)
#     degen_order_idx = np.argsort(degen_order)
#     C, CP, CN = np.empty(P.size, P.dtype), np.zeros(P.size, P.dtype), 0

#     for v_i in range(n):
#         v = degen_order[v_i]
#         Rbuf[0] = v
#         sep, P_v, not_P_v = update_P_fast(GI, GS, GE, in_P, P, pos, v)
#         in_dG_v[dGI[dGS[v]:dGE[v]]] = True

#         #print 'v:', v, 'P_v:', P_v, dGI[dGS[v]:dGE[v]], in_P.nonzero()[0]
        
#         for w in P_v:
#             if in_dG_v[w] and degen_order_idx[w] > v_i:
#                 sep, P_w, not_P_w = update_P_fast(GI, GS, GE, in_P, P_v, pos, w)
#                 Rbuf[1] = w                
#                 C, CP, CN = cont(GI, GS, GE, in_P, P_w, pos, Rbuf, 2, cores, C, CP, CN)
#                 in_P[not_P_w] = True

#         in_dG_v[dGI[dGS[v]:dGE[v]]] = False
#         in_P[not_P_v] = True

#     C, CP, CN = trim_cliques(C, CP, CN)
#     #C, CP, CN = get_unique_cliques(C, CP, CN)
#     return C, CP, CN


###############################################################
# Greedy method: only branch on one of the nodes with max core

# @jit(nopython=True, cache=cache)
# def cont(GI, GS, GE, P, pos, sep, Rbuf, RE, cores, C, CP, CN, degen_order_idx):
#     ## Greedily select the vertex with the largest k-core number

#     P = P[:sep]

#     # if verbose:
#     #     indent = '\t'*(RE-1)
#     #     print indent, '---cont----'
#     #     print indent, 'R:', Rbuf[:RE], 'P:', P, 'cores[P]:', cores[P]
    
#     if P.size==0:
#         C, CP, CN = update_cliques(C, CP, CN, Rbuf[:RE])
#         tmp = Rbuf[:RE].copy()
#         tmp.sort()
#         #if verbose: print indent, 'Return:', tmp
#         return C, CP, CN
    
#     max_core = cores[P].max()
#     for v in P[cores[P]==max_core][:1]:
#         new_sep = update_P_fast(GI, GS, GE, P, pos, sep, v)
#         Rbuf[RE] = v
#         C, CP, CN = cont(GI, GS, GE, P, pos, new_sep,
#                          Rbuf, RE+1, cores, C, CP, CN, degen_order_idx)
#         sep -=1
#         swap_pos(P, pos, v, sep)

#     # if verbose:
#     #     print indent, 'pos:', pos
#     return C, CP, CN

# @jit(nopython=True, cache=cache)
# def update_P_fast(GI, GS, GE, P, pos, sep, v):
#     new_sep = 0
#     for w in GI[GS[v]:GE[v]]:
#         if pos[w]<sep:
#             swap_pos(P, pos, w, new_sep)
#             new_sep += 1
#     return new_sep
        
# @jit(nopython=True, cache=cache)
# def approx_edges(GI, GS, GE, dGI, dGS, dGE):
#     n = GS.size
#     P = np.arange(n).astype(GS.dtype)
#     pos = P.copy()
#     Rbuf = np.empty(n, P.dtype)
#     in_dG_v = np.zeros(n, np.bool_)
#     Fbuf = np.zeros(n, np.bool_)

#     degen_order, degen_deg, cores = get_degeneracy(GI, GS, GE, P)
#     degen_order_idx = np.argsort(degen_order)
#     C, CP, CN = np.empty(P.size, P.dtype), np.zeros(P.size, P.dtype), 0

#     stack = np.empty(n).astype(GS.dtype)
#     stack_end = 0
    
#     for v_i in range(n):
#         v = degen_order[v_i]
#         dG_v = dGI[dGS[v]:dGE[v]]
#         in_dG_v[dG_v] = True
#         Rbuf[0] = v        
#         sep_v = update_P_fast(GI, GS, GE, P, pos, n, v)        
#         P_v = P[:sep_v].copy()

#         # Find vertex cover
#         in_branches = np.zeros(n, np.bool_)
#         Fbuf[P_v] = True
#         for w in P_v:
#             needed = False
#             for u in GI[GS[w]:GE[w]]:
#                 needed = needed or (Fbuf[u] and not in_branches[u])
#             in_branches[w] = in_branches[w] or needed
#         branches = P_v[in_branches[P_v]]
#         Fbuf[P_v] = False        
         
#         #if verbose: print 'v:', v, 'P_v:', P_v
#         # for w in P_v:
#         #     if in_dG_v[w] and degen_order_idx[w] > v_i:
#         for w in branches:
#             sep_w = update_P_fast(GI, GS, GE, P[:sep_v], pos, sep_v, w)
#             Rbuf[1] = w
#             C, CP, CN = cont(GI, GS, GE, P[:sep_w], pos, sep_w, Rbuf, 2, cores, C, CP, CN, degen_order_idx)
#         in_dG_v[dG_v] = False

#     C, CP, CN = trim_cliques(C, CP, CN)
#     return C, CP, CN

@jit(nopython=True, cache=cache)
def update_P_fast(GI, GS, GE, PX, pos, PS, sep, XE, v):
    new_PS, new_XE = sep, sep
    for w in GI[GS[v]:GE[v]]:
        w_pos = pos[w]
        if (PS<=w_pos) and (w_pos<sep):
            new_PS -= 1
            swap_pos(PX, pos, w, new_PS)
        elif (sep<=w_pos) and (w_pos<XE):
            swap_pos(PX, pos, w, new_XE)
            new_XE += 1
    return new_PS, new_XE
        

@jit(nopython=True, cache=cache)
def cont(GI, GS, GE, PX, pos, PS, sep, XE, Rbuf, RE, cores, C, CP, CN,
         cover, max_cover, rank, stats, tree, colors, Fbuf, stack):
    ## Greedily select the vertex with the largest k-core number

    if tree.shape[1] == stats[0] + 2:
        tree = expand_2d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]
    
    P, X, R = PX[PS:sep], PX[sep:XE], Rbuf[:RE]
    sub_cover = 0
    
    # if verbose:
    #     indent = '\t'*RE
    #     print indent, '---cont----'
    #     print indent, 'R:', Rbuf[:RE], 'P:', P, 'cores[P]:', cores[P], 'X:', X, 'max_cover:', max_cover
        
    if P.size==0:
        if X.size==0:
            # tmp = Rbuf[:RE].copy()
            # tmp.sort()
            # if verbose: print indent, 'Return:', tmp
            C, CP, CN = update_cliques(C, CP, CN, R)
            return C, CP, CN, tree, max_cover, R.size

        return C, CP, CN, tree, max_cover, 0

    branches = [P[np.argmax(rank[P])]]
    
    # max_core = cores[P].max()
    # if max_core <= max_cover:
    #     return C, CP, CN, tree, max_cover, sub_cover
    # branches = P[cores[P]==max_core]

    # if verbose:
    #     print indent, 'branches:', branches
        
    #for v in branches[:1]:
    for v in branches:
        new_PS, new_XE = update_P_fast(GI, GS, GE, PX, pos, PS, sep, XE, v)
        # if verbose:
        #     print indent, 'v:', v, 'R.size:', R.size, 'P_v.size:', sep - new_PS, 'max_core:', max_core
        
        if R.size + 1 + sep - new_PS > max_cover:
            
            # color_bound = get_color_size(PX[new_PS:sep], colors, Fbuf, stack)
            # if (R.size + 1 + color_bound) <= max_cover:
            #     continue

            Rbuf[RE] = v
            tree[0, stats[0]+1] = curr_node
            C, CP, CN, tree, max_cover, tmp_cover = cont(GI, GS, GE, PX, pos, new_PS, sep, new_XE, 
                                                         Rbuf, RE+1, cores, C, CP, CN, cover, max_cover, rank,
                                                         stats, tree, colors, Fbuf, stack)
            sub_cover = max(sub_cover, tmp_cover)
            max_cover = max(max_cover, sub_cover)
            if sub_cover > 0:
                cover_v = cover[v,:]
                for r in R:
                    # Todo: only update covers for dG edges
                    cover_v[r] = max(cover_v[r], sub_cover)
                    
            break
    
        sep -= 1
        swap_pos(PX, pos, v, sep)
                 
    # if verbose:
    #     print indent, 'pos:', pos
    return C, CP, CN, tree, max_cover, sub_cover

@jit(nopython=True)
def get_color_size(P, colors, Fbuf, stack):
    count = 0
    for u in P:
        u_color = colors[u]
        if not Fbuf[u_color]:
            Fbuf[u_color] = True
            stack[count] = u_color
            count += 1
    Fbuf[stack[:count]] = False
    return count
            
@jit(nopython=True, cache=cache)
def approx_edges(GI, GS, GE, dGI, dGS, dGE):
    n = GS.size
    PX = np.arange(n).astype(GS.dtype)
    pos = PX.copy()
    Rbuf = np.empty(n, PX.dtype)
    in_dG_v = np.zeros(n, np.bool_)
    Tbuf = np.ones(n, np.bool_)
    Fbuf = np.zeros(n, np.bool_)

    tree = np.zeros((100000, 4), np.int32).T
    stats = np.zeros(4, np.int32)
    curr_node = stats[0]
    
    degen_order, degen_deg, cores = get_degeneracy(GI, GS, GE, PX)
    degen_order_idx = np.argsort(degen_order)
    C, CP, CN = np.empty(PX.size, PX.dtype), np.zeros(PX.size, PX.dtype), 0

    cover = np.zeros((n, n), np.int32)    
    rank = -1 * (cores * 100000 + degen_order)
    colors_v = np.empty(n, np.int32)
    stack = np.empty(n).astype(GS.dtype)
    stack_end = 0

    for v_i in range(n):
        v = degen_order[v_i]
        dG_v = dGI[dGS[v]:dGE[v]]
        in_dG_v[dG_v] = True
        Rbuf[0] = v
        sep_v = n
        PS_v, XE_v = update_P_fast(GI, GS, GE, PX, pos, 0, n, n, v)
        assert XE_v==n
        P_v = PX[PS_v:n].copy()
        P_v = P_v[np.argsort(rank[P_v])]

        if P_v.size==0:
            continue

        # print 'PX:', PX
        # print 'pos:', pos
        # print 'PS_v/XE_v/sep_v:', PS_v, XE_v, sep_v
        # print 'v:', v

        # print zip(GS, GE)
        # print sparse_str_I(GI, GS, GE)
        
        prev_GE = GE[P_v]
        for w in P_v:
            # import pdb
            # pdb.set_trace()
            # print 'w:', w
            w_degree, GE[w] = move_PX(GI, GS, GE, pos, 0, n, PS_v, XE_v, sep_v, 0, w, GS[w])
#            print 'v:', v, 'w:', w, GI[GS[w]:GE[w]]

        # print zip(GS, GE)
        # print sparse_str_I(GI, GS, GE)
        colors, color_branches = color_nodes(GI, GS, GE, P_v)
        colors_v[color_branches] = colors
        
        # if verbose: print 'v:', v, 'P_v:', P_v

        if tree.shape[1] == stats[0] + 2:
            tree = expand_2d_arr(tree)
        stats[0] += 1
        curr_node_v = stats[0]
        tree[0, curr_node_v] = curr_node

        for w in P_v:
            if in_dG_v[w] and degen_order_idx[w] > v_i:
                # if verbose: indent = '\t'
                # print indent, 'PX:', PX
                # print indent, 'pos:', pos
                # print indent, 'PS_v/XE_v/sep_v:', PS_v, XE_v, sep_v
                # print indent, 'v:', v, 'w:', w, GI[GS[w]:GE[w]]
                                
                PS_w, XE_w = update_P_fast(GI, GS, GE, PX, pos, PS_v, sep_v, XE_v, w)
                max_cover = max(cover[v,w], cover[w,v])

                # if verbose: indent = '\t'
                # if verbose: print indent, 'w:', w, 'max_cover:', max_cover, 'P_w:', PX[PS_w:sep_v], 'cores[P_w]:', cores[PX[PS_w:sep_v]]

                # Remove all nodes in which the core number is not high enough
                for curr in range(PS_w, sep_v):
                    u = PX[curr]
                    if cores[u] < max_cover:
                        swap_pos(PX, pos, u, PS_w)
                        PS_w += 1

                #if verbose: print indent, 'w:', w, 'max_cover:', max_cover, 'P_w:', PX[PS_w:sep_v], 'cores[P_w]:', cores[PX[PS_w:sep_v]]

                if sep_v > PS_w and min(2 + sep_v - PS_w, cores[PX[PS_w:sep_v]].max()) > max_cover:
                    # Get bound based on coloring
                    color_bound = get_color_size(PX[PS_w:sep_v], colors_v, Fbuf, stack)
                    if 2 + color_bound > max_cover:
                        # if verbose: indent = '\t'
                        # if verbose: print indent, 'v:', v, 'w:', w, 'max_cover:', max_cover, 'P_w:', PX[PS_w:sep_v], 'cores[P_w]:', cores[PX[PS_w:sep_v]]
                        Rbuf[1] = w
                        tree[0, stats[0]+1] = curr_node_v
                        C, CP, CN, tree, _, sub_cover = cont(GI, GS, GE, PX, pos, PS_w, sep_v, XE_w,
                                                             Rbuf, 2, cores, C, CP, CN, cover, max_cover, rank,
                                                             stats, tree, colors_v, Fbuf, stack)
                        if sub_cover > max_cover:
                            cover[v,w] = max(max_cover, sub_cover)
                            cover[w,v] = max(max_cover, sub_cover)

                        sep_v -= 1
                        swap_pos(PX, pos, w, sep_v)
                    
        in_dG_v[dG_v] = False
        GE[P_v] = prev_GE

    C, CP, CN = trim_cliques(C, CP, CN)
    tree = tree[:,:stats[0]+1]    
    return C, CP, CN, stats, tree

# ################################
# # Branch on all max cores

# @jit(nopython=True, cache=cache)
# def cont(GI, GS, GE, PX, pos, PS, sep, XE, Rbuf, RE, cores, C, CP, CN, hashes, key, degen_order_idx):
#     ## Greedily select the vertex with the largest k-core number

#     P = PX[PS:sep]
#     X = PX[sep:XE]

#     if verbose:
#         indent = '\t'*(RE-1)
#         print indent, '---cont----'
#         print indent, 'R:', Rbuf[:RE], 'P:', P, 'cores[P]:', cores[P], 'X:', X
        
#     if P.size==0:
#         if X.size==0:
#             C, CP, CN = update_cliques(C, CP, CN, Rbuf[:RE])
#             tmp = Rbuf[:RE].copy()
#             tmp.sort()
#             if verbose: print indent, 'Return:', tmp
#         return C, CP, CN
    
#     #######
#     # TODO: Consider X in order to filter out non-/maximal cliques

#     max_core = cores[P].max()
#     branches = P[cores[P]==max_core]
#     Fbuf[branches] = True
        
#     # tmp_sep = 0
#     # for b in branches:
#     #     swap_pos(P, pos, 
#     # first = branches[0]
#     # for v in GI[GS[first]:GE[first]]:
    
#     max_core = cores[P].max()
#     for v in P[cores[P]==max_core]:
#         new_PS, new_XE = update_P_fast(GI, GS, GE, PX, pos, sep, v)
#         Rbuf[RE] = v
#         C, CP, CN = cont(GI, GS, GE, PX, pos, new_PS, sep, new_XE, 
#                          Rbuf, RE+1, cores, C, CP, CN, hashes, key, degen_order_idx)
#         sep -= 1
#         swap_pos(PX, pos, v, sep)
                 
#     # if verbose:
#     #     print indent, 'pos:', pos
#     return C, CP, CN

# @jit(nopython=True, cache=cache)
# def update_P_fast(GI, GS, GE, PX, pos, PS, sep, XE, v):
#     new_PS, new_XE = sep, sep
#     for w in GI[GS[v]:GE[v]]:
#         w_pos = pos[w]
#         if (PS<=w_pos) and (w_pos<sep):
#             new_PS -= 1
#             swap_pos(PX, pos, w, new_PS)
#         elif (sep<=w_pos) and (w_pos<XE):
#             swap_pos(PX, pos, w, new_XE)
#             new_XE += 1
#     return new_PS, new_XE
        
# @jit(nopython=True, cache=cache)
# def approx_edges(GI, GS, GE, dGI, dGS, dGE):
#     n = GS.size
#     P = np.arange(n).astype(GS.dtype)
#     pos = P.copy()
#     Rbuf = np.empty(n, P.dtype)
#     in_dG_v = np.zeros(n, np.bool_)

#     degen_order, degen_deg, cores = get_degeneracy(GI, GS, GE, P)
#     degen_order_idx = np.argsort(degen_order)
#     C, CP, CN = np.empty(P.size, P.dtype), np.zeros(P.size, P.dtype), 0

#     stack = np.empty(n).astype(GS.dtype)
#     stack_end = 0
    
#     for v_i in range(n):
#         v = degen_order[v_i]
#         dG_v = dGI[dGS[v]:dGE[v]]
#         in_dG_v[dG_v] = True
#         Rbuf[0] = v        
#         PS_v, XE_v = update_P_fast(GI, GS, GE, PX, pos, 0, n, n, v)

#         # if sep_v > 0:
#         #     max_core = cores[P[:sep_v]].max()
            
#         #     first = P[sep_v - 1]
#         #     for w in GI[GS[first]:GE[first]]:
#         #         if cores[w]==cores[first]
#         #         swap_pos(P, pos, v, sep_v)
        
#         P_v = P[:sep_v].copy()
#         # P_v.sort()
#         # first = P_v[P_v.size - 1]
        
#         if verbose: print 'v:', v, 'P_v:', P_v
#         for w in P_v:
#             if in_dG_v[w] and degen_order_idx[w] > v_i:
#                 PS_w, XE_w = update_P_fast(GI, GS, GE, PX, pos, PS_v, sep_v, XE_v, w)
#                 Rbuf[1] = w
#                 hashes = set()
#                 key = np.random.random(n)
#                 C, CP, CN = cont(GI, GS, GE, PX, pos, PS_w, sep_w, XE_w,
#                                  Rbuf, 2, cores, C, CP, CN, hashes, key, degen_order_idx)
#         in_dG_v[dG_v] = False

#     C, CP, CN = trim_cliques(C, CP, CN)
#     return C, CP, CN


def MC_cover_approx_py(G, dG, verbose=False):
    ##
    # G : old edges
    # dG : new edges

    tmp = G + dG

    start = time.time()
    C, CP, CN, stats, tree = approx_edges(tmp.indices, tmp.indptr[:-1], tmp.indptr[1:].copy(),
                                          dG.indices, dG.indptr[:-1], dG.indptr[1:])
    print 'approx_edges time:', time.time() - start
    if verbose:
        print 'Found approx cliques:', CN
        #print sparse_str_I(C, CP[:-1], CP[1:])
    C, CP, CN, key = get_unique_cliques(C, CP, CN, G.shape[0])
    if verbose:
        print 'Found unique cliques:', CN
        #print key
        #print sparse_str_I(C, CP[:-1], CP[1:])
    cliques = cliques_to_csc(C, CP, CN, G.shape[0])

    unexplained = clixov_utils.get_unexplained_edges(cliques, dG)
    print 'Unexplained edges:', zip(*unexplained.nonzero())

    cliques_idx, cover_G = get_largest_clique_covers(cliques, dG, ret_edges=True, assert_covered=False)

    print 'stats:', stats[0]
    
    return cliques[:,cliques_idx], cover_G, tree
