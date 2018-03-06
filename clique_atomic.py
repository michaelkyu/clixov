import numpy as np
from numba import jit

from constants import cache

#@jit(nopython=True, cache=cache):
def initialize_structures(k, PX=None):
    if PX is None:
        PX = np.arange(k).astype(np.int32)
        pos = np.empty(PX.size, np.int32)
        pos[PX] = np.arange(PX.size)
    else:
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

    return PX, pos, R, R_end, sep, PS, sep, XE, PXbuf, PXbuf2, C, CP, CN, btw_new, btw_stack, btw_end

@jit(nopython=True, cache=cache)
def swap_pos(PX, pos, w, PX_idx):
    pos_w = pos[w]
    b = PX[PX_idx]
    PX[PX_idx], PX[pos_w] = w, b
    pos[w], pos[b] = PX_idx, pos_w

@jit(nopython=True, cache=cache)
def update_cliques2(cliques, cliques_indptr, cliques_n, R):
    new_cliques_end = cliques_indptr[cliques_n] + R.size
    if new_cliques_end > cliques.size:
        cliques = np.concatenate((cliques, np.empty(max(new_cliques_end - cliques.size, 2 * cliques.size), cliques.dtype)))
    new_cliques_n = cliques_n + 1
    if cliques_n > cliques_indptr.size - 2:
        cliques_indptr = np.concatenate((cliques_indptr, np.empty(max(new_cliques_n - (cliques_indptr.size-1), 2 * cliques_indptr.size), np.int32)))
        
    cliques[cliques_indptr[cliques_n]: new_cliques_end] = R
    cliques_indptr[new_cliques_n] = cliques_indptr[cliques_n] + R.size
    return cliques, cliques_indptr, new_cliques_n

@jit(nopython=True, cache=cache)
def move_PX(GI, GS, GE, pos, oldPS, oldXE, PS, XE, sep, v_degree, v, curr):
    # Move P and X to the bottom
    for w_i in range(GS[v], GE[v]):
        w = GI[w_i]
        w_pos = pos[w]
        if w_pos < oldPS or w_pos >= oldXE:
            break
        elif PS <= w_pos and w_pos < XE:
            v_degree += w_pos < sep
            GI[curr], GI[w_i] = w, GI[curr]
            curr += 1
    return v_degree, curr

@jit(nopython=True, cache=cache)
def move_PX_2(GI, GS, GE, pos, oldPS, oldXE, PS, XE, sep, v_degree, v, curr):
    # Move P and X to the bottom
    for w_i in range(GS[v], GE[v], 2):
        w = GI[w_i]
        w_pos = pos[w]
        if w_pos < oldPS or w_pos >= oldXE:
            break
        elif PS <= w_pos and w_pos < XE:
            v_degree += w_pos < sep
            GI[curr], GI[w_i] = w, GI[curr]
            curr += 2
    return v_degree, curr

@jit(nopython=True, cache=cache)
def update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE):
    for w in GI[GS[v] : GE[v]]:
        w_pos = pos[w]
        if (PS <= w_pos) and (w_pos < sep):
            new_PS -= 1
            swap_pos(PX, pos, w, new_PS)
        elif (sep <= w_pos) and (w_pos < XE):
            swap_pos(PX, pos, w, new_XE)
            new_XE += 1
        else:
            break
    return new_PS, new_XE

@jit(nopython=True, cache=cache)
def update_PX_skip(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE):
    for w in GI[GS[v] : GE[v]]:
        w_pos = pos[w]
        if (PS <= w_pos) and (w_pos < sep):
            new_PS -= 1
            swap_pos(PX, pos, w, new_PS)
        elif (sep <= w_pos) and (w_pos < XE):
            swap_pos(PX, pos, w, new_XE)
            new_XE += 1
        elif w_pos < oldPS or w_pos >= oldXE:
            break
    return new_PS, new_XE

@jit(nopython=True, cache=cache)
def expand_2d_arr(arr):
    arr2 = np.zeros((2*arr.shape[1], arr.shape[0]), arr.dtype).T
    arr2[:,:arr.shape[1]] = arr
    return arr2

@jit(nopython=True, cache=cache)
def expand_1d_arr(arr):
    arr2 = np.zeros(2*arr.size, arr.dtype)
    arr2[:arr.size] = arr
    return arr2

@jit(nopython=True, cache=cache)
def move_PX_fast(GI, GS, GE, pos, PS, sep, v):
    """Assumes X is not used, and that all v from GS to GE must be
    traversed.

    """
    
    # Move P and X to the bottom
    curr = GS[v]
    for w_i in range(GS[v], GE[v]):
        w = GI[w_i]
        w_pos = pos[w]
        if PS <= w_pos and w_pos < sep:
            GI[curr], GI[w_i] = w, GI[curr]
            curr += 1
    return curr - GS[v], curr


@jit(nopython=True, cache=cache)
def move_PX_fast_bool(GI, GS, GE, in_P, v):
    """Assumes X is not used, and that all v from GS to GE must be
    traversed"""
    
    # Move P and X to the bottom
    curr = 0
    G_tmp = GI[GS[v] : GE[v]]
    for w_i in range(G_tmp.size):
        if in_P[G_tmp[w_i]]:
            G_tmp[curr], G_tmp[w_i] = G_tmp[w_i], G_tmp[curr]
            curr += 1
    return curr, GS[v] + curr

@jit(nopython=True, cache=cache)
def update_PX_bool_color(GI, GS, GE,
                         GI_new, GS_new, GE_end,
                         nei_bool, colors_v, in_P, pos, PX, sep, v):
    new_PS = sep
    nei_count = 0
    for w in GI[GS[v] : GE[v]]:
        if in_P[w]:
            new_PS -= 1
            swap_pos(PX, pos, w, new_PS)
            w_color = colors_v[w]
            nei_count += not nei_bool[w_color]
            nei_bool[w_color] = True
    for w in GI_new[GS_new[v] : GE_end[v]]:
        if in_P[w]:
            new_PS -= 1
            swap_pos(PX, pos, w, new_PS)
            w_color = colors_v[w]
            nei_count += not nei_bool[w_color]
            nei_bool[w_color] = True
    nei_bool[colors_v[PX[new_PS : sep]]] = False
    return new_PS, nei_count

@jit(nopython=True, cache=cache)
def update_tree_size(tree_size, depth, max_cover):
    curr_tree_size = 4 + tree_size[0, 0]
    if tree_size.shape[1] <= curr_tree_size:
        tree_size = expand_2d_arr(tree_size)
    tree_size[0, curr_tree_size] = depth
    tree_size[1, curr_tree_size] = max_cover
    tree_size[0, 0] += 1
    
    return tree_size, curr_tree_size

@jit(nopython=True, cache=cache)
def update_tree_size_branch(tree_size, curr_tree_size, v, bound, P_size):
    sub_tree_size = 4 + tree_size[0, 0]
    if tree_size.shape[1] <= sub_tree_size:  tree_size = expand_2d_arr(tree_size)
    tree_size[2, sub_tree_size] = curr_tree_size
    tree_size[3, sub_tree_size] = v
    tree_size[4, sub_tree_size] = bound
    tree_size[5, sub_tree_size] = P_size
    return tree_size

@jit(nopython=True, cache=cache)
def update_P(GI, GS, GE, PX, old_sep, sep, pos, v):
    new_sep = 0
        
    for u in GI[GS[v] : GE[v]]:
        if pos[u] >= old_sep:
            break
        elif pos[u] < sep:
            swap_pos(PX, pos, u, new_sep)
            new_sep += 1   
    return new_sep, PX[ : new_sep]

@jit(nopython=True, cache=cache)
def reduce_G(GI, GS, GE,
             PXbuf, core_nums, core_bound, PX, pos, sep):
    changed = False
    for u in PX[ : sep]:
        if core_nums[u] < core_bound:
            sep -= 1
            swap_pos(PX, pos, u, sep)
            changed = True
    P = PX[ : sep]
    
    if changed:
        # Push down GI
        PXbuf[P] = True
        for u in P:
            u_degree, GE[u] = move_PX_fast_bool(GI, GS, GE, PXbuf, u)
        PXbuf[P] = False
    return sep, P
