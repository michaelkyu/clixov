import numpy as np
from numba import jit

from constants import cache

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
def move_PX(G_indices, G_start, G_end, pos, oldPS, oldXE, PS, XE, sep, v_degree, v, curr):
    # Move P and X to the bottom
    for w_i in range(G_start[v], G_end[v]):
        w = G_indices[w_i]
        w_pos = pos[w]
        if w_pos < oldPS or w_pos >= oldXE:
            break
        elif PS <= w_pos and w_pos < XE:
            v_degree += w_pos < sep
            G_indices[curr], G_indices[w_i] = w, G_indices[curr]
            curr += 1
    return v_degree, curr

@jit(nopython=True, cache=cache)
def move_PX_2(G_indices, G_start, G_end, pos, oldPS, oldXE, PS, XE, sep, v_degree, v, curr):
    # Move P and X to the bottom
    for w_i in range(G_start[v], G_end[v], 2):
        w = G_indices[w_i]
        w_pos = pos[w]
        if w_pos < oldPS or w_pos >= oldXE:
            break
        elif PS <= w_pos and w_pos < XE:
            v_degree += w_pos < sep
            G_indices[curr], G_indices[w_i] = w, G_indices[curr]
            curr += 2
    return v_degree, curr

@jit(nopython=True, cache=cache)
def update_PX(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE):
    for w in G_indices[G_start[v] : G_end[v]]:
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
def update_PX_skip(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE):
    for w in G_indices[G_start[v] : G_end[v]]:
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
def move_PX_fast(G_indices, G_start, G_end, pos, PS, sep, v):
    """Assumes X is not used, and that all v from G_start to G_end must be
    traversed.

    """
    
    # Move P and X to the bottom
    curr = G_start[v]
    for w_i in range(G_start[v], G_end[v]):
        w = G_indices[w_i]
        w_pos = pos[w]
        if PS <= w_pos and w_pos < sep:
            G_indices[curr], G_indices[w_i] = w, G_indices[curr]
            curr += 1
    return curr - G_start[v], curr


@jit(nopython=True, cache=cache)
def move_PX_fast_bool(G_indices, G_start, G_end, in_P, v):
    """Assumes X is not used, and that all v from G_start to G_end must be
    traversed"""
    
    # Move P and X to the bottom
    curr = 0
    G_tmp = G_indices[G_start[v] : G_end[v]]
    for w_i in range(G_tmp.size):
        if in_P[G_tmp[w_i]]:
            G_tmp[curr], G_tmp[w_i] = G_tmp[w_i], G_tmp[curr]
            curr += 1
    return curr, G_start[v] + curr

@jit(nopython=True, cache=cache)
def update_PX_bool_color(G_indices, G_start, G_end,
                         Gnew_indices, Gnew_start, Gnew_end,
                         nei_bool, colors_v, in_P, pos, PX, sep, v):
    new_PS = sep
    nei_count = 0
    for w in G_indices[G_start[v] : G_end[v]]:
        if in_P[w]:
            new_PS -= 1
            swap_pos(PX, pos, w, new_PS)
            w_color = colors_v[w]
            nei_count += not nei_bool[w_color]
            nei_bool[w_color] = True
    for w in Gnew_indices[Gnew_start[v] : Gnew_end[v]]:
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
def update_P(G_indices, G_start, G_end, PX, old_sep, sep, pos, v):
    new_sep = 0
        
    for u in G_indices[G_start[v] : G_end[v]]:
        if pos[u] >= old_sep:
            break
        elif pos[u] < sep:
            swap_pos(PX, pos, u, new_sep)
            new_sep += 1   
    return new_sep, PX[ : new_sep]

@jit(nopython=True, cache=cache)
def reduce_G(G_indices, G_start, G_end,
             PXbuf, core_nums, core_bound, PX, pos, sep):
    changed = False
    for u in PX[ : sep]:
        if core_nums[u] < core_bound:
            sep -= 1
            swap_pos(PX, pos, u, sep)
            changed = True
    P = PX[ : sep]
    
    if changed:
        # Push down G_indices
        PXbuf[P] = True
        for u in P:
            u_degree, G_end[u] = move_PX_fast_bool(G_indices, G_start, G_end, PXbuf, u)
        PXbuf[P] = False
    return sep, P
