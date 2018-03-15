import numpy as np
import numpy.random
from numba import jit

from constants import cache

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
    Fbuf = np.zeros(k, np.bool_)
    Tbuf = np.ones(k, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])
    btw_new = np.zeros(PX.size, np.bool_)
    btw_stack = np.arange(PX.size).astype(np.int32)
    btw_end = 0

    C, CP, CN = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    CP[:2] = 0

    stats = np.array([0], np.int32)
    tree = np.asfortranarray(np.zeros(1000, np.int32))

    return PX, pos, R, R_end, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree


@jit(nopython=True, cache=cache)
def trim_cliques(cliques, cliques_indptr, cliques_n):
    return cliques[:cliques_indptr[cliques_n]], cliques_indptr[:cliques_n+1], cliques_n

@jit(nopython=True, cache=cache)
def swap_pos(PX, pos, w, PX_idx):
    pos_w = pos[w]
    b = PX[PX_idx]
    PX[PX_idx], PX[pos_w] = w, b
    pos[w], pos[b] = PX_idx, pos_w

@jit(nopython=True, cache=cache)
def update_cliques(C, CP, CN, R):    
    next_CP = CP[CN] + R.size
    if next_CP > C.size:
        C = np.concatenate((C, np.empty(max(next_CP - C.size, 2 * C.size), C.dtype)))
    new_CN = CN + 1
    if CN > CP.size - 2:
        CP = np.concatenate((CP, np.empty(max(new_CN - (CP.size-1), 2 * CP.size), np.int32)))
        
    C[CP[CN]: next_CP] = R
    CP[new_CN] = CP[CN] + R.size
    return C, CP, new_CN

@jit(nopython=True, cache=cache)
def move_PX(GI, GS, GE, pos, oldPS, oldXE, PS, XE, sep, v_degree, v, curr):
    # Move P and X to the bottom
    for w_i in range(GS[v], GE[v]):
        w = GI[w_i]
        w_pos = pos[w]
        #print '\t', 'w, w_pos, w_i:', w, w_pos, w_i
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
def move_PX_fast_bool_unused(GI, GS, GE, in_P, used, v):
    """Assumes X is not used, and that all v from GS to GE must be
    traversed"""
    
    # Move P and X to the bottom
    curr = 0
    G_tmp = GI[GS[v] : GE[v]]
    used_v = used[:,v]
    used_v_T = used[v,:]
    for w_i in range(G_tmp.size):
        w = G_tmp[w_i]
        if in_P[w] and (not used_v[w]) and (not used_v_T[w]):
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
    ### Modifies GE[u] through move_PX_fast_bool
    ### 
    
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

@jit(nogil=True)
def get_unique_cliques_exact(C, CP, CN):
    # Filter for a set of unique cliques. Slow implementation because
    # cannot converts numpys into tuples using nopython=True mode.
    
    n = (CP[1:] - CP[:-1]).max()
    D, DP, DN = np.empty(n, C.dtype), np.zeros(n, C.dtype), 0

    unique_cliques = set()
    for r_i in range(CN):
        r = C[CP[r_i]:CP[r_i+1]]
        r = r.copy()
        r.sort()        
        r_tup = tuple(r)
        if r_tup not in unique_cliques:
            unique_cliques.add(r_tup)
            D, DP, DN = update_cliques(D, DP, DN, r)
        
    D, DP, DN = trim_cliques(D, DP, DN)
    return D, DP, DN

@jit(nopython=True, cache=cache)
def get_unique_cliques(C, CP, CN, n):
    # Filter for a set of unique cliques. Fast implementation using
    # randomized algorithm for computing a hash for each clique. Has
    # small chance of failing because of colliding hashes

    # if n is None:
    #     n = C.max() + 1
    key = np.random.random(n)
    D, DP, DN = np.empty(n, C.dtype), np.zeros(n, C.dtype), 0

#    print key
    hash_table = np.empty(CN, np.float64)
    for r_i in range(CN):
        r = C[CP[r_i]:CP[r_i+1]]
        r_hash = 0
        for rr in r:
#            print rr, key.size
            r_hash += key[rr]
#        print r, r_hash
        hash_table[r_i] = r_hash
    hash_table_idx = np.argsort(hash_table)

    # print 'key:', key
    # print 'CN:', CN, hash_table
    if CN > 0:
        i = hash_table_idx[0]
        D, DP, DN = update_cliques(D, DP, DN, C[CP[i]:CP[i+1]])

    for ii in range(1, CN):
        i = hash_table_idx[ii]
        j = hash_table_idx[ii-1]
        if hash_table[i]!=hash_table[j]:
            D, DP, DN = update_cliques(D, DP, DN, C[CP[i]:CP[i+1]])
        
    D, DP, DN = trim_cliques(D, DP, DN)
    return D, DP, DN, key
