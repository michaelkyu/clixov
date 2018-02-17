import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def swap_pos(PX, pos, w, PX_idx):
    pos_w = pos[w]
    b = PX[PX_idx]
    PX[PX_idx], PX[pos_w] = w, b
    pos[w], pos[b] = PX_idx, pos_w

@jit(nopython=True, cache=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
def expand_2d_arr(arr):
    arr2 = np.zeros((2*arr.shape[1], arr.shape[0]), arr.dtype).T
    arr2[:,:arr.shape[1]] = arr
    return arr2


# In[3]:

# Assumes X is not used, and that all v from G_start to G_end must be trversed
@jit(nopython=True)
def move_PX_fast(G_indices, G_start, G_end, pos, PS, sep, v):
    # Move P and X to the bottom
    curr = G_start[v]
    for w_i in range(G_start[v], G_end[v]):
        w = G_indices[w_i]
        w_pos = pos[w]
        if PS <= w_pos and w_pos < sep:
            G_indices[curr], G_indices[w_i] = w, G_indices[curr]
            curr += 1
    return curr - G_start[v], curr

# Assumes X is not used, and that all v from G_start to G_end must be trversed
@jit(nopython=True)
def move_PX_fast_bool(G_indices, G_start, G_end, in_P, v):
    # Move P and X to the bottom
    curr = 0
    G_tmp = G_indices[G_start[v] : G_end[v]]
    for w_i in range(G_tmp.size):
        if in_P[G_tmp[w_i]]:
            G_tmp[curr], G_tmp[w_i] = G_tmp[w_i], G_tmp[curr]
            curr += 1
    return curr, G_start[v] + curr

@jit(nopython=True)
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


# ## Degeneracy

# In[4]:

@jit(nopython=True)
def get_degeneracy(G_indices, G_start, G_end, P):
    # P is the nodes to calculate a degeneracy ordering
    
    n = G_start.size
    deg_indices = np.empty(n * n, np.int32)
    deg_start = np.arange(0, n * n, n)
    deg_end = np.arange(0, n * n + n, n) - 1
    degrees = np.zeros(n, np.int32)
    degrees[P] = G_end[P] - G_start[P]
    
    pos = np.empty(n, np.int32)
    min_deg = 100000000
    for v in P:
        v_deg = degrees[v]
        deg_end[v_deg] += 1
        deg_indices[deg_end[v_deg]] = v
        pos[v] = deg_end[v_deg]
        min_deg = min(v_deg, min_deg)
    
    degen_deg = np.zeros(P.size, np.int32)
    degen_order = np.zeros(P.size, np.int32)
    for i in range(P.size):
        for min_deg in range(n):
            if deg_end[min_deg] >= deg_start[min_deg]:
                break
                
        v = deg_indices[deg_end[min_deg]]
        deg_end[min_deg] -= 1
        degrees[v] = 0
        degen_deg[i] = min_deg
        degen_order[i] = v
        
        if min_deg > 0:
            for w in G_indices[G_start[v] : G_end[v]]:            
                w_deg = degrees[w]
                if w_deg > 0:
                    w_pos = pos[w]
                    u = deg_indices[deg_end[w_deg]]
                    pos[u] = w_pos
                    deg_indices[w_pos], deg_indices[deg_end[w_deg]] = u, deg_indices[w_pos]
                    degrees[w] -= 1
                    deg_end[w_deg] -= 1

                    w_deg -= 1
                    deg_end[w_deg] += 1
                    deg_indices[deg_end[w_deg]] = w
                    pos[w] = deg_end[w_deg]                
                    
    return degen_order, degen_deg


# ## Color nodes

# In[5]:

@jit(nopython=True)
def color_nodes(G_indices, G_start, G_end, order):
    colors = np.empty(order.size, np.int32)
    branches = np.empty(order.size, np.int32)
    i = 0
    c = 1
    
    uncolored = np.ones(order.max() + 1, np.bool_)
    uncolored_next = 0
    
    while i < order.size:
        colorable = uncolored.copy()
        for j in range(uncolored_next, order.size):
            v = order[j]
            if colorable[v]:
                colors[i] = c
                branches[i] = v
                i += 1
                
                if j == uncolored_next:
                    uncolored_next += 1
                    
                uncolored[v] = False
                for w in G_indices[G_start[v] : G_end[v]]:
                    colorable[w] = False
        c += 1
    return colors, branches


# ## Tree Bookkeeping

# In[7]:

@jit(nopython=True)
def update_tree_size(tree_size, depth, max_cover):
    curr_tree_size = 4 + tree_size[0, 0]
    if tree_size.shape[1] <= curr_tree_size:
        tree_size = expand_2d_arr(tree_size)
    tree_size[0, curr_tree_size] = depth
    tree_size[1, curr_tree_size] = max_cover
    tree_size[0, 0] += 1
    
    return tree_size, curr_tree_size

@jit(nopython=True)
def update_tree_size_branch(tree_size, curr_tree_size, v, bound, P_size):
    sub_tree_size = 4 + tree_size[0, 0]
    if tree_size.shape[1] <= sub_tree_size:  tree_size = expand_2d_arr(tree_size)
    tree_size[2, sub_tree_size] = curr_tree_size
    tree_size[3, sub_tree_size] = v
    tree_size[4, sub_tree_size] = bound
    tree_size[5, sub_tree_size] = P_size
    return tree_size


# # Maximum Clique

# In[8]:

@jit(nopython=True)
def update_P(G_indices, G_start, G_end, PX, old_sep, sep, pos, v):
    new_sep = 0
        
    for u in G_indices[G_start[v] : G_end[v]]:
        if pos[u] >= old_sep:
            break
        elif pos[u] < sep:
            swap_pos(PX, pos, u, new_sep)
            new_sep += 1   
    return new_sep, PX[ : new_sep]


# In[9]:

@jit(nopython=True)
def reduce_G(G_indices, G_start, G_end, PXbuf, core_nums, core_bound, PX, pos, sep):
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


@jit(nopython=True)
def do_color(G_indices, G_start, G_end, order, k):
    in_P = np.zeros(k, np.bool_)
    in_P[order] = True
    
    colors = np.empty(order.size, np.int32)
    branches = np.empty(order.size, np.int32)
    i = 0
    c = 1
    
    uncolored = np.ones(order.max() + 1, np.bool_)
    uncolored_next = 0
    
    while i < order.size:
        colorable = uncolored.copy()
        for j in range(uncolored_next, order.size):
            v = order[j]
            if colorable[v]:
                colors[i] = c
                branches[i] = v
                i += 1
                
                if j == uncolored_next:
                    uncolored_next += 1
                    
                uncolored[v] = False
                for w in G_indices[G_start[v] : G_end[v]]:
                    if in_P[w]:
                        colorable[w] = False
        c += 1
    return colors, branches
