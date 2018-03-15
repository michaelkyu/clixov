import numpy as np
from numba import jit
from constants import cache

# @jit(nopython=True, cache=cache)
# def get_degeneracy_ordering_sp(GS, GE, GI):
#     n = GS.size
#     GE = GE.copy()
#     degrees = GE - GS
#     order = np.arange(n).astype(np.int32)
#     degen = 0
#     degen_vec = np.zeros(n, np.int32)
#     default_deg = 10000000
    
#     for i in range(n):
#         min_deg = default_deg
#         for w_idx in range(i, n):
#             w = order[w_idx]
#             w_deg = degrees[w]
#             if w_deg > 0 and w_deg < min_deg:
#                 min_deg, v, v_idx = w_deg, w, w_idx
#         if min_deg == default_deg:
#             break
        
#         v_nei = GI[GS[v] : GE[v]]
#         for w in v_nei:
#             w_nei = GI[GS[w] : GE[w]]
#             v_i = (w_nei == v).nonzero()[0][0]
            
#             # This changes GI, so maybe need to copy it?
#             w_nei[v_i], w_nei[-1] = w_nei[-1], w_nei[v_i]
#             GE[w] -= 1
                    
#         degen_vec[i] = min_deg
#         degrees[v] -= v_nei.size
#         degrees[v_nei] -= 1
#         order[i], order[v_idx] = v, order[i]
    
#     return order, degen_vec

# @jit(nopython=True, cache=cache)
# def get_degeneracy_max(GS, GE, GI):
#     n = GS.size
#     deg_indices = np.empty(n * n, np.int32)
#     deg_start = np.arange(0, n * n, n)
#     deg_end = np.arange(0, n * n + n, n) - 1
    
#     degrees = GE - GS
#     pos = np.empty(n, np.int32)
#     max_deg = 0

#     for v in range(n):        
#         v_deg = degrees[v]  
#         deg_end[v_deg] += 1
#         deg_indices[deg_end[v_deg]] = v
#         pos[v] = deg_end[v_deg]
#         max_deg = max(v_deg, max_deg)
    
#     unused = np.ones(n, np.bool_)
    
#     degen_deg = np.zeros(n, np.int32)
#     degen_order = np.zeros(n, np.int32)
#     i = 0
#     while max_deg > 0:
#         v = deg_indices[deg_end[max_deg]]
#         deg_end[max_deg] -= 1
#         degrees[v] = 0

#         degen_deg[i] = max_deg
#         degen_order[i] = v
#         unused[v] = False
        
#         for w in GI[GS[v] : GE[v]]:            
#             w_deg = degrees[w]
#             if w_deg > 0:
#                 w_pos = pos[w]
#                 u = deg_indices[deg_end[w_deg]]
#                 pos[u] = w_pos
#                 deg_indices[w_pos], deg_indices[deg_end[w_deg]] = u, deg_indices[w_pos]
#                 degrees[w] -= 1
#                 deg_end[w_deg] -= 1                
#                 w_deg -= 1                
#                 if w_deg > 0:
#                     deg_end[w_deg] += 1
#                     deg_indices[deg_end[w_deg]] = w
#                     pos[w] = deg_end[w_deg]
        
#         while max_deg > 0 and deg_end[max_deg] < deg_start[max_deg]:
#             max_deg -= 1
#         i += 1

#         degen_deg[i:n] = 0
#     for v in range(n):        
#         if unused[v]:
#             degen_order[i] = v
#             i += 1
    
#     return degen_order, degen_deg

@jit(nopython=True, cache=cache)
def get_degeneracy_min(GS, GE, GI):
    """Version where you take the minimum degree, even if its 0"""
    
    n = GS.size
    deg_indices = np.empty(n * n, np.int32)
    deg_start = np.arange(0, n * n, n)
    deg_end = np.arange(0, n * n + n, n) - 1
    degrees = GE - GS
    
    pos = np.empty(n, np.int32)
    min_deg = 100000000    
    for v in range(n):        
        v_deg = degrees[v]  
        deg_end[v_deg] += 1
        deg_indices[deg_end[v_deg]] = v
        pos[v] = deg_end[v_deg]    
        min_deg = min(v_deg, min_deg)
    
    degen_deg = np.zeros(n, np.int32)
    degen_order = np.zeros(n, np.int32)
    for i in range(n):
        for min_deg in range(n):
            if deg_end[min_deg] >= deg_start[min_deg]:
                break
                
        v = deg_indices[deg_end[min_deg]]
        deg_end[min_deg] -= 1
        degrees[v] = 0        
        degen_deg[i] = min_deg
        degen_order[i] = v
        
        if min_deg > 0:
            for w in GI[GS[v] : GE[v]]:            
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

@jit(nopython=True, cache=cache)
def get_degeneracy_max(GS, GE, GI):
    n = GS.size
    deg_indices = np.empty(n * n, np.int32)
    deg_start = np.arange(0, n * n, n)
    deg_end = np.arange(0, n * n + n, n) - 1
    degrees = GE - GS
    
    pos = np.empty(n, np.int32)
    max_deg = 0    
    for v in range(n):        
        v_deg = degrees[v]  
        deg_end[v_deg] += 1
        deg_indices[deg_end[v_deg]] = v
        pos[v] = deg_end[v_deg]    
        max_deg = max(v_deg, max_deg)
    
    degen_deg = np.zeros(n, np.int32)
    degen_order = np.zeros(n, np.int32)
    for i in range(n):
        for max_deg in range(n-1, -1, -1):
            if deg_end[max_deg] >= deg_start[max_deg]:
                break
                
        v = deg_indices[deg_end[max_deg]]
        deg_end[max_deg] -= 1
        degrees[v] = 0
        
        degen_deg[i] = max_deg
        degen_order[i] = v
        
        if max_deg > 0:
            for w in GI[GS[v] : GE[v]]:            
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

@jit(nopython=True, cache=cache)
def get_degeneracy(GI, GS, GE, P):
    # P is the nodes to calculate a degeneracy ordering
    
    n = GS.size
    deg_indices = np.empty(n * n, np.int32)
    deg_start = np.arange(0, n * n, n)
    deg_end = np.arange(0, n * n + n, n) - 1
    degrees = np.zeros(n, np.int32)
    degrees[P] = GE[P] - GS[P]
    
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
            for w in GI[GS[v] : GE[v]]:            
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

    cores = degen_deg.copy()
    for ii in range(1, cores.size):
        i, j = degen_order[ii], degen_order[ii-1]
        cores[i] = max(cores[i], cores[j])

    return degen_order, degen_deg, cores

def test_degen():
    n = 10
    x = scipy.sparse.random(n, n, density=0.4, format='csr')
    x[np.arange(n), np.arange(n)] = 0
    x.eliminate_zeros()
    x.data = (x.data > 0).astype(np.int32)
    y = x + x.T
    y.data = (y.data > 0).astype(np.int32)
    print y.toarray()

    # %time degen_order, degen_deg = get_degeneracy_max(y.indptr[:-1], y.indptr[1:], y.indices)
    degen_order, degen_deg = get_degeneracy_min(y.indptr[:-1], y.indptr[1:], y.indices)
    print 'order:', degen_order
    print 'degree:', degen_deg

    tmp = as_dense_flat(dG.sum(1))
    # tmp
    degen_order, degen_deg = get_degeneracy_min(dG.indptr[:-1], dG.indptr[1:], dG.indices)
    # print ','.join(map(str,tmp[degen_order].astype(np.int32)))
    # print ','.join(map(str, degen_deg))
    print degen_order
    print degen_deg.max()
    degen_pos = np.empty(dG.shape[0], np.int32)
    degen_pos[degen_order] = np.arange(dG.shape[0]).astype(np.int32)
    print degen_pos[degen_order]
    # print ','.join(map(str,degen_deg))



