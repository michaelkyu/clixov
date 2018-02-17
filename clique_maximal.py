import time
import igraph
import numpy as np
from clique_atomic import *
from clixov_utils import trim_cliques

#interest = [1640, 2670, 4782, 6681, 6918]
#interest = [1640, 2670, 6681, 6918]
interest = []

@jit(nopython=True)
def BKPivotSparse2(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos, G_start, G_end, G_indices, PXbuf, depth,
                   cliques, cliques_indptr, cliques_n):
    R = R_buff[:R_end]    
    P, X = PX[PS:sep], PX[sep:XE]
    
#     indent = '\t' * depth
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X    
    
    if P.size==0:
        if X.size==0:
            cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R)
        return cliques, cliques_indptr, cliques_n

#     u = -1
#     max_degree = -1
#     within_P_degree = 0
#     for v in P:
#         v_degree, curr = move_PX(G_indices, G_start, G_end,
#                                  pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])                
#         if v_degree > max_degree:
#             max_degree, u, u_curr = v_degree, v, curr
#         within_P_degree += v_degree
#     max_X_to_P_degree = 0
#     for v in X:
#         v_degree, curr = move_PX(G_indices, G_start, G_end,
#                                  pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])                
#         if v_degree > max_degree:
#             max_degree, u, u_curr = v_degree, v, curr
#         max_X_to_P_degree = max(max_X_to_P_degree, v_degree)
#     if within_P_degree==P.size * (P.size -1):
#         if max_X_to_P_degree < P.size:
#             R_buff[R_end : R_end + P.size] = P
#             cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R_buff[:R_end + P.size])
#         return cliques, cliques_indptr, cliques_n

    u = -1
    max_degree = -1
    for v in PX[PS:XE]:
        v_degree, curr = move_PX(G_indices, G_start, G_end,
                                 pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])                
        if v_degree > max_degree:
            max_degree, u, u_curr = v_degree, v, curr

    Padj_u = pos[G_indices[G_start[u] : u_curr]]
    PXbuf[Padj_u] = False
    branches = P[PXbuf[PS:sep]]
    PXbuf[Padj_u] = True
    
    for v in branches:
        new_PS, new_XE = update_PX(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)            
        R_buff[R_end] = v
            
        cliques, cliques_indptr, cliques_n = BKPivotSparse2(R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                                                            G_start, G_end, G_indices, PXbuf, depth+1,
                                                            cliques, cliques_indptr, cliques_n)
        
        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    # Move back all branches into P
    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1
    
    return cliques, cliques_indptr, cliques_n        

def BKPivotSparse2_wrapper(G, PX=None):
    if PX is None:
        k = G.shape[0]
        PX = np.arange(k).astype(np.int32)
        pos = np.empty(PX.size, np.int32)
        pos[PX] = np.arange(PX.size)
    else:
        k = G.shape[0]
        pos = np.empty(k, np.int32)
        pos[:] = -1
        pos[PX] = np.arange(PX.size)        
        initialize_PX(G.indices, G.indptr[:-1], G.indptr[1:], pos, PX)
        
    R = np.zeros(PX.size, np.int32)
    R_end = np.int32(0)    
    sep = PX.size
    PXbuf = np.zeros(k, np.bool_)
    PXbuf2 = np.ones(k, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])            
    cliques, cliques_indptr, cliques_n = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0
    
    cliques, cliques_indptr, cliques_n = BKPivotSparse2(R, R_end, PX, PS, sep, XE, PS, XE, pos,
                                                         G.indptr[:-1], G.indptr[1:], G.indices,
                                                         PXbuf2, 0, cliques, cliques_indptr, cliques_n)

    cliques, cliques_indptr, cliques_n = trim_cliques(cliques, cliques_indptr, cliques_n)
    return cliques, cliques_indptr, cliques_n

@jit(nopython=True)
def BKPivotSparse2_Gsep(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                        G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
                        PXbuf, depth, cliques, cliques_indptr, cliques_n, tree_size):
    if tree_size.size <= 4 + tree_size[0]:
        tree_size = np.concatenate((tree_size, np.empty(tree_size.size, tree_size.dtype)))
    tree_size[2+tree_size[0]] = depth
    tree_size[0] += 1
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]
    
#     indent = '\t' * depth
#     print indent, '---------------'
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX`
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X
    
    if P.size==0:
        tree_size[1] += 1
        if X.size==0:
            cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R)
        return cliques, cliques_indptr, cliques_n, tree_size

    u = -1
    max_degree = -1
    for v in PX[PS:XE][::-1]:
        v_degree, curr_new = move_PX(Gnew_indices, Gnew_start, Gnew_end,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, Gnew_start[v])        
        v_degree, curr = move_PX(G_indices, G_start, G_end,
                                     pos, oldPS, oldXE, PS, XE, sep, v_degree, v, G_start[v])   
        if v_degree > max_degree: 
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new

#     # Swap the pivot to the beginning of P so that it is the first branch
#     if pos[u] < sep:
#         swap_pos(PX, pos, u, PS)
        
    Padj_u = pos[G_indices[G_start[u] : u_curr]]
    Padj_new_u = pos[Gnew_indices[Gnew_start[u] : u_curr_new]]
    
    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
    branches = P[PXbuf[PS:sep]]
    PXbuf[Padj_u] = True
    PXbuf[Padj_new_u] = True
    
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'Padj_new_u:', Padj_new_u
#     print indent, 'branches:', branches

    for v in branches:
#         print indent, 'branching at v:', v
        
        new_PS, new_XE = update_PX(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        new_PS, new_XE = update_PX(Gnew_indices, Gnew_start, Gnew_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)

        R_buff[R_end] = v

        cliques, cliques_indptr, cliques_n, tree_size = BKPivotSparse2_Gsep(R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                                                                G_start, G_end, G_indices,
                                                                Gnew_start, Gnew_end, Gnew_indices,
                                                                PXbuf, depth+1,
                                                                cliques, cliques_indptr, cliques_n, tree_size)        

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return cliques, cliques_indptr, cliques_n, tree_size

def BKPivotSparse2_Gsep_wrapper(Gold, Gnew):
    k = Gold.shape[0]
    R = np.zeros(k, np.int32)
    R_end = np.int32(0)
    PX = np.arange(k).astype(np.int32)
    sep = PX.size
    pos = np.empty(PX.size, np.int32)
    pos[PX] = np.arange(PX.size)
    PXbuf = np.zeros(PX.size, np.bool_)
    PXbuf2 = np.ones(PX.size, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])
    
    cliques, cliques_indptr, cliques_n = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0    
    
    cliques, cliques_indptr, cliques_n, tree_size = BKPivotSparse2_Gsep(R, R_end, PX, PS, sep, XE, PS, XE, pos,
                                                         Gold.indptr[:-1], Gold.indptr[1:], Gold.indices,
                                                         Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices,
                                                         PXbuf2, 0, cliques, cliques_indptr, cliques_n)
    cliques, cliques_indptr, cliques_n = trim_cliques(cliques, cliques_indptr, cliques_n)
    return cliques, cliques_indptr, cliques_n


@jit(nopython=True)
def BKPivotSparse2_Gnew(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                        G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
                        PXbuf, depth,
                        between_new, between_stack, between_end,
                        cliques, cliques_indptr, cliques_n, tree_size):
    """
    between_new[v]: Is there a new edge crossing between R and node v in P?
    """
    if tree_size.size <= 4 + tree_size[0]:
        tree_size = np.concatenate((tree_size, np.empty(tree_size.size, tree_size.dtype)))
    tree_size[2+tree_size[0]] = depth
    tree_size[0] += 1
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]

#     indent = '\t' * depth
#     print indent, '---------------'
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X
#     print indent, 'between_new:', between_new.astype(np.int32)

    if P.size==0:
        tree_size[1] += 1
        return cliques, cliques_indptr, cliques_n, tree_size
    
    incd = np.empty(sep - PS, PX.dtype)
    incd_count = 0
    max_degree = -1
    X_incd = np.empty(XE - sep, PX.dtype)
    X_incd_count = 0
    
    # Iterate over new edges from X too
    for v in X:
        v_degree_new, curr_new = move_PX(Gnew_indices, Gnew_start, Gnew_end,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, Gnew_start[v])        
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new = v, curr_new
        if v_degree_new > 0:
            tmp, curr = move_PX(G_indices, G_start, G_end,
                                pos, oldPS, oldXE, PS, XE, sep, v_degree_new, v, G_start[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
    for v in P:
        v_degree_new, curr_new = move_PX(Gnew_indices, Gnew_start, Gnew_end,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, Gnew_start[v])        
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new, u_incd = v, curr_new, incd_count
        if v_degree_new > 0 or between_new[v]:
            incd[incd_count] = v
            incd_count += 1
            
    u_curr = G_end[u]
    
#     # Move the pivot to the beginning of incd, so that it'll be the first branch
#     if incd_count > 0 and pos[u] < sep:
#         incd[0], incd[u_incd] = incd[u_incd], incd[0]        

    if incd_count == 0:
        tree_size[1] += 1
        return cliques, cliques_indptr, cliques_n, tree_size
    incd = incd[:incd_count]
    X_incd = X_incd[:X_incd_count]
    
    PXbuf[incd] = False
    PXbuf[X_incd] = False
    new_incd = np.empty(XE - PS - incd.size, PX.dtype)
    new_incd_end = 0
    
    ## Should we iterate on X_incd to update PX for old edges?
    
    for v in incd:
        v_degree_old = 0
        curr = G_start[v]
        
        # Move P and X to the bottom
        for w_i in range(G_start[v], G_end[v]):
            w = G_indices[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            elif PS <= w_pos and w_pos < XE:
                v_degree_old += w_pos < sep
                G_indices[curr], G_indices[w_i] = w, G_indices[curr]
                curr += 1
                
                if PXbuf[w]:
                    new_incd[new_incd_end] = w
                    new_incd_end += 1
                    PXbuf[w] = False
                    
    PXbuf[incd] = True
    PXbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    PXbuf[new_incd] = True
    
    # Only needed if we eventually call Gsep. If only returning clique "branch roots" over new edges, then this is not needed
    for v in new_incd:
        v_degree_old, curr = move_PX(G_indices, G_start, G_end,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])
    
    #--------------------------------#

    Padj_u = G_indices[G_start[u] : u_curr]
    Padj_new_u = Gnew_indices[Gnew_start[u] : u_curr_new]    
    
    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
    
    # Always keep the between_new
    PXbuf[P[between_new[P]]] = True
    
    # Some vertices will not be reachable because we've already
    # removed a lot of nodes. Expand to find a vertex cover of incd.
    for v in incd[~ PXbuf[incd]]:
        if not PXbuf[v]:
            # By construction because of move_PX above, this will only
            # iterate over w's that are in incd
            for w in Gnew_indices[Gnew_start[v] : Gnew_end[v]]:
                w_pos = pos[w]
                PXbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
                
    branches = incd[PXbuf[incd]]

#     # Alternative to setting P_between_new to True and finding vertex cover of incd. Empirically slower
#     tmp = new_incd[pos[new_incd] < sep]
#     branches = np.concatenate((incd[PXbuf[incd]], tmp[PXbuf[tmp]]))

    PXbuf[Padj_new_u] = True
    PXbuf[Padj_u] = True
    
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'Padj_new_u:', Padj_new_u
#     print indent, 'incd:', incd
#     print indent, 'new_incd:', new_incd
#     print indent, 'branches:', branches
        
    for v in branches:
#         print indent, 'branching at:', v
                
        new_PS, new_XE = update_PX(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        
        between_added = 0
        for w in Gnew_indices[Gnew_start[v] : Gnew_end[v]]:
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)                
                if not between_new[w]:
                    between_stack[between_end + between_added] = w
                    between_added += 1
                    between_new[w] = True
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < oldPS or w_pos >= oldXE:
                break

        R_buff[R_end] = v
        if between_new[v]:
            cliques, cliques_indptr, cliques_n, tree_size = BKPivotSparse2_Gsep(R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                                                                    G_start, G_end, G_indices,
                                                                    Gnew_start, Gnew_end, Gnew_indices,
                                                                    PXbuf, depth+1,
                                                                    cliques, cliques_indptr, cliques_n, tree_size)
        else:
            cliques, cliques_indptr, cliques_n, tree_size = BKPivotSparse2_Gnew(R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                                                    G_start, G_end, G_indices,
                                                    Gnew_start, Gnew_end, Gnew_indices,
                                                    PXbuf, depth+1,
                                                    between_new, between_stack, between_end + between_added,
                                                    cliques, cliques_indptr, cliques_n, tree_size)
        # Reset the between_new
        between_new[between_stack[between_end : between_end + between_added]] = False

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return cliques, cliques_indptr, cliques_n, tree_size

def BKPivotSparse2_Gnew_wrapper(Gold, Gnew, PX=None, between_new=None):
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
        
        initialize_PX(Gold.indices, Gold.indptr[:-1], Gold.indptr[1:], pos, PX)
        initialize_PX(Gnew.indices, Gnew.indptr[:-1], Gnew.indptr[1:], pos, PX)
        
    R = np.zeros(PX.size, np.int32)
    R_end = np.int32(0)    
    sep = PX.size
    PXbuf = np.zeros(k, np.bool_)
    PXbuf2 = np.ones(k, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])
    
    between_stack = np.arange(PX.max() + 1).astype(np.int32)
    if between_new is None:
        between_new = np.zeros(PX.max() + 1, np.bool_)        
        between_end = 0
    else:
        between_end = between_new.sum()
        between_stack[:between_end] = between_new.nonzero()[0]

    cliques, cliques_indptr, cliques_n = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0
    
    tree_size = np.asfortranarray(np.zeros(1000, np.int32))
    tree_size[:2] = np.array([0, 0], np.int32)
    
    cliques, cliques_indptr, cliques_n, tree_size = BKPivotSparse2_Gnew(R, R_end, PX, PS, sep, XE, PS, XE, pos,
                                                         Gold.indptr[:-1], Gold.indptr[1:], Gold.indices,
                                                         Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices,
                                                         PXbuf2, 0, between_new, between_stack, between_end,
                                                        cliques, cliques_indptr, cliques_n, tree_size)
#     print 'tree_size:', tree_size[:2]

    cliques, cliques_indptr, cliques_n = trim_cliques(cliques, cliques_indptr, cliques_n)
    return cliques, cliques_indptr, cliques_n, tree_size


#@jit(nopython=True)
def BK_hier_Gsep(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                 G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,
                 H_start, H_end, H_indices,
                 topo,
                 PXbuf, depth, cliques, cliques_indptr, cliques_n, tree_size):
    if tree_size.size <= 4 + tree_size[0]:
        tree_size = np.concatenate((tree_size, np.empty(tree_size.size, tree_size.dtype)))
    tree_size[2+tree_size[0]] = depth
    tree_size[0] += 1

#    verbose = (depth > 0 and R_buff[0] in interest)
    verbose = False
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]
    
    indent = '\t' * depth

    if verbose:
        print indent, '----Gsep-----------'
        print indent, 'DEPTH:', depth
        print indent, 'PX:', PX
        print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
        print indent, 'R:', R, 'P:', P, 'X:', X

    # print indent, '---------------'
    # print indent, 'DEPTH:', depth
    # print indent, 'PX:', PX
    # print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    # print indent, 'R:', R, 'P:', P, 'X:', X
    
    if P.size==0:
        if verbose and R_buff[0] in interest:
            print indent, depth, 'Returning', 'R:', R, 'P:', P, 'X:', X
            
        tree_size[1] += 1
        if X.size==0:
            cliques, cliques_indptr, cliques_n = update_cliques2(cliques, cliques_indptr, cliques_n, R)
        return cliques, cliques_indptr, cliques_n, tree_size
        
    # Filter branches. Follow the topological sorting, going top-down
    # the hierarchy
    tmp = P[np.argsort(topo[P])]
    for v in tmp:
        curr = H_start[v]
        for w_i in range(H_start[v], H_end[v]):
            w = H_indices[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            #elif PS <= w_pos and w_pos < XE:
            elif PS <= w_pos and w_pos < sep:
                H_indices[curr], H_indices[w_i] = w, H_indices[curr]
                curr += 1

                # Remove descendants of v from being branches
                if PXbuf[v]:
                    PXbuf[w] = False

        if verbose:
            print indent, 'v:', v, 'H to curr:', H_indices[H_start[v] : curr]
            print indent, 'v:', v, 'H to end:', H_indices[H_start[v] : H_end[v]]
                    
    branches = P[PXbuf[P]]

    if verbose:
        print indent, 'branches:', branches.tolist()
        
    u = -1
    max_degree = -1
    for v in PX[PS:XE][::-1]:
        v_degree, curr_new = move_PX(Gnew_indices, Gnew_start, Gnew_end,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, Gnew_start[v])        
        v_degree, curr = move_PX(G_indices, G_start, G_end,
                                 pos, oldPS, oldXE, PS, XE, sep, v_degree, v, G_start[v])
        if PXbuf[v] and (v_degree > max_degree): 
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new

    PXbuf[P] = True
    assert np.all(PXbuf)
    
    Padj_u = pos[G_indices[G_start[u] : u_curr]]
    Padj_new_u = pos[Gnew_indices[Gnew_start[u] : u_curr_new]]

    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
#    branches = P[PXbuf[PS:sep]]
    branches = branches[PXbuf[branches]]
    PXbuf[Padj_u] = True
    PXbuf[Padj_new_u] = True
        
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'Padj_new_u:', Padj_new_u
#     print indent, 'branches:', branches

    if verbose:
        print indent, 'branches:', branches.tolist()

    branches = branches[np.argsort(topo[branches])]
    
    for v in branches:
#         print indent, 'branching at v:', v

        new_PS, new_XE = update_PX(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        new_PS, new_XE = update_PX(Gnew_indices, Gnew_start, Gnew_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)

        for w in H_indices[H_start[v] : H_end[v]]:
            w_pos = pos[w]
            if (new_PS <= w_pos) and (w_pos < sep):                
                swap_pos(PX, pos, w, new_PS)
                new_PS += 1
            # elif (sep <= w_pos) and (w_pos < XE):
            #     swap_pos(PX, pos, w, new_XE)
            #     new_XE += 1
            elif (w_pos < oldPS) or (w_pos >= oldXE):
                break

        # new_PS, new_XE = update_PX(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        # new_PS, new_XE = update_PX(Gnew_indices, Gnew_start, Gnew_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)
        
        R_buff[R_end] = v

        cliques, cliques_indptr, cliques_n, tree_size = BK_hier_Gsep(
            R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            G_start, G_end, G_indices,
            Gnew_start, Gnew_end, Gnew_indices,
            H_start, H_end, H_indices,
            topo,
            PXbuf, depth+1,
            cliques, cliques_indptr, cliques_n, tree_size)        
            
        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return cliques, cliques_indptr, cliques_n, tree_size

#@jit(nopython=True)
def BK_hier_Gnew(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                 G_start, G_end, G_indices, Gnew_start, Gnew_end, Gnew_indices,                 
                 H_start, H_end, H_indices,
                 topo,
                 PXbuf, depth,
                 between_new, between_stack, between_end,
                 cliques, cliques_indptr, cliques_n, tree_size):
    """
    between_new[v]: Is there a new edge crossing between R and node v in P?
    """
    if tree_size.size <= 4 + tree_size[0]:
        tree_size = np.concatenate((tree_size, np.empty(tree_size.size, tree_size.dtype)))
    tree_size[2+tree_size[0]] = depth
    tree_size[0] += 1
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]

#    verbose = depth==0 or (depth > 0 and R_buff[0] in interest)
    verbose = False

    indent = '\t' * depth
    if verbose:
        print indent, '----Gnew-----------'
        print indent, 'DEPTH:', depth
        print indent, 'PX:', PX
        print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
        print indent, 'R:', R, 'P:', P, 'X:', X
        print indent, 'between_new:', between_new.nonzero()[0]
        
#     print indent, '---------------'
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X
#     print indent, 'between_new:', between_new.astype(np.int32)

    if P.size==0:
        if verbose and (R_buff[0] in interest):
            print indent, 'Returning', 'R:', R, 'P:', P, 'X:', X

        tree_size[1] += 1
        return cliques, cliques_indptr, cliques_n, tree_size
    
    incd = np.empty(sep - PS, PX.dtype)
    incd_count = 0
    X_incd = np.empty(XE - sep, PX.dtype)
    X_incd_count = 0
    
    # Iterate over new edges from X too
    for v in X:
        v_degree_new, curr_new = move_PX(Gnew_indices, Gnew_start, Gnew_end,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, Gnew_start[v])        
        if v_degree_new > 0:
            tmp, curr = move_PX(G_indices, G_start, G_end,
                                pos, oldPS, oldXE, PS, XE, sep, v_degree_new, v, G_start[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
    for v in P:
        v_degree_new, curr_new = move_PX(Gnew_indices, Gnew_start, Gnew_end,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, Gnew_start[v])        
        if v_degree_new > 0 or between_new[v]:
            incd[incd_count] = v
            incd_count += 1

    if incd_count == 0:
        if verbose and (R_buff[0] in interest):
            print indent, 'Returning because incd.size==0', 'R:', R, 'P:', P, 'X:', X

        tree_size[1] += 1
        return cliques, cliques_indptr, cliques_n, tree_size
    incd = incd[:incd_count]
    X_incd = X_incd[:X_incd_count]

    is_incd = np.zeros(PX.size, np.bool_)
    is_incd[incd] = True
    
    # Filter branches. Follow the topological sorting, going top-down
    # the hierarchy
    #for v in incd[np.argsort(topo[incd])]:
    for v in P[np.argsort(topo[P])]:
        curr = H_start[v]
        for w_i in range(H_start[v], H_end[v]):
            w = H_indices[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            #elif PS <= w_pos and w_pos < XE:
            elif PS <= w_pos and w_pos < sep:
                H_indices[curr], H_indices[w_i] = w, H_indices[curr]
                curr += 1

                # Remove descendants of v from being branches
#                if PXbuf[v]:
                if is_incd[v] and PXbuf[v]:
                   PXbuf[w] = False

        if verbose and v in interest:
            print indent, 'v:', v, 'H to curr:', H_indices[H_start[v] : curr]
            print indent, 'v:', v, 'H to end:', H_indices[H_start[v] : H_end[v]]
            
    branches = incd[PXbuf[incd]]
    PXbuf[P] = True

    if verbose:
        print indent, 'incd:', incd.tolist()
#        print indent, 'incd[np.argsort(topo[incd])]', incd[np.argsort(topo[incd])].tolist()
#        print indent, 'branches:', branches.tolist()
    
    PXbuf[incd] = False
    PXbuf[X_incd] = False
    new_incd = np.empty(XE - PS - incd.size, PX.dtype)
    new_incd_end = 0
    
    ## Should we iterate on X_incd to update PX for old edges?

    max_degree = -1
    
    # Calculate new_incd := the neighbors of incd
    is_branch = np.zeros(PX.size, np.bool_)
    is_branch[branches] = True
    
    for v in incd:
        if is_branch[v]:
            v_degree_old = 0
            curr = G_start[v]
            # Move P and X to the bottom
            for w_i in range(G_start[v], G_end[v]):
                w = G_indices[w_i]
                w_pos = pos[w]
                if w_pos < oldPS or w_pos >= oldXE:
                    break
                elif PS <= w_pos and w_pos < XE:
                    v_degree_old += w_pos < sep
                    G_indices[curr], G_indices[w_i] = w, G_indices[curr]
                    curr += 1

                    if PXbuf[w]:
                        new_incd[new_incd_end] = w
                        new_incd_end += 1
                        PXbuf[w] = False
        else:
            v_degree_old, curr = move_PX(G_indices, G_start, G_end,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])

        v_degree, curr_new = move_PX(Gnew_indices, Gnew_start, Gnew_end,
                                     pos, oldPS, oldXE, PS, XE, sep, v_degree_old, v, Gnew_start[v])
        if v_degree > max_degree:
            max_degree = v_degree
            u, u_curr_new, u_curr = v, curr_new, curr
                    
    PXbuf[incd] = True
    PXbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    PXbuf[new_incd] = True
    
    # Only needed if we eventually call Gsep. If only returning clique
    # "branch roots" over new edges, then this is not needed
    for v in new_incd:
        v_degree_old, curr = move_PX(G_indices, G_start, G_end,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, G_start[v])
    
    #--------------------------------#

    Padj_u = G_indices[G_start[u] : u_curr]
    Padj_new_u = Gnew_indices[Gnew_start[u] : u_curr_new]    
    
    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
    
    # Always keep the between_new
    PXbuf[P[between_new[P]]] = True
    
    # Some vertices will not be reachable because we've already
    # removed a lot of nodes. Expand to find a vertex cover of incd.
#    for v in incd[~ PXbuf[incd]]:
    for v in branches[~ PXbuf[branches]]:
        if not PXbuf[v]:
            # By construction because of move_PX above, this will only
            # iterate over w's that are in incd
            for w in Gnew_indices[Gnew_start[v] : Gnew_end[v]]:
                w_pos = pos[w]
                PXbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
                
#    branches = incd[PXbuf[incd]]
    branches = branches[PXbuf[branches]]

#     # Alternative to setting P_between_new to True and finding vertex cover of incd. Empirically slower
#     tmp = new_incd[pos[new_incd] < sep]
#     branches = np.concatenate((incd[PXbuf[incd]], tmp[PXbuf[tmp]]))

    PXbuf[Padj_new_u] = True
    PXbuf[Padj_u] = True
    
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'Padj_new_u:', Padj_new_u
#     print indent, 'incd:', incd
#     print indent, 'new_incd:', new_incd
#     print indent, 'branches:', branches

    if verbose:
        print indent, 'branches:', branches.tolist()

    branches = branches[np.argsort(topo[branches])]
    
    for v in branches:
#         print indent, 'branching at:', v
                
        new_PS, new_XE = update_PX(G_indices, G_start, G_end, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        
        between_added = 0
        for w in Gnew_indices[Gnew_start[v] : Gnew_end[v]]:
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)                
                if not between_new[w]:
                    between_stack[between_end + between_added] = w
                    between_added += 1
                    between_new[w] = True
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < oldPS or w_pos >= oldXE:
                break

        for w in H_indices[H_start[v] : H_end[v]]:
            w_pos = pos[w]
            if (new_PS <= w_pos) and (w_pos < sep):                
                swap_pos(PX, pos, w, new_PS)
                new_PS += 1
            # elif (sep <= w_pos) and (w_pos < XE):
            #     swap_pos(PX, pos, w, new_XE)
            #     new_XE += 1
            elif (w_pos < oldPS) or (w_pos >= oldXE):
                break

        if verbose and (v in interest):
            print indent, 'Branching at v:', v, 'R:', R
            print indent, 'new P:', PX[new_PS:sep]
            print indent, 'H[v:end]:', H_indices[H_start[v] : H_end[v]]
            
        R_buff[R_end] = v
        if between_new[v]:
            cliques, cliques_indptr, cliques_n, tree_size = BK_hier_Gsep(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                G_start, G_end, G_indices,
                Gnew_start, Gnew_end, Gnew_indices,
                H_start, H_end, H_indices,
                topo,
                PXbuf, depth+1,
                cliques, cliques_indptr, cliques_n, tree_size)
        else:
            cliques, cliques_indptr, cliques_n, tree_size = BK_hier_Gnew(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                G_start, G_end, G_indices,
                Gnew_start, Gnew_end, Gnew_indices,
                H_start, H_end, H_indices,
                topo,
                PXbuf, depth+1,
                between_new, between_stack, between_end + between_added,
                cliques, cliques_indptr, cliques_n, tree_size)
            
        # Reset the between_new
        between_new[between_stack[between_end : between_end + between_added]] = False

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

#    if depth==0 or R_buff[0]==55:
#        print depth, 'Returning R:', R
        # print depth, 'branches:', branches.tolist()
        # print depth, 'incd:', incd.tolist()
        # if depth > 0:
        #     print depth, 'P:', P.tolist()
            
    return cliques, cliques_indptr, cliques_n, tree_size


def BK_hier_Gsep_wrapper(Gold, Gnew, H):
    k = Gold.shape[0]
    R = np.zeros(k, np.int32)
    R_end = np.int32(0)
    PX = np.arange(k).astype(np.int32)
    sep = PX.size
    pos = np.empty(PX.size, np.int32)
    pos[PX] = np.arange(PX.size)
    PXbuf = np.zeros(PX.size, np.bool_)
    PXbuf2 = np.ones(PX.size, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])

    cliques, cliques_indptr, cliques_n = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0    

    g = igraph.Graph(n=H.shape[0], edges=zip(*H.nonzero()), directed=True)
    topo = np.array(g.topological_sorting(mode='out'))
    
    tree_size = np.asfortranarray(np.zeros(1000, np.int32))
    tree_size[:2] = np.array([0, 0], np.int32)
    
    cliques, cliques_indptr, cliques_n, tree_size = BK_hier_Gsep(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        Gold.indptr[:-1], Gold.indptr[1:], Gold.indices,
        Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices,
        H.indptr[:-1], H.indptr[1:], H.indices,
        topo,
        PXbuf2, 0, cliques, cliques_indptr, cliques_n, tree_size)
    cliques, cliques_indptr, cliques_n = trim_cliques(cliques, cliques_indptr, cliques_n)
    
    return cliques, cliques_indptr, cliques_n, tree_size

def BK_hier_Gnew_wrapper(Gold, Gnew, H):
    k = Gold.shape[0]
    R = np.zeros(k, np.int32)
    R_end = np.int32(0)
    PX = np.arange(k).astype(np.int32)
    sep = PX.size
    pos = np.empty(PX.size, np.int32)
    pos[PX] = np.arange(PX.size)
    PXbuf = np.zeros(PX.size, np.bool_)
    PXbuf2 = np.ones(PX.size, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])

    between_stack = np.arange(PX.max() + 1).astype(np.int32)
    between_new = np.zeros(PX.max() + 1, np.bool_)        
    between_end = 0

    cliques, cliques_indptr, cliques_n = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    cliques_indptr[:2] = 0    

    g = igraph.Graph(n=H.shape[0], edges=zip(*H.nonzero()), directed=True)
    topo = np.array(g.topological_sorting(mode='out'))

    print 'interest:', interest
    print 'topo:', zip(interest, topo[interest])
    print Gold[interest,:][:,interest].toarray().astype(np.int32)
    print Gnew[interest,:][:,interest].toarray().astype(np.int32)
    # 0 / asdf
        
    tree_size = np.asfortranarray(np.zeros(1000, np.int32))
    tree_size[:2] = np.array([0, 0], np.int32)
    
    cliques, cliques_indptr, cliques_n, tree_size = BK_hier_Gnew(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        Gold.indptr[:-1], Gold.indptr[1:], Gold.indices,
        Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices,
        H.indptr[:-1], H.indptr[1:], H.indices,
        topo,
        PXbuf2, 0,
        between_new, between_stack, between_end,
        cliques, cliques_indptr, cliques_n, tree_size)
    cliques, cliques_indptr, cliques_n = trim_cliques(cliques, cliques_indptr, cliques_n)
    
    return cliques, cliques_indptr, cliques_n, tree_size

def get_cliques_igraph(n, G, Gnew=None, input_fmt='edgelist'):
    if input_fmt=='edgelist':
        g = igraph.Graph(n, edges=G, directed=False)
        
    elif input_fmt=='matrix':
        i, j = G.nonzero()
        tmp = i < j
        i, j = i[tmp], j[tmp]
        
        g = igraph.Graph(n, edges=zip(i,j), directed=False)

    else:
        raise Exception('Invalid input format')
        
    clique_list = g.maximal_cliques()

    if Gnew is not None:
        if input_fmt=='edgelist':
            i, j = zip(*Gnew)
            Gnew = scipy.sparse.coo_matrix((np.ones(len(i), dtype=np.bool), (i, j)), shape=(n, n))

        # Check that the diagonal is all zeros
        tmp = np.arange(G.shape[0])
        assert G[tmp, tmp].nonzero()[0].size == 0

        # Filter for cliques that have an edge among the new edges
        clique_list = [c for c in clique_list if Gnew[c,:][:,c].sum() > 0]

    return [tuple(sorted(c)) for c in clique_list]

