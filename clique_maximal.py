import time
import igraph
import numpy as np
from clique_atomic import *
from clixov_utils import trim_cliques
from constants import cache

debug = False

#interest = [1640, 2670, 4782, 6681, 6918]
#interest = [1640, 2670, 6681, 6918]
#interest = [326, 34, 247, 328]
interest = []

@jit(nopython=True, cache=cache)
def BK(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                   GS, GE, GI, PXbuf, depth,
                   C, CP, CN):
    R = R_buff[:R_end]    
    P, X = PX[PS:sep], PX[sep:XE]
    
#     indent = '\t' * depth
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X    
    
    if P.size==0:
        if X.size==0:
            C, CP, CN = update_cliques2(C, CP, CN, R)
        return C, CP, CN

#     u = -1
#     max_degree = -1
#     within_P_degree = 0
#     for v in P:
#         v_degree, curr = move_PX(GI, GS, GE,
#                                  pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])                
#         if v_degree > max_degree:
#             max_degree, u, u_curr = v_degree, v, curr
#         within_P_degree += v_degree
#     max_X_to_P_degree = 0
#     for v in X:
#         v_degree, curr = move_PX(GI, GS, GE,
#                                  pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])                
#         if v_degree > max_degree:
#             max_degree, u, u_curr = v_degree, v, curr
#         max_X_to_P_degree = max(max_X_to_P_degree, v_degree)
#     if within_P_degree==P.size * (P.size -1):
#         if max_X_to_P_degree < P.size:
#             R_buff[R_end : R_end + P.size] = P
#             C, CP, CN = update_cliques2(C, CP, CN, R_buff[:R_end + P.size])
#         return C, CP, CN

    u = -1
    max_degree = -1
    for v in PX[PS:XE]:
        v_degree, curr = move_PX(GI, GS, GE,
                                 pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])                
        if v_degree > max_degree:
            max_degree, u, u_curr = v_degree, v, curr

    Padj_u = pos[GI[GS[u] : u_curr]]
    PXbuf[Padj_u] = False
    branches = P[PXbuf[PS:sep]]
    PXbuf[Padj_u] = True
    
    for v in branches:
        new_PS, new_XE = update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)            
        R_buff[R_end] = v
            
        C, CP, CN = BK(
            R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            GS, GE, GI, PXbuf, depth+1,
            C, CP, CN)
        
        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    # Move back all branches into P
    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1
    
    return C, CP, CN        

def BK_py(G, PX=None):
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
    C, CP, CN = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    CP[:2] = 0
    
    C, CP, CN = BK(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        PXbuf2, 0, C, CP, CN)

    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN

@jit(nopython=True, cache=cache)
def BK_Gsep(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                        GS, GE, GI, GS_new, GE_new, GI_new,
                        PXbuf, depth, C, CP, CN, tree):
    if tree.size <= 4 + tree[0]:
        tree = np.concatenate((tree, np.empty(tree.size, tree.dtype)))
    tree[2+tree[0]] = depth
    tree[0] += 1
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]
    
#     indent = '\t' * depth
#     print indent, '---------------'
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX`
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X
    
    if P.size==0:
        tree[1] += 1
        if X.size==0:
            C, CP, CN = update_cliques2(C, CP, CN, R)
        return C, CP, CN, tree

    u = -1
    max_degree = -1
    for v in PX[PS:XE][::-1]:
        v_degree, curr_new = move_PX(GI_new, GS_new, GE_new,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, GS_new[v])        
        v_degree, curr = move_PX(GI, GS, GE,
                                     pos, oldPS, oldXE, PS, XE, sep, v_degree, v, GS[v])   
        if v_degree > max_degree: 
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new

#     # Swap the pivot to the beginning of P so that it is the first branch
#     if pos[u] < sep:
#         swap_pos(PX, pos, u, PS)
        
    Padj_u = pos[GI[GS[u] : u_curr]]
    Padj_new_u = pos[GI_new[GS_new[u] : u_curr_new]]
    
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
        
        new_PS, new_XE = update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        new_PS, new_XE = update_PX(GI_new, GS_new, GE_new, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)

        R_buff[R_end] = v

        C, CP, CN, tree = BK_Gsep(
            R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            GS, GE, GI,
            GS_new, GE_new, GI_new,
            PXbuf, depth+1,
            C, CP, CN, tree)        

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return C, CP, CN, tree

def BK_Gsep_py(G, dG):
    k = G.shape[0]
    R = np.zeros(k, np.int32)
    R_end = np.int32(0)
    PX = np.arange(k).astype(np.int32)
    sep = PX.size
    pos = np.empty(PX.size, np.int32)
    pos[PX] = np.arange(PX.size)
    PXbuf = np.zeros(PX.size, np.bool_)
    PXbuf2 = np.ones(PX.size, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])
    
    C, CP, CN = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    CP[:2] = 0    
    
    C, CP, CN, tree = BK_Gsep(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        PXbuf2, 0, C, CP, CN)
    
    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN


@jit(nopython=True, cache=cache)
def BK_dG(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                        GS, GE, GI, GS_new, GE_new, GI_new,
                        PXbuf, depth,
                        btw_new, btw_stack, btw_end,
                        C, CP, CN, tree):
    """
    btw_new[v]: Is there a new edge crossing btw R and node v in P?
    """
    if tree.size <= 4 + tree[0]:
        tree = np.concatenate((tree, np.empty(tree.size, tree.dtype)))
    tree[2+tree[0]] = depth
    tree[0] += 1
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]

#     indent = '\t' * depth
#     print indent, '---------------'
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X
#     print indent, 'btw_new:', btw_new.astype(np.int32)

    if P.size==0:
        tree[1] += 1
        return C, CP, CN, tree
    
    incd = np.empty(sep - PS, PX.dtype)
    incd_count = 0
    max_degree = -1
    X_incd = np.empty(XE - sep, PX.dtype)
    X_incd_count = 0
    
    # Iterate over new edges from X too
    for v in X:
        v_degree_new, curr_new = move_PX(GI_new, GS_new, GE_new,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, GS_new[v])        
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new = v, curr_new
        if v_degree_new > 0:
            tmp, curr = move_PX(GI, GS, GE,
                                pos, oldPS, oldXE, PS, XE, sep, v_degree_new, v, GS[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
    for v in P:
        v_degree_new, curr_new = move_PX(GI_new, GS_new, GE_new,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, GS_new[v])        
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new, u_incd = v, curr_new, incd_count
        if v_degree_new > 0 or btw_new[v]:
            incd[incd_count] = v
            incd_count += 1
            
    u_curr = GE[u]

    if incd_count == 0:
        tree[1] += 1
        return C, CP, CN, tree
    incd = incd[:incd_count]
    X_incd = X_incd[:X_incd_count]
    
    PXbuf[incd] = False
    PXbuf[X_incd] = False
    new_incd = np.empty(XE - PS - incd.size, PX.dtype)
    new_incd_end = 0
    
    ## Should we iterate on X_incd to update PX for old edges?
    
    for v in incd:
        v_degree_old = 0
        curr = GS[v]
        
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
                    
    PXbuf[incd] = True
    PXbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    PXbuf[new_incd] = True
    
    # Only needed if we eventually call Gsep. If only returning clique "branch roots" over new edges, then this is not needed
    for v in new_incd:
        v_degree_old, curr = move_PX(GI, GS, GE,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])
    
    #--------------------------------#

    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = GI_new[GS_new[u] : u_curr_new]    
    
    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
    
    # Always keep the btw_new
    PXbuf[P[btw_new[P]]] = True
    
    # Some vertices will not be reachable because we've already
    # removed a lot of nodes. Expand to find a vertex cover of incd.
    for v in incd[~ PXbuf[incd]]:
        if not PXbuf[v]:
            # By construction because of move_PX above, this will only
            # iterate over w's that are in incd
            for w in GI_new[GS_new[v] : GE_new[v]]:
                w_pos = pos[w]
                PXbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
                
    branches = incd[PXbuf[incd]]

#     # Alternative to setting P_btw_new to True and finding vertex cover of incd. Empirically slower
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
                
        new_PS, new_XE = update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        
        btw_added = 0
        for w in GI_new[GS_new[v] : GE_new[v]]:
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)                
                if not btw_new[w]:
                    btw_stack[btw_end + btw_added] = w
                    btw_added += 1
                    btw_new[w] = True
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < oldPS or w_pos >= oldXE:
                break

        R_buff[R_end] = v
        if btw_new[v]:
            C, CP, CN, tree = BK_Gsep(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                PXbuf, depth+1,
                C, CP, CN, tree)
        else:
            C, CP, CN, tree = BK_dG(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                PXbuf, depth+1,
                btw_new, btw_stack, btw_end + btw_added,
                C, CP, CN, tree)
            
        # Reset the btw_new
        btw_new[btw_stack[btw_end : btw_end + btw_added]] = False

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return C, CP, CN, tree

def BK_dG_py(G, dG, PX=None, btw_new=None):
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
        initialize_PX(dG.indices, dG.indptr[:-1], dG.indptr[1:], pos, PX)
        
    R = np.zeros(PX.size, np.int32)
    R_end = np.int32(0)    
    sep = PX.size
    PXbuf = np.zeros(k, np.bool_)
    PXbuf2 = np.ones(k, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])
    
    btw_stack = np.arange(PX.max() + 1).astype(np.int32)
    if btw_new is None:
        btw_new = np.zeros(PX.max() + 1, np.bool_)        
        btw_end = 0
    else:
        btw_end = btw_new.sum()
        btw_stack[:btw_end] = btw_new.nonzero()[0]

    C, CP, CN = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    CP[:2] = 0
    
    tree = np.asfortranarray(np.zeros(1000, np.int32))
    tree[:2] = np.array([0, 0], np.int32)
    
    C, CP, CN, tree = BK_dG(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        PXbuf2, 0, btw_new, btw_stack, btw_end,
        C, CP, CN, tree)

    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN, tree

@jit(nopython=True, cache=cache)
def BK_hier_Gsep(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
                 GS, GE, GI, GS_new, GE_new, GI_new,
                 HS, HE, HI,
                 topo,
                 PXbuf, depth, C, CP, CN,
                 stats, tree):
    if tree.size == stats[0] + 2:
        tree = expand_1d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]

#    verbose = (depth > 0 and all(x in interest for x in R_buff[:R_end]))
#    verbose = (depth > 0 and R_buff[0] in interest)
#    verbose = True
#    verbose = False
    verbose = debug
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]
    
    # indent = '\t' * depth
    # if verbose:
    #     print indent, '----Gsep-----------'
    #     print indent, 'DEPTH:', depth
    #     print indent, 'PX:', PX
    #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    #     print indent, 'R:', R, 'P:', P, 'X:', X
    
    if P.size==0:
        # if verbose:
        #     print indent, depth, 'Returning', 'R:', R, 'P:', P, 'X:', X
            
        if X.size==0:
            C, CP, CN = update_cliques2(C, CP, CN, R)
        return C, CP, CN, tree
        
    # Filter branches. Follow the topological sorting, going top-down
    # the hierarchy
    tmp = P[np.argsort(topo[P])]
    for v in tmp:
        curr = HS[v]
        for w_i in range(HS[v], HE[v]):
            w = HI[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            #elif PS <= w_pos and w_pos < XE:
            elif PS <= w_pos and w_pos < sep:
                HI[curr], HI[w_i] = w, HI[curr]
                curr += 1

                # Remove descendants of v from being branches
                if PXbuf[v]:
                    PXbuf[w] = False
        
        # if verbose:
        #     print indent, 'v:', v, 'H to curr:', HI[HS[v] : curr]
        #     print indent, 'v:', v, 'H to end:', HI[HS[v] : HE[v]]
                    
    branches = P[PXbuf[P]]

    # if verbose:
    #     print indent, 'branches:', branches
        
    u = -1
    max_degree = -1
    for v in PX[PS:XE][::-1]:
        v_degree, curr_new = move_PX(GI_new, GS_new, GE_new,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, GS_new[v])        
        v_degree, curr = move_PX(GI, GS, GE,
                                 pos, oldPS, oldXE, PS, XE, sep, v_degree, v, GS[v])
        if PXbuf[v] and (v_degree > max_degree): 
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new

    # TODO: make this more efficient by not having to make it |P| time
    PXbuf[P] = True
    
    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = GI_new[GS_new[u] : u_curr_new]

    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
#    branches = P[PXbuf[PS:sep]]
    branches = branches[PXbuf[branches]]
    PXbuf[Padj_u] = True
    PXbuf[Padj_new_u] = True

    branches = branches[np.argsort(topo[branches])]
        
    # if verbose:
    #     print indent, 'pivot u:', u
    #     print indent, 'Padj_u:', Padj_u
    #     print indent, 'Padj_new_u:', Padj_new_u
    #     print indent, 'branches:', branches

    for v in branches:
#         print indent, 'branching at v:', v

        new_PS, new_XE = update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        new_PS, new_XE = update_PX(GI_new, GS_new, GE_new, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)

        for w in HI[HS[v] : HE[v]]:
            w_pos = pos[w]
            if (new_PS <= w_pos) and (w_pos < sep):                
                swap_pos(PX, pos, w, new_PS)
                new_PS += 1
            elif (w_pos < oldPS) or (w_pos >= oldXE):
                break

        R_buff[R_end] = v
        tree[stats[0]+1] = curr_node
        C, CP, CN, tree = BK_hier_Gsep(
            R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            GS, GE, GI,
            GS_new, GE_new, GI_new,
            HS, HE, HI,
            topo,
            PXbuf, depth+1,
            C, CP, CN, stats, tree)
            
        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return C, CP, CN, tree

@jit(nopython=True, cache=cache)
def BK_hier_dG(R_buff, R_end, PX, PS, sep, XE, oldPS, oldXE, pos,
               GS, GE, GI, GS_new, GE_new, GI_new,                 
               HS, HE, HI,
               topo,
               PXbuf, depth,
               btw_new, btw_stack, btw_end,
               C, CP, CN,
               stats, tree):
    """
    btw_new[v]: Is there a new edge crossing btw R and node v in P?
    """
    if tree.size == stats[0] + 2:
        tree = expand_1d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]
    
    R = R_buff[:R_end]
    P, X = PX[PS:sep], PX[sep:XE]

#    verbose = depth==0 or (depth > 0 and all(x in interest for x in R_buff[:R_end]))
#    verbose = depth==0 or (depth > 0 and R_buff[0] in interest)
#    verbose = True
#    verbose = False
    verbose = debug

    # indent = '\t' * depth
    # if verbose:
    #     print indent, '----Gnew-----------'
    #     print indent, 'DEPTH:', depth
    #     print indent, 'PX:', PX
    #     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
    #     print indent, 'R:', R, 'P:', P, 'X:', X
    #     print indent, 'btw_new:', btw_new.nonzero()[0]

    if P.size==0:
        # if verbose and (R_buff[0] in interest):
        #     print indent, 'Returning', 'R:', R, 'P:', P, 'X:', X

        return C, CP, CN, tree
    
    incd = np.empty(sep - PS, PX.dtype)
    incd_count = 0
    X_incd = np.empty(XE - sep, PX.dtype)
    X_incd_count = 0
    
    # Iterate over new edges from X too
    for v in X:
        v_degree_new, curr_new = move_PX(GI_new, GS_new, GE_new,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, GS_new[v])        
        if v_degree_new > 0:
            tmp, curr = move_PX(GI, GS, GE,
                                pos, oldPS, oldXE, PS, XE, sep, v_degree_new, v, GS[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
    for v in P:
        v_degree_new, curr_new = move_PX(GI_new, GS_new, GE_new,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, GS_new[v])        
        if v_degree_new > 0 or btw_new[v]:
            incd[incd_count] = v
            incd_count += 1

    if incd_count == 0:
        # if verbose:
        #     print indent, 'Returning because incd.size==0', 'R:', R, 'P:', P, 'X:', X
            
        return C, CP, CN, tree
    incd = incd[:incd_count]
    X_incd = X_incd[:X_incd_count]

    is_incd = np.zeros(PX.size, np.bool_)
    is_incd[incd] = True

    # Filter branches. Follow the topological sorting, going top-down
    # the hierarchy
    #for v in incd[np.argsort(topo[incd])]:
    for v in P[np.argsort(topo[P])]:
        curr = HS[v]
        for w_i in range(HS[v], HE[v]):
            w = HI[w_i]
            w_pos = pos[w]
            if w_pos < oldPS or w_pos >= oldXE:
                break
            #elif PS <= w_pos and w_pos < XE:
            elif PS <= w_pos and w_pos < sep:
                HI[curr], HI[w_i] = w, HI[curr]
                curr += 1

                # Remove descendants of v from being branches
                if is_incd[v] and PXbuf[v]:
                   PXbuf[w] = False
        
        # if verbose and curr > HS[v]:
        #     print indent, 'v:', v, 'H to curr:', HI[HS[v] : curr]
        #     print indent, 'v:', v, 'H to end:', HI[HS[v] : HE[v]]
            
    branches = incd[PXbuf[incd]]
    PXbuf[P] = True

    # if verbose:
    #     print indent, 'incd:', incd.tolist()
    #     print indent, 'branches:', branches.tolist()
    
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
            curr = GS[v]
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
        else:
            v_degree_old, curr = move_PX(GI, GS, GE,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])

        v_degree, curr_new = move_PX(GI_new, GS_new, GE_new,
                                     pos, oldPS, oldXE, PS, XE, sep, v_degree_old, v, GS_new[v])
        
#        if v_degree > max_degree:
        if is_branch[v] and v_degree > max_degree:
            max_degree = v_degree
            u, u_curr_new, u_curr = v, curr_new, curr
                    
    PXbuf[incd] = True
    PXbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    PXbuf[new_incd] = True
    
    # Only needed if we eventually call Gsep. If only returning clique
    # "branch roots" over new edges, then this is not needed
    for v in new_incd:
        v_degree_old, curr = move_PX(GI, GS, GE,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])
    
    #--------------------------------#

    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = GI_new[GS_new[u] : u_curr_new]    
    
    PXbuf[Padj_u] = False
    PXbuf[Padj_new_u] = False
    
    # Always keep the btw_new
    PXbuf[P[btw_new[P]]] = True
        
    # Some vertices in incd will not be reachable because we've already
    # removed a lot of nodes. Expand to find a vertex cover of the new edges.
#    for v in incd[~ PXbuf[incd]]:
    for v in branches[~ PXbuf[branches]]:
        if not PXbuf[v]:
            # By construction because of move_PX above, this will only
            # iterate over w's that are in incd
            for w in GI_new[GS_new[v] : GE_new[v]]:
                # if verbose:
                #     print indent, 'v:', v, 'w:', w
                w_pos = pos[w]
                PXbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
                
#    branches = incd[PXbuf[incd]]
    branches = branches[PXbuf[branches]]

#     # Alternative to setting P_btw_new to True and finding vertex cover of incd. Empirically slower
#     tmp = new_incd[pos[new_incd] < sep]
#     branches = np.concatenate((incd[PXbuf[incd]], tmp[PXbuf[tmp]]))

    PXbuf[Padj_new_u] = True
    PXbuf[Padj_u] = True

    branches = branches[np.argsort(topo[branches])]
    
    # if verbose:
    #     print indent, 'pivot u:', u
    #     print indent, 'Padj_u:', Padj_u
    #     print indent, 'Padj_new_u:', Padj_new_u
    #     print indent, 'incd:', incd
    #     print indent, 'new_incd:', new_incd
    #     print indent, 'branches:', branches

    for v in branches:
#        print indent, 'branching at:', v
                
        new_PS, new_XE = update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        
        btw_added = 0
        for w in GI_new[GS_new[v] : GE_new[v]]:
            w_pos = pos[w]
            if (PS <= w_pos) and (w_pos < sep):
                new_PS -= 1
                swap_pos(PX, pos, w, new_PS)                
                if not btw_new[w]:
                    btw_stack[btw_end + btw_added] = w
                    btw_added += 1
                    btw_new[w] = True
            elif (sep <= w_pos) and (w_pos < XE):
                swap_pos(PX, pos, w, new_XE)
                new_XE += 1
            elif w_pos < oldPS or w_pos >= oldXE:
                break

        for w in HI[HS[v] : HE[v]]:
            w_pos = pos[w]
            if (new_PS <= w_pos) and (w_pos < sep):                
                swap_pos(PX, pos, w, new_PS)
                new_PS += 1
            elif (w_pos < oldPS) or (w_pos >= oldXE):
                break
            
        R_buff[R_end] = v
        tree[stats[0]+1] = curr_node
        if btw_new[v]:
            C, CP, CN, tree = BK_hier_Gsep(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                HS, HE, HI,
                topo,
                PXbuf, depth+1,
                C, CP, CN, stats, tree)
        else:
            C, CP, CN, tree = BK_hier_dG(
                R_buff, R_end + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                HS, HE, HI,
                topo,
                PXbuf, depth+1,
                btw_new, btw_stack, btw_end + btw_added,
                C, CP, CN, stats, tree)
            
        # Reset the btw_new
        btw_new[btw_stack[btw_end : btw_end + btw_added]] = False

        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return C, CP, CN, tree

def BK_hier_Gsep_py(G, dG, H, verbose=False):
    k = G.shape[0]
    R = np.zeros(k, np.int32)
    R_end = np.int32(0)
    PX = np.arange(k).astype(np.int32)
    sep = PX.size
    pos = np.empty(PX.size, np.int32)
    pos[PX] = np.arange(PX.size)
    PXbuf = np.zeros(PX.size, np.bool_)
    PXbuf2 = np.ones(PX.size, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])

    C, CP, CN = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    CP[:2] = 0    

    g = igraph.Graph(n=H.shape[0], edges=zip(*H.nonzero()), directed=True)
    topo = np.array(g.topological_sorting(mode='out'))
    
    tree = np.asfortranarray(np.zeros(1000, np.int32))
    tree[:2] = np.array([0, 0], np.int32)

    start = time.time()
    C, CP, CN, tree = BK_hier_Gsep(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        H.indptr[:-1], H.indptr[1:], H.indices,
        topo,
        PXbuf2, 0, C, CP, CN, tree)
    if verbose: print 'BK_hier_Gsep time:', time.time() - start
    C, CP, CN = trim_cliques(C, CP, CN)
    
    return C, CP, CN, tree

def BK_hier_dG_py(G, dG, H, verbose=False):
    k = G.shape[0]
    R = np.zeros(k, np.int32)
    R_end = np.int32(0)
    PX = np.arange(k).astype(np.int32)
    sep = PX.size
    pos = np.empty(PX.size, np.int32)
    pos[PX] = np.arange(PX.size)
    PXbuf = np.zeros(PX.size, np.bool_)
    PXbuf2 = np.ones(PX.size, np.bool_)
    PS, sep, XE = np.int32([0, sep, PX.size])

    btw_stack = np.arange(PX.max() + 1).astype(np.int32)
    btw_new = np.zeros(PX.max() + 1, np.bool_)        
    btw_end = 0

    C, CP, CN = np.empty(PX.size, R.dtype), np.empty(PX.size + 1, np.int32), 0
    CP[:2] = 0    

    g = igraph.Graph(n=H.shape[0], edges=zip(*H.nonzero()), directed=True)
    topo = np.array(g.topological_sorting(mode='out'))

    stats = np.array([0], np.int32)
    tree = np.asfortranarray(np.zeros(1000, np.int32))

    start = time.time()
    C, CP, CN, tree = BK_hier_dG(
        R, R_end, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        H.indptr[:-1], H.indptr[1:], H.indices,
        topo,
        PXbuf2, 0,
        btw_new, btw_stack, btw_end,
        C, CP, CN,
        stats, tree)
    if verbose: print 'BK_hier time:', time.time() - start
    C, CP, CN = trim_cliques(C, CP, CN)
    tree = tree[:stats[0]]
    
    return C, CP, CN, tree

def get_cliques_igraph(n, G, dG=None, input_fmt='edgelist'):
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

    if dG is not None:
        if input_fmt=='edgelist':
            i, j = zip(*dG)
            dG = scipy.sparse.coo_matrix((np.ones(len(i), dtype=np.bool), (i, j)), shape=(n, n))

        # Check that the diagonal is all zeros
        tmp = np.arange(G.shape[0])
        assert G[tmp, tmp].nonzero()[0].size == 0

        # Filter for cliques that have an edge among the new edges
        clique_list = [c for c in clique_list if dG[c,:][:,c].sum() > 0]

    return [tuple(sorted(c)) for c in clique_list]

