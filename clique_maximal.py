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
def BK(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
       GS, GE, GI, Tbuf, depth,
       C, CP, CN):
    R = Rbuf[:RE]    
    P, X = PX[PS:sep], PX[sep:XE]
    
#     indent = '\t' * depth
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X    
    
    if P.size==0:
        if X.size==0:
            C, CP, CN = update_cliques(C, CP, CN, R)
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
#             Rbuf[RE : RE + P.size] = P
#             C, CP, CN = update_cliques(C, CP, CN, Rbuf[:RE + P.size])
#         return C, CP, CN

    u = -1
    max_degree = -1
    for v in PX[PS:XE]:
        v_degree, curr = move_PX(GI, GS, GE,
                                 pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])                
        if v_degree > max_degree:
            max_degree, u, u_curr = v_degree, v, curr

    Padj_u = pos[GI[GS[u] : u_curr]]
    Tbuf[Padj_u] = False
    branches = P[Tbuf[PS:sep]]
    Tbuf[Padj_u] = True
    
    for v in branches:
        new_PS, new_XE = update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)            
        Rbuf[RE] = v
            
        C, CP, CN = BK(
            Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            GS, GE, GI, Tbuf, depth+1,
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
    k = G.shape[0]
    custom_PX = PX is not None
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k, PX=PX)
    if custom_PX:
        initialize_PX(G.indices, G.indptr[:-1], G.indptr[1:], pos, PX)
    
    C, CP, CN = BK(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        Tbuf, 0, C, CP, CN)

    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN

@jit(nopython=True, cache=cache)
def BK_Gsep(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
            GS, GE, GI, GS_new, GE_new, GI_new,
            Tbuf, depth, C, CP, CN, tree):
    if tree.size <= 4 + tree[0]:
        tree = np.concatenate((tree, np.empty(tree.size, tree.dtype)))
    tree[2+tree[0]] = depth
    tree[0] += 1
    
    R = Rbuf[:RE]
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
            C, CP, CN = update_cliques(C, CP, CN, R)
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
    
    Tbuf[Padj_u] = False
    Tbuf[Padj_new_u] = False
    branches = P[Tbuf[PS:sep]]
    Tbuf[Padj_u] = True
    Tbuf[Padj_new_u] = True
    
#     print indent, 'pivot u:', u
#     print indent, 'Padj_u:', Padj_u
#     print indent, 'Padj_new_u:', Padj_new_u
#     print indent, 'branches:', branches

    for v in branches:
#         print indent, 'branching at v:', v
        
        new_PS, new_XE = update_PX(GI, GS, GE, oldPS, oldXE, PS, XE, sep, PX, pos, v, sep, sep)
        new_PS, new_XE = update_PX(GI_new, GS_new, GE_new, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)

        Rbuf[RE] = v

        C, CP, CN, tree = BK_Gsep(
            Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            GS, GE, GI,
            GS_new, GE_new, GI_new,
            Tbuf, depth+1,
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
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k)
    
    C, CP, CN, tree = BK_Gsep(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        Tbuf, 0, C, CP, CN)
    
    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN


@jit(nopython=True, cache=cache)
def BK_dG(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
          GS, GE, GI, GS_new, GE_new, GI_new,
          Tbuf, depth,
          btw_new, btw_stack, btw_end,
          C, CP, CN, tree):
    """
    btw_new[v]: Is there a new edge crossing btw R and node v in P?
    """
    if tree.size <= 4 + tree[0]:
        tree = np.concatenate((tree, np.empty(tree.size, tree.dtype)))
    tree[2+tree[0]] = depth
    tree[0] += 1
    
    R = Rbuf[:RE]
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
    
    Tbuf[incd] = False
    Tbuf[X_incd] = False
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
                
                if Tbuf[w]:
                    new_incd[new_incd_end] = w
                    new_incd_end += 1
                    Tbuf[w] = False
                    
    Tbuf[incd] = True
    Tbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    Tbuf[new_incd] = True
    
    # Only needed if we eventually call Gsep. If only returning clique "branch roots" over new edges, then this is not needed
    for v in new_incd:
        v_degree_old, curr = move_PX(GI, GS, GE,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])
    
    #--------------------------------#

    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = GI_new[GS_new[u] : u_curr_new]    
    
    Tbuf[Padj_u] = False
    Tbuf[Padj_new_u] = False
    
    # Always keep the btw_new
    Tbuf[P[btw_new[P]]] = True
    
    # Some vertices will not be reachable because we've already
    # removed a lot of nodes. Expand to find a vertex cover of incd.
    for v in incd[~ Tbuf[incd]]:
        if not Tbuf[v]:
            # By construction because of move_PX above, this will only
            # iterate over w's that are in incd
            for w in GI_new[GS_new[v] : GE_new[v]]:
                w_pos = pos[w]
                Tbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
                
    branches = incd[Tbuf[incd]]

#     # Alternative to setting P_btw_new to True and finding vertex cover of incd. Empirically slower
#     tmp = new_incd[pos[new_incd] < sep]
#     branches = np.concatenate((incd[Tbuf[incd]], tmp[Tbuf[tmp]]))

    Tbuf[Padj_new_u] = True
    Tbuf[Padj_u] = True
    
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

        Rbuf[RE] = v
        if btw_new[v]:
            C, CP, CN, tree = BK_Gsep(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                Tbuf, depth+1,
                C, CP, CN, tree)
        else:
            C, CP, CN, tree = BK_dG(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                Tbuf, depth+1,
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

def BK_dG_py(G, dG, PX=None):
    k = G.shape[0]
    custom_PX = PX is not None
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k, PX=PX)
    if custom_PX:
        initialize_PX(G.indices, G.indptr[:-1], G.indptr[1:], pos, PX)
        initialize_PX(dG.indices, dG.indptr[:-1], dG.indptr[1:], pos, PX)
    
    C, CP, CN, tree = BK_dG(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        Tbuf, 0, btw_new, btw_stack, btw_end,
        C, CP, CN, tree)

    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN, tree

@jit(nopython=True, cache=cache)
def BK_hier_Gsep(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
                 GS, GE, GI, GS_new, GE_new, GI_new,
                 HS, HE, HI,
                 topo,
                 Tbuf, depth, C, CP, CN,
                 stats, tree):
    if tree.size == stats[0] + 2:
        tree = expand_1d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]

#    verbose = (depth > 0 and all(x in interest for x in Rbuf[:RE]))
#    verbose = (depth > 0 and Rbuf[0] in interest)
#    verbose = True
#    verbose = False
    verbose = debug
    
    R = Rbuf[:RE]
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
            C, CP, CN = update_cliques(C, CP, CN, R)
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
                if Tbuf[v]:
                    Tbuf[w] = False
        
        # if verbose:
        #     print indent, 'v:', v, 'H to curr:', HI[HS[v] : curr]
        #     print indent, 'v:', v, 'H to end:', HI[HS[v] : HE[v]]

    ## TODO: 
    ## Need to make sure we have an edge cover
    ## Iterate across all edges between removed nodes and see if any edge has been left out?
    
    branches = P[Tbuf[P]]

    # if verbose:
    #     print indent, 'branches:', branches
        
    u = -1
    max_degree = -1
    for v in PX[PS:XE][::-1]:
        v_degree, curr_new = move_PX(GI_new, GS_new, GE_new,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, GS_new[v])        
        v_degree, curr = move_PX(GI, GS, GE,
                                 pos, oldPS, oldXE, PS, XE, sep, v_degree, v, GS[v])
        if Tbuf[v] and (v_degree > max_degree): 
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new

    # TODO: make this more efficient by not having to make it |P| time
    Tbuf[P] = True
    
    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = GI_new[GS_new[u] : u_curr_new]

    Tbuf[Padj_u] = False
    Tbuf[Padj_new_u] = False
#    branches = P[Tbuf[PS:sep]]
    branches = branches[Tbuf[branches]]
    Tbuf[Padj_u] = True
    Tbuf[Padj_new_u] = True

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

        Rbuf[RE] = v
        tree[stats[0]+1] = curr_node
        C, CP, CN, tree = BK_hier_Gsep(
            Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            GS, GE, GI,
            GS_new, GE_new, GI_new,
            HS, HE, HI,
            topo,
            Tbuf, depth+1,
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
def BK_hier_dG(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
               GS, GE, GI, GS_new, GE_new, GI_new,                 
               HS, HE, HI,
               topo,
               Tbuf, depth,
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
    
    R = Rbuf[:RE]
    P, X = PX[PS:sep], PX[sep:XE]

#    verbose = depth==0 or (depth > 0 and all(x in interest for x in Rbuf[:RE]))
#    verbose = depth==0 or (depth > 0 and Rbuf[0] in interest)
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
        # if verbose and (Rbuf[0] in interest):
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
                if is_incd[v] and Tbuf[v]:
                   Tbuf[w] = False
        
        # if verbose and curr > HS[v]:
        #     print indent, 'v:', v, 'H to curr:', HI[HS[v] : curr]
        #     print indent, 'v:', v, 'H to end:', HI[HS[v] : HE[v]]
            
    branches = incd[Tbuf[incd]]
    Tbuf[P] = True

    # if verbose:
    #     print indent, 'incd:', incd.tolist()
    #     print indent, 'branches:', branches.tolist()
    
    Tbuf[incd] = False
    Tbuf[X_incd] = False
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

                    if Tbuf[w]:
                        new_incd[new_incd_end] = w
                        new_incd_end += 1
                        Tbuf[w] = False
        else:
            v_degree_old, curr = move_PX(GI, GS, GE,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])

        v_degree, curr_new = move_PX(GI_new, GS_new, GE_new,
                                     pos, oldPS, oldXE, PS, XE, sep, v_degree_old, v, GS_new[v])
        
#        if v_degree > max_degree:
        if is_branch[v] and v_degree > max_degree:
            max_degree = v_degree
            u, u_curr_new, u_curr = v, curr_new, curr
                    
    Tbuf[incd] = True
    Tbuf[X_incd] = True
    new_incd = new_incd[:new_incd_end]
    Tbuf[new_incd] = True
    
    # Only needed if we eventually call Gsep. If only returning clique
    # "branch roots" over new edges, then this is not needed
    for v in new_incd:
        v_degree_old, curr = move_PX(GI, GS, GE,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, GS[v])
    
    #--------------------------------#

    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = GI_new[GS_new[u] : u_curr_new]    
    
    Tbuf[Padj_u] = False
    Tbuf[Padj_new_u] = False
    
    # Always keep the btw_new
    Tbuf[P[btw_new[P]]] = True
        
    # Some vertices in incd will not be reachable because we've already
    # removed a lot of nodes. Expand to find a vertex cover of the new edges.
#    for v in incd[~ Tbuf[incd]]:
    for v in branches[~ Tbuf[branches]]:
        if not Tbuf[v]:
            # By construction because of move_PX above, this will only
            # iterate over w's that are in incd
            for w in GI_new[GS_new[v] : GE_new[v]]:
                # if verbose:
                #     print indent, 'v:', v, 'w:', w
                w_pos = pos[w]
                Tbuf[w] = True
                if w_pos < PS or w_pos >= XE:
                    break
                
#    branches = incd[Tbuf[incd]]
    branches = branches[Tbuf[branches]]

#     # Alternative to setting P_btw_new to True and finding vertex cover of incd. Empirically slower
#     tmp = new_incd[pos[new_incd] < sep]
#     branches = np.concatenate((incd[Tbuf[incd]], tmp[Tbuf[tmp]]))

    Tbuf[Padj_new_u] = True
    Tbuf[Padj_u] = True

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
            
        Rbuf[RE] = v
        tree[stats[0]+1] = curr_node
        if btw_new[v]:
            C, CP, CN, tree = BK_hier_Gsep(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                HS, HE, HI,
                topo,
                Tbuf, depth+1,
                C, CP, CN, stats, tree)
        else:
            C, CP, CN, tree = BK_hier_dG(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                GS_new, GE_new, GI_new,
                HS, HE, HI,
                topo,
                Tbuf, depth+1,
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
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k)

    g = igraph.Graph(n=H.shape[0], edges=zip(*H.nonzero()), directed=True)
    topo = np.array(g.topological_sorting(mode='out'))
    
    start = time.time()
    C, CP, CN, tree = BK_hier_Gsep(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        H.indptr[:-1], H.indptr[1:], H.indices,
        topo,
        Tbuf, 0, C, CP, CN, tree)
    if verbose: print 'BK_hier_Gsep time:', time.time() - start
    C, CP, CN = trim_cliques(C, CP, CN)
    
    return C, CP, CN, tree

def BK_hier_dG_py(G, dG, H, verbose=False):
    k = G.shape[0]
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k)
        
    g = igraph.Graph(n=H.shape[0], edges=zip(*H.nonzero()), directed=True)
    topo = np.array(g.topological_sorting(mode='out'))

    start = time.time()
    C, CP, CN, tree = BK_hier_dG(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        H.indptr[:-1], H.indptr[1:], H.indices,
        topo,
        Tbuf, 0,
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

