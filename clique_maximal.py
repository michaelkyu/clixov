import time
import igraph
import numpy as np
from clique_atomic import *
from constants import cache, parallel

debug = False

#interest = [1640, 2670, 4782, 6681, 6918]
#interest = [1640, 2670, 6681, 6918]
#interest = [326, 34, 247, 328]
interest = []


from numba import prange

@jit(nopython=True, cache=cache, parallel=True)
def BK_par(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
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
    
    if depth==0:        
#        C_list = []
        for v_i in prange(branches.size):
            v = branches[v_i]

            C2, CP2, CN2 = C.copy(), CP.copy(), CN
            PX2, pos2 = PX.copy(), pos.copy()
            Rbuf2, Tbuf2 = Rbuf.copy(), Tbuf.copy()
            RE2, sep2, PS2, XE2, oldPS2, oldXE2 = oldPS, oldXE, PS, XE, sep, RE
            
            for v_ii in range(v_i):
                sep2 -= 1
                swap_pos(PX2, pos2, branches[v_ii], sep2)
            
            new_PS, new_XE = update_PX(GI, GS, GE, oldPS2, oldXE2, PS2, XE2, sep2, PX2, pos2, v, sep2, sep2)            
            Rbuf2[RE2] = v

            # GI2, GS2, GE2 = GI.copy(), GS.copy(), GE.copy()
                        
            GS2, GE2 = GS.copy(), GE.copy()
            tmp = PX2[new_PS:new_XE]
            size = 0
            for x in tmp:
                size += GE2[x] - GS2[x]
            # print size

            if size > 0:
                GI2 = np.empty(size, GI.dtype)
                #GI2 = np.empty((GE2[tmp] - GS2[tmp]).sum(), GI.dtype)
                #GI2 = np.empty(GI.size, GI.dtype)
                #GI2 = np.empty(10, GI.dtype)
                curr = 0
                for x_i in range(tmp.size):
                    x = tmp[x_i]
                    offset = GE2[x] - GS2[x]
                    assert curr + offset <= GI2.size
                    GI2[curr : curr + offset] = GI[GS[x]:GE[x]]
                    GS2[x] = curr
                    GE2[x] = curr + offset
                    curr += offset

                C2, CP2, CN2 = BK_par(
                    Rbuf2, RE + 1, PX2, new_PS, sep2, new_XE, PS2, XE2, pos2,
                    GS2, GE2, GI2, Tbuf2, depth+1,
                    C2, CP2, CN2)
    else:
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

@jit(nopython=True, cache=cache, parallel=parallel)
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

def BK_py(G, PX=None, par=False):
    k = G.shape[0]
    custom_PX = PX is not None
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k, PX=PX)
    if custom_PX:
        initialize_PX(G.indices, G.indptr[:-1], G.indptr[1:], pos, PX)

    if par:
        C, CP, CN = BK_par(
            R, RE, PX, PS, sep, XE, PS, XE, pos,
            G.indptr[:-1], G.indptr[1:], G.indices,
            Tbuf, 0, C, CP, CN)
    else:
        C, CP, CN = BK(
            R, RE, PX, PS, sep, XE, PS, XE, pos,
            G.indptr[:-1], G.indptr[1:], G.indices,
            Tbuf, 0, C, CP, CN)

    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN

@jit(nopython=True, cache=cache, parallel=parallel)
def BK_Gsep(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
            GS, GE, GI, dGS, dGE, dGI,
            Tbuf, depth, C, CP, CN, stats, tree):
    if tree.shape[1] == stats[0] + 2:
        tree = expand_2d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]
    tree[1,curr_node] = Rbuf[RE-1]
    
    R = Rbuf[:RE]
    P, X = PX[PS:sep], PX[sep:XE]
    
#     indent = '\t' * depth
#     print indent, '---------------'
#     print indent, 'DEPTH:', depth
#     print indent, 'PX:', PX`
#     print indent, 'PS:', PS, 'XE:', XE, 'oldPS:', oldPS, 'oldXE:', oldXE, 'sep:', sep
#     print indent, 'R:', R, 'P:', P, 'X:', X
    
    if P.size==0:
        if X.size==0:
            C, CP, CN = update_cliques(C, CP, CN, R)
        return C, CP, CN, tree

    u = -1
    max_degree = -1
    for v in PX[PS:XE][::-1]:
        v_degree, curr_new = move_PX(dGI, dGS, dGE,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, dGS[v])        
        v_degree, curr = move_PX(GI, GS, GE,
                                     pos, oldPS, oldXE, PS, XE, sep, v_degree, v, GS[v])   
        if v_degree > max_degree: 
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new

#     # Swap the pivot to the beginning of P so that it is the first branch
#     if pos[u] < sep:
#         swap_pos(PX, pos, u, PS)
        
    Padj_u = pos[GI[GS[u] : u_curr]]
    Padj_new_u = pos[dGI[dGS[u] : u_curr_new]]
    
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
        new_PS, new_XE = update_PX(dGI, dGS, dGE, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)

        Rbuf[RE] = v
        tree[0, stats[0]+1] = curr_node
        C, CP, CN, tree = BK_Gsep(
            Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
            GS, GE, GI,
            dGS, dGE, dGI,
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

@jit(nopython=True, cache=cache, parallel=parallel)
def BK_dG(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
          GS, GE, GI, dGS, dGE, dGI,
          Tbuf, depth,
          btw_new, btw_stack, btw_end,
          C, CP, CN, stats, tree):
    """
    btw_new[v]: Is there a new edge crossing btw R and node v in P?
    """
    if tree.shape[1] == stats[0] + 2:
        tree = expand_2d_arr(tree)
    stats[0] += 1
    curr_node = stats[0]
    tree[1,curr_node] = Rbuf[RE-1]
    
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
        return C, CP, CN, tree
    
    incd = np.empty(sep - PS, PX.dtype)
    incd_count = 0
    max_degree = -1
    X_incd = np.empty(XE - sep, PX.dtype)
    X_incd_count = 0
    
    # Iterate over new edges from X too
    for v in X:
        v_degree_new, curr_new = move_PX(dGI, dGS, dGE,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, dGS[v])        
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new = v, curr_new
        if v_degree_new > 0:
            tmp, curr = move_PX(GI, GS, GE,
                                pos, oldPS, oldXE, PS, XE, sep, v_degree_new, v, GS[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
    for v in P:
        v_degree_new, curr_new = move_PX(dGI, dGS, dGE,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, dGS[v])        
        if v_degree_new > max_degree:
            max_degree = v_degree_new
            u, u_curr_new, u_incd = v, curr_new, incd_count
        if v_degree_new > 0 or btw_new[v]:
            incd[incd_count] = v
            incd_count += 1
            
    u_curr = GE[u]

    if incd_count == 0:
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

    # Padj_u = GI[GS[u] : u_curr]
    # Padj_new_u = dGI[dGS[u] : u_curr_new]    

    # Tbuf[Padj_new_u] = False
    # Tbuf[P[btw_new[P]]] = True
    # branches = branches[Tbuf[branches]]
    # # Always keep the btw_new

    # assert u in incd

    # new_incd

    # Tbuf[Padj_u] = False
    # Tbuf[Padj_new_u] = True
    # Tbuf[Padj_u] = True


    # for v in incd:
    #     if not Tbuf[v]:
    #         for w in dGI[dGS[v] : dGE[v]]:
    #             w_pos = pos[w]
    #             if w_pos < PS or w_pos >= XE:
    #                 break
    #             if not Tbuf[w]:
    #                 Tbuf[v] = True
    #                 Tbuf[w] = True
    #                 break

        # # Some vertices will not be reachable because we've already
    # # removed a lot of nodes. Expand to find a vertex cover of incd.
    # for v in incd:
    #     if not Tbuf[v]:
    #         needed = False
    #         # By construction because of move_PX above, this will only
    #         # iterate over w's that are in incd
    #         for w in dGI[dGS[v] : dGE[v]]:                
    #             w_pos = pos[w]
    #             if w_pos < PS or w_pos >= XE:
    #                 break
    #             needed = needed or (not Tbuf[w])


        

    ##-----------------------------------------#

    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = dGI[dGS[u] : u_curr_new]    
    
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
            for w in dGI[dGS[v] : dGE[v]]:
                w_pos = pos[w]
                if w_pos < PS or w_pos >= XE:
                    break
                Tbuf[w] = True
    
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
        for w in dGI[dGS[v] : dGE[v]]:
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
        tree[0, stats[0]+1] = curr_node
        if btw_new[v]:
            C, CP, CN, tree = BK_Gsep(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                dGS, dGE, dGI,
                Tbuf, depth+1,
                C, CP, CN, stats, tree)
        else:
            C, CP, CN, tree = BK_dG(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                dGS, dGE, dGI,
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

def BK_dG_py(G, dG, PX=None):
    k = G.shape[0]
    custom_PX = PX is not None
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k, PX=PX)
    
    tree = np.asfortranarray(np.zeros((14, 100000), np.int32))
    tree.fill(-1)

    if custom_PX:
        initialize_PX(G.indices, G.indptr[:-1], G.indptr[1:], pos, PX)
        initialize_PX(dG.indices, dG.indptr[:-1], dG.indptr[1:], pos, PX)
    
    C, CP, CN, tree = BK_dG(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        Tbuf, 0, btw_new, btw_stack, btw_end,
        C, CP, CN, stats, tree)

    tree = tree[:,:stats[0]+1]    
    C, CP, CN = trim_cliques(C, CP, CN)
    return C, CP, CN, tree

@jit(nopython=True, cache=cache, parallel=parallel)
def BK_hier_Gsep(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
                 GS, GE, GI, dGS, dGE, dGI,
                 HS, HE, HI,
                 topo,
                 Fbuf, Tbuf, depth, C, CP, CN,
                 stats, tree):
    #### TODO: find a better pivot by considering the total node degree, not just new degree
    
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
        if X.size==0:
            # if verbose:
            #     print indent, depth, 'Returning', 'R:', R, 'P:', P, 'X:', X            
            C, CP, CN = update_cliques(C, CP, CN, R)
        return C, CP, CN, tree
        
    # Filter branches. Follow the topological sorting, going top-down
    # the hierarchy
    for v in P[np.argsort(topo[P])]:
        if Tbuf[v]:
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
    
    branches = P[Tbuf[P]]

    # if verbose:
    #     print indent, 'branches:', branches
        
    u = -1
    max_degree = -1
    for v in PX[PS:XE][::-1]:
        v_degree, curr_new = move_PX(dGI, dGS, dGE,
                                     pos, oldPS, oldXE, PS, XE, sep, 0, v, dGS[v])        
        v_degree, curr = move_PX(GI, GS, GE,
                                 pos, oldPS, oldXE, PS, XE, sep, v_degree, v, GS[v])
        if Tbuf[v] and (v_degree > max_degree): 
            max_degree = v_degree
            u, u_curr, u_curr_new = v, curr, curr_new

    # TODO: make this more efficient by not having to make it |P| time
    Tbuf[P] = True
    
    Padj_u = GI[GS[u] : u_curr]
    Padj_new_u = dGI[dGS[u] : u_curr_new]

    Tbuf[Padj_u] = False
    Tbuf[Padj_new_u] = False
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
        new_PS, new_XE = update_PX(dGI, dGS, dGE, oldPS, oldXE, PS, XE, sep, PX, pos, v, new_PS, new_XE)

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
            dGS, dGE, dGI,
            HS, HE, HI,
            topo,
            Fbuf, Tbuf, depth+1,
            C, CP, CN, stats, tree)
            
        # Swap v to the end of P, and then decrement separator
        sep -= 1
        swap_pos(PX, pos, v, sep)

    for v in branches[::-1]:
        # Move v to the beginning of X and increment separator
        swap_pos(PX, pos, v, sep)
        sep += 1

    return C, CP, CN, tree

@jit(nopython=True, cache=cache, parallel=parallel)
def BK_hier_dG(Rbuf, RE, PX, PS, sep, XE, oldPS, oldXE, pos,
               GS, GE, GI, dGS, dGE, dGI,                 
               HS, HE, HI,
               topo,
               Fbuf, Tbuf, depth,
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
        v_degree_new, curr_new = move_PX(dGI, dGS, dGE,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, dGS[v])        
        if v_degree_new > 0:
            tmp, curr = move_PX(GI, GS, GE,
                                pos, oldPS, oldXE, PS, XE, sep, v_degree_new, v, GS[v])
            X_incd[X_incd_count] = v
            X_incd_count += 1
    for v in P:
        v_degree_new, curr_new = move_PX(dGI, dGS, dGE,
                                         pos, oldPS, oldXE, PS, XE, sep, 0, v, dGS[v])        
        if v_degree_new > 0 or btw_new[v]:
            incd[incd_count] = v
            incd_count += 1

    if incd_count == 0:
        # if verbose:
        #     print indent, 'Returning because incd.size==0', 'R:', R, 'P:', P, 'X:', X
            
        return C, CP, CN, tree
    incd = incd[:incd_count]
    X_incd = X_incd[:X_incd_count]

    is_incd = Fbuf #np.zeros(PX.size, np.bool_)
    is_incd[incd] = True

    # print('incd:', incd.size)
    
    # Filter branches. Follow the topological sorting, going top-down
    # the hierarchy
    #for v in incd[np.argsort(topo[incd])]:
    for v in P[np.argsort(topo[P])]:
        if Tbuf[v]:
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

    is_incd[incd] = False
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
    is_branch = Fbuf # np.zeros(PX.size, np.bool_)
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

        v_degree, curr_new = move_PX(dGI, dGS, dGE,
                                     pos, oldPS, oldXE, PS, XE, sep, v_degree_old, v, dGS[v])
        
#        if v_degree > max_degree:
        if is_branch[v] and v_degree > max_degree:
            max_degree = v_degree
            u, u_curr_new, u_curr = v, curr_new, curr

    is_branch[branches] = False
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
    Padj_new_u = dGI[dGS[u] : u_curr_new]    
    
    Tbuf[Padj_u] = False
    Tbuf[Padj_new_u] = False

    ## TODO: maybe shouldn't do this
    # Always keep the btw_new
    Tbuf[P[btw_new[P]]] = True
        
    # Some vertices in incd will not be reachable because we've already
    # removed a lot of nodes. Expand to find a vertex cover of the new edges.
    for v in branches[~ Tbuf[branches]]:
        if not Tbuf[v]:
            # By construction because of move_PX above, this will only
            # iterate over w's that are in incd
            for w in dGI[dGS[v] : dGE[v]]:
                w_pos = pos[w]
                if w_pos < PS or w_pos >= XE:
                    break
                Tbuf[w] = True
    branches = branches[Tbuf[branches]]

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
        for w in dGI[dGS[v] : dGE[v]]:
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
                dGS, dGE, dGI,
                HS, HE, HI,
                topo,
                Fbuf, Tbuf, depth+1,
                C, CP, CN, stats, tree)
        else:
            C, CP, CN, tree = BK_hier_dG(
                Rbuf, RE + 1, PX, new_PS, sep, new_XE, PS, XE, pos,
                GS, GE, GI,
                dGS, dGE, dGI,
                HS, HE, HI,
                topo,
                Fbuf, Tbuf, depth+1,
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

def BK_hier_dG_py(G, dG, H, verbose=False):
    k = G.shape[0]
    PX, pos, R, RE, sep, PS, sep, XE, Fbuf, Tbuf, C, CP, CN, btw_new, btw_stack, btw_end, stats, tree = initialize_structures(k)
        
    g = igraph.Graph(n=H.shape[0], edges=zip(*H.nonzero()), directed=True)
    topo = np.array(g.topological_sorting(mode='out'))
    topo = topo[::-1]
    
    #print 'topo:', topo[:5].tolist(), topo[-5:].tolist()
    
    start = time.time()
    C, CP, CN, tree = BK_hier_dG(
        R, RE, PX, PS, sep, XE, PS, XE, pos,
        G.indptr[:-1], G.indptr[1:], G.indices,
        dG.indptr[:-1], dG.indptr[1:], dG.indices,
        H.indptr[:-1], H.indptr[1:], H.indices,
        topo,
        Fbuf, Tbuf, 0,
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

