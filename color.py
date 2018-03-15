import numpy as np
from numba import jit
from constants import cache

@jit(nopython=True, cache=cache)
def get_branch_sizes(dG, dGS, dGE,
                     GI, GS, GE,
                     pos, sep, PS, XE, colors, nei_bool, nei_list, branches):
    branches_sizes = np.empty(branches.size, np.int32)    
    for v_i in range(branches.size):
        v = branches[v_i]
        nei_count = 0
        for w in dG[dGS[v] : dGE[v] : 2]:                
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif PS <= w_pos and w_pos < sep:
                c = colors[w_pos - PS]
                if not nei_bool[c]:
                    nei_bool[c] = True
                    nei_list[nei_count] = c
                    nei_count += 1
        for w in GI[GS[v] : GE[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif PS <= w_pos and w_pos < sep:
                c = colors[w_pos - PS]
                if not nei_bool[c]:
                    nei_bool[c] = True
                    nei_list[nei_count] = c
                    nei_count += 1
        nei_bool[nei_list[:nei_count]] = False
        branches_sizes[v_i] = nei_count
    return branches_sizes


@jit(nopython=True, cache=cache)
def get_branch_sizes_v(dG, dGS, dGE,
                       GI, GS, GE,
                       pos, sep, PS, XE, colors, nei_bool, nei_list, v):
    nei_count = 0
    for w in dG[dGS[v] : dGE[v] : 2]:                
        w_pos = pos[w]
        if w_pos < PS or w_pos >= XE:
            break
        elif PS <= w_pos and w_pos < sep:
            c = colors[w_pos - PS]
            if not nei_bool[c]:
                nei_bool[c] = True
                nei_list[nei_count] = c
                nei_count += 1
    for w in GI[GS[v] : GE[v]]:
        w_pos = pos[w]
        if w_pos < PS or w_pos >= XE:
            break
        elif PS <= w_pos and w_pos < sep:
            c = colors[w_pos - PS]
            if not nei_bool[c]:
                nei_bool[c] = True
                nei_list[nei_count] = c
                nei_count += 1
    nei_bool[nei_list[:nei_count]] = False
    return nei_count

@jit(nopython=True, cache=cache)
def get_branch_sizes_vw(dG, dGS, dGE,
                        GI, GS, GE,
                        pos, sep, PS, XE, colors_v, nei_bool, nei_list, v):
    nei_count = 0
    for w in dG[dGS[v] : dGE[v] : 2]:
        w_pos = pos[w]
        if w_pos < PS or w_pos >= XE:
            break
        elif PS <= w_pos and w_pos < sep:
            c = colors_v[w]
            if not nei_bool[c]:
                nei_bool[c] = True
                nei_list[nei_count] = c
                nei_count += 1
    for w in GI[GS[v] : GE[v]]:
        w_pos = pos[w]
        if w_pos < PS or w_pos >= XE:
            break
        elif PS <= w_pos and w_pos < sep:
            c = colors_v[w]
            if not nei_bool[c]:
                nei_bool[c] = True
                nei_list[nei_count] = c
                nei_count += 1
    nei_bool[nei_list[:nei_count]] = False
    return nei_count

# @jit(nopython=True, cache=cache)
# def get_branch_sizes_vw(GI, GS, GE,
#                         pos, sep, PS, XE,
#                         colors_v, nei_bool, nei_list, v):
#     nei_count = 0
#     for w in GI[GS[v] : GE[v]]:
#         w_pos = pos[w]
#         if w_pos < PS or w_pos >= XE:
#             break
#         elif PS <= w_pos and w_pos < sep:
#             c = colors_v[w]
#             if not nei_bool[c]:
#                 nei_bool[c] = True
#                 nei_list[nei_count] = c
#                 nei_count += 1
#     nei_bool[nei_list[:nei_count]] = False
#     return nei_count

@jit(nopython=True)
def count_unique(arr, Fbuf, stack):
    stack_n = 0
    for v in arr:
        if not Fbuf[v]:
            stack[stack_n] = v
            stack_n += 1
            Fbuf[v] = True
    Fbuf[stack[:stack_n]] = False
    return stack_n

@jit(nopython=True, cache=cache)
def color_nodes(GI, GS, GE, order, verbose=False):
    # if verbose: print '************************'
    n = order.size
    colors = np.empty(n, np.int32)
    branches = np.empty(n, np.int32)
    i = 0
    c = 1
    
    colored = np.zeros(order.max() + 1, np.bool_)
    first_to_color = 0
    
    while i < n:
        # if verbose: print 'i:', i
        can_color = colored.copy()
        # if verbose: print 'can_color:', can_color.astype(np.int32)
        # if verbose: print 'colors/branches:', zip(colors, branches)[:i]
        for j in range(first_to_color, n):
            v = order[j]
            # if verbose: print 'j:', j, 'v:', v, 'can_color[v]:', can_color[v]
            if not can_color[v]:
                # if verbose: print 'Coloring node', v
                colors[i] = c
                colored[v] = True                    
                branches[i] = v
                can_color[GI[GS[v] : GE[v]]] = True
                i += 1
                # if verbose: print 'colors/branches:', zip(colors, branches)[:i]
                # if verbose: print 'GI[GS[v] : GE[v]]:', GI[GS[v] : GE[v]]
                    
                if j == first_to_color:
                    first_to_color += 1
                    # if verbose: print 'first_to_color:', first_to_color

        c += 1
    return colors, branches


@jit(nopython=True, cache=cache)
def set_color(dG, dGS, dGE,
              GI, GS, GE,
              pos, P, sep, PS, XE,
              v_i, colors, nei_bool, nei_list, max_colors):
    v = P[v_i]
    if colors[v_i] == 0:
        nei_count = 0
        for w in dG[dGS[v] : dGE[v] : 2]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif PS <= w_pos and w_pos < sep:
                c = colors[w_pos - PS]
                if c!=0 and not nei_bool[c]:
                    nei_bool[c] = True
                    nei_list[nei_count] = c
                    nei_count += 1
        for w in GI[GS[v] : GE[v]]:
            w_pos = pos[w]
            if w_pos < PS or w_pos >= XE:
                break
            elif PS <= w_pos and w_pos < sep:
                c = colors[w_pos - PS]
                if c!=0 and not nei_bool[c]:
                    nei_bool[c] = True
                    nei_list[nei_count] = c
                    nei_count += 1
        nei_list[:nei_count].sort()
        if nei_count == 0 or nei_list[0] > 1:
            colors[v_i] = 1
        else:
            for i in range(nei_count - 1):
                if nei_list[i+1] - nei_list[i] > 1:
                    colors[v_i] = nei_list[i] + 1                        
                    break
            if colors[v_i]==0:
                colors[v_i] = nei_list[nei_count - 1] + 1
        max_colors = max(max_colors, colors[v_i])
        nei_bool[nei_list[:nei_count]] = False        
    return max_colors

