import time, os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from numba import jit

import scipy, scipy.sparse
from scipy.sparse import isspmatrix_csc, isspmatrix_csr, issparse, isspmatrix, csc_matrix, csr_matrix, coo_matrix

import clixov_utils
from clixov_utils import *
import clique_maximal
import clique_maximum
from clique_atomic import *
from color import set_color, get_branch_sizes, get_branch_sizes_vw, color_nodes, count_unique
from degeneracy import get_degeneracy

verbose = False

def max_clique_cover(to_cover, G, verbose=False, use_pmc=False):
    """Call's PMC maximum clique on each new edge"""
    
    if isspmatrix_csc(G):
        G = csr_matrix(G)
        assert isspmatrix_csr(G)
    else:
        raise Exception()
    
    clique_list = []
    clique_set = set([])
    
    # Unexplained edges
    e_i, e_j = to_cover.nonzero()
    tmp = e_i < e_j
    e_i, e_j = e_i[tmp], e_j[tmp]
    edges = set(zip(e_i, e_j))

    it = 0
    start = time.time()
        
    # Iterate until there are no more unexplained edges
    while len(edges) > 0:
        # if verbose:
        #     print 'edges:', edges

        it += 1
        i, j = edges.pop()
        
        # nonzero()[1] assumes that G is sparse. Otherwise do nonzero()[0]
        if verbose:
            print i, j, G[i, :].nonzero()[1], G[j, :].nonzero()[1]
        P = np.intersect1d(G[i, :].nonzero()[1], G[j, :].nonzero()[1])
        P.sort()

        if P.size > 1:
            G_P = G[P,:][:,P]

            if use_pmc:
                import pmc
                clique = pmc.pmc(G_P, threads=48, verbose=False)            
                if clique.size==0:
                    # Pick a random elt from P to make the clique
                    clique = (P[0], i, j)
                else:
                    clique = tuple([P[k] for k in clique] + [i,j])
                cliques = [clique]
            else:
                cliques, _, _ = clique_maximum.MC_py(G_P, verbose=False)
                cliques = [tuple([P[k] for k in c] + [i,j]) for c in clixov_utils.csc_to_cliques_list(cliques)]

        elif P.size==1:
            cliques = [(P[0], i, j)]
        else:
            cliques = [(i,j)]

        cliques = [tuple(sorted(c)) for c in cliques]

        if verbose:
            print 'c:', cliques, 'P:', P, 'i,j:', (i,j)
            
        clique_list.extend(cliques)

    clique_list = sorted(set(clique_list))
    
    print 'Augment iterations:', it, 'time:', time.time() - start
    
    return clique_list
    
def max_clique_cover_new(to_cover, G, dG, verbose=False, pmc=False):
    """Call's PMC maximum clique on each new edge"""
    
    if isspmatrix_csc(G):
        G = csr_matrix(G)
        assert isspmatrix_csr(G)
    else:
        raise Exception()
    
    clique_list = []
    clique_set = set([])
    
    # Unexplained edges
    e_i, e_j = to_cover.nonzero()
    tmp = e_i < e_j
    e_i, e_j = e_i[tmp], e_j[tmp]
    edges = set(zip(e_i, e_j))

    it = 0
    start = time.time()
        
    # Iterate until there are no more unexplained edges
    while len(edges) > 0:
        # if verbose:
        #     print 'edges:', edges

        it += 1
        i, j = edges.pop()
        
        # nonzero()[1] assumes that G is sparse. Otherwise do nonzero()[0]
        if verbose:
            print i, j, G[i, :].nonzero()[1], G[j, :].nonzero()[1]
        P = np.intersect1d(G[i, :].nonzero()[1], G[j, :].nonzero()[1])
        P.sort()

        if P.size > 1:
            G_P = G[P,:][:,P]

            if pmc:
                clique = pmc.pmc(G_P, threads=48, verbose=False)            
                if clique.size==0:
                    # Pick a random elt from P to make the clique
                    clique = (P[0], i, j)
                else:
                    clique = tuple([P[k] for k in clique] + [i,j])
                cliques = [clique]
            else:
                cliques, _, _ = clique_maximum.MC_py(G_P, verbose=False)
                cliques = [tuple([P[k] for k in c] + [i,j]) for c in clixov_utils.csc_to_cliques_list(cliques)]

        elif P.size==1:
            cliques = [(P[0], i, j)]
        else:
            cliques = [(i,j)]

        cliques = [tuple(sorted(c)) for c in cliques]

        if verbose:
            print 'c:', cliques, 'P:', P, 'i,j:', (i,j)
            
        clique_list.extend(cliques)

    clique_list = sorted(set(clique_list))
    
    print 'Augment iterations:', it, 'time:', time.time() - start
    
    return clique_list


