import time

import numpy as np
from scipy.sparse import csc_matrix

import clixov_utils
import clique_maximal
import clique_maximum
import clique_maxcover
from constants import cache

def test_clique_maximal(method, k, r, s, check=True):
    G = get_test_network(method, k=k, r=r, s=s)
    start = time.time()
    cliques, cliques_indptr, cliques_n = clique_maximal.BKPivotSparse2_wrapper(G)
    print 'Time:', time.time() - start
    cliques = clixov_utils.cliques_to_csc(cliques, cliques_indptr, cliques_n, G.shape[0])

    if check:
        tmp1 = clixov_utils.csc_to_cliques_list(cliques)
        tmp2 = clique_maximal.get_cliques_igraph(G.shape[0], G, input_fmt='matrix')
        assert set(tmp1) == set(tmp2)    

def test_clique_maximal_new(method, k, r, s, check=True):
    Gold, Gnew = get_test_new_network(method, k=k, r=r, s=s, verbose=False)
    start = time.time()
    cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BKPivotSparse2_Gnew_wrapper(Gold, Gnew)
    print 'Time:', time.time() - start
    cliques = clixov_utils.cliques_to_csc(cliques, cliques_indptr, cliques_n, Gold.shape[0])

    if check:
        tmp1 = clixov_utils.csc_to_cliques_list(cliques)
        tmp2 = clique_maximal.get_cliques_igraph(Gold.shape[0], Gold + Gnew, Gnew, input_fmt='matrix')
        assert set(tmp1) == set(tmp2)

def test_clique_maximal_hier_new(method, k, r, s, check=True):
    Gold, Gnew = get_test_new_network(method, k=k, r=r, s=s, verbose=False)
    start = time.time()
    cliques, cliques_indptr, cliques_n, tree_size = clique_maximal.BKPivotSparse2_Gnew_wrapper(Gold, Gnew)
    print 'Time:', time.time() - start
    cliques = clixov_utils.cliques_to_csc(cliques, cliques_indptr, cliques_n, Gold.shape[0])

    if check:
        tmp1 = clixov_utils.csc_to_cliques_list(cliques)
        tmp2 = clique_maximal.get_cliques_igraph(Gold.shape[0], Gold + Gnew, Gnew, input_fmt='matrix')
        assert set(tmp1) == set(tmp2)

def test_clique_maximum(method, k, r, s, seed=None, verbose=False, check=True):
    G = get_test_network(method, k=k, r=r, s=s, seed=seed)
    start = time.time()
    cliques, _, __size = clique_maximum.MC_py(G)
    print 'Time:', time.time() - start

    if check:
        tmp1 = clixov_utils.csc_to_cliques_list(cliques)
            
        tmp2 = clique_maximal.get_cliques_igraph(G.shape[0], G, input_fmt='matrix')
        tmp2 = clixov_utils.tuples_to_csc(tmp2, G.shape[0])
        tmp2, _ = clixov_utils.get_largest_cliques(tmp2)
        tmp2 = clixov_utils.csc_to_cliques_list(tmp2)

        if verbose:
            print clixov_utils.sparse_str(G)

        try:
            assert set(tmp1) == set(tmp2)
        except:
            if verbose:
                print 'Shared:', sorted(set(tmp1) & set(tmp2))
                print 'Found:', sorted(set(tmp1) - set(tmp2))
                print 'Not found:', sorted(set(tmp2) - set(tmp1))
            raise

        print '%s cliques of size %s' % (len(tmp1), len(tmp1[0]))

def test_clique_maxcover_new(method, k, r, s, seed=None, verbose=False, check=True):
    Gold, Gnew = get_test_new_network(method, k=k, r=r, s=s, seed=seed, verbose=verbose)
    start = time.time()
    cliques, cliques_indptr, cliques_n, tree_size = clique_maxcover.BKPivotSparse2_Gnew_cover_wrapper(Gold, Gnew)
    print 'Time:', time.time() - start
    cliques = clixov_utils.cliques_to_csc(cliques, cliques_indptr, cliques_n, Gold.shape[0])

    cover_idx = clixov_utils.get_largest_clique_covers(cliques, Gnew)
    cliques = cliques[:,cover_idx]

    if check:
        tmp1 = clixov_utils.csc_to_cliques_list(cliques)
        start = time.time()
        tmp2 = clique_maxcover.max_clique_cover(Gnew, Gold + Gnew, verbose=verbose)
        print 'Alternative time:', time.time() - start

        try:
            assert set(tmp1) == set(tmp2)
        except:
            if verbose:
                print 'Shared:', sorted(set(tmp1) & set(tmp2))
                G = Gold + Gnew
                print 'Found:', sorted(set(tmp1) - set(tmp2))
                print 'Not found:', sorted(set(tmp2) - set(tmp1))
                for c in sorted(set(tmp1) - set(tmp2)):
                    try:
                        clixov_utils.assert_clique(c, G)
                    except:
                        print 'Not a real clique:', c
                        print G[:,c][c,:].toarray().astype(np.int32)
                        raise
            raise

def get_test_network(method, k, r, s, invert=False, seed=None):
    start = time.time()
    if seed is None:
        seed = np.random.randint(0, 2**32, 1, dtype=np.uint32)[0]
        print 'Seed:', seed
    np.random.seed(seed)
    
    if method=='cluster':
        G = np.zeros((k,k), dtype=np.int32, order='F')
        clusters = [np.sort(np.unique(np.random.randint(0, k, r))) for i in range(s)]
        for c in clusters:
            G[np.ix_(c,c)] = True
        np.fill_diagonal(G, 0)
        G = csc_matrix(G)
    elif method=='erdos':
        i = np.random.randint(0, k, r)
        j = np.random.randint(0, k, r)
        
        # Remove indices that are equal
        tmp = i!=j
        if tmp.sum() > 0:
            i, j = i[tmp], j[tmp]
        # Switch indices that are in lower triangle
        tmp = i>j
        if tmp.sum() > 0:
            i_tmp = i.copy()
            i[tmp] = j[tmp]
            j[tmp] = i_tmp[tmp]
        # Take only r of the indices
        i, j = i[:r], j[:r]        

        G = np.zeros((k,k), dtype=np.bool, order='F')
        G[i, j] = True
        G = np.logical_or(G, G.T)
        np.fill_diagonal(G, 0)
        G = G.astype(np.int32, order='F')
        G = csc_matrix(G)

#        print G.nonzero()
        
    if invert:
        G = 1 - G.toarray()
        np.fill_diagonal(G, 0)
        G = csc_matrix(G)
    
    #print 'Simulation time:', time.time() - start
    return G

def get_test_new_network(method,
                         k=None,
                         r=None,
                         s=None,
                         p=0.5,
                         n=None,
                         seed=None,
                         verbose=False):
    """
    ## Cluster model
    if method=='cluster':
        k : number of nodes
        r : size of each cluster
        s : number of clusters

    ## Erdos-Renyi model
    if method=='erdos':        
        k : number of nodes
        k * r : probability of an edge being added
        p : probability of an added edge being new

    ## Hierarchy model
    if method=='hierarchy':

    """
    
    if seed is None:
        seed = np.random.randint(0, 2**32, 1, dtype=np.uint32)[0]
        print 'Seed:', seed

    np.random.seed(seed=seed)
    
    if method=='cluster':
        Gold = np.zeros((k,k), dtype=np.int32, order='F')
        old_clusters = [np.random.randint(0, k, r) for i in range(s)]
        for c in old_clusters:
            Gold[np.ix_(c,c)] = True
        np.fill_diagonal(Gold, 0)
        
        Gnew = np.zeros((k,k), dtype=np.int32, order='F')
        new_clusters = [np.random.randint(0, k, r) for i in range(s)]
        for c in new_clusters:
            Gnew[np.ix_(c,c)] = True
        np.fill_diagonal(Gnew, 0)
        
        Gnew -= (Gnew * Gold)
        Gold, Gnew = csc_matrix(Gold), csc_matrix(Gnew)        
    elif method=='erdos':
        G = np.zeros((k,k), dtype=np.bool, order='F')
        G[np.random.randint(0, k, k * r), np.random.randint(0, k, k * r)] = True
        G = np.logical_or(G, G.T)
        np.fill_diagonal(G, 0)
        G = G.astype(np.bool, order='F')

        Gnew = G.copy()
        Gnew.data[np.random.random(G.data.size) < p] = 0
        Gnew.eliminate_zeros()
        Gnew = csc_matrix((Gnew.T.toarray() * Gnew.toarray()))
        Gold = csc_matrix(G.toarray() & ~ Gnew.toarray())
    elif method=='hierarchy':
        edges, H = generate_hierarchy(n, p, r, seed=seed)

        Gold = H + (H.dot(H.T) > 0)
        
        # Gold = np.zeros((n,n), np.bool_)

        
        
        # for i in range(H.shape[1]):
        #     members = H[:n, i].nonzero()[0]
        #     Gold[np.ix_(members, members)] = 1

#        get_test_network('cluster', 
        
    else:
        raise Exception('Unsupported method')
    
    assert Gnew.sum() > 0
    assert (Gnew.multiply(Gold)).sum() == 0
        
    if verbose:
        print Gnew.sum() / 2, Gold.sum() / 2, 'Gnew/old edges'
        # print Gold.toarray().astype(np.int32)
#        print 'Gold:', sorted(set([tuple(sorted(x)) for x in zip(*Gold.nonzero())]))
        print 'Gold:', clixov_utils.sparse_str(Gold)
        # print Gnew.toarray().astype(np.int32)
#        print 'Gnew:', sorted(set([tuple(sorted(x)) for x in zip(*Gnew.nonzero())]))
        print 'Gnew:', clixov_utils.sparse_str(Gnew)
    
    # G_start, G_end, G_indices = Gold.indptr[:-1], Gold.indptr[1:], Gold.indices
    # Gnew_start, Gnew_end, Gnew_indices = Gnew.indptr[:-1], Gnew.indptr[1:], Gnew.indices
    # print 'Gold:', [(v, G_start[v], G_end[v], list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(k) if G_end[v] > G_start[v]]
    # print 'Gnew:', [(v, Gnew_start[v], Gnew_end[v], list(G_indices[Gnew_start[v] : Gnew_end[v]])) for v in np.arange(k) if Gnew_end[v] > Gnew_start[v]]

    return Gold, Gnew

def generate_hierarchy(n, p, r, seed=None):
    """
    n : leaves
    p : max number of frontier nodes to aggregate. uniform distribution between 2 and p.
    r : max number of other nodes to aggregate. These nodes must not be descendants among the chosen frontier nodes. Uniform distribution between 0 and r.
    """

    if seed is None:
        seed = np.random.randint(0, 2**32, 1, dtype=np.uint32)[0]
        print 'Seed:', seed

    frontier = np.arange(n)
    parent = n
    edges = []
    H = np.zeros((10*n,10*n), np.bool_)
    np.fill_diagonal(H, 1)
        
    while frontier.size > 1:
        children = np.random.choice(
            frontier,
            np.random.randint(2, min(p, frontier.size) + 1),
            replace=False)
        print 'children:', children
        non_descendants = (H[:parent, children].sum(1) == 0).nonzero()[0]
        assert np.intersect1d(non_descendants, children).size == 0
        if non_descendants.size > 0:
            print 'non_descendants:', non_descendants
            other_children = np.random.choice(
                non_descendants,
                np.random.randint(0, min(r, non_descendants.size) + 1),
                replace=False)
            print 'other children:', other_children
            children = np.append(children, other_children)
        print 'parent:', parent
        
        for c in children:
            edges.append((c, parent))

        H[children, parent] = 1
        H[:parent, parent] = H[:parent, children].sum(1) > 0

        frontier = np.append(np.setdiff1d(frontier, children), parent)
        
        parent += 1

    H = H[:parent,:parent]
    
    return edges, H
