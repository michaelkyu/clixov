import time
import numpy as np
from scipy.sparse import isspmatrix, isspmatrix_csc, isspmatrix_csr, csc_matrix, csr_matrix
from collections import OrderedDict, Counter

from mkl_spgemm import dot

from numba import jit


@jit(nopython=True)
def trim_cliques(cliques, cliques_indptr, cliques_n):
    return cliques[:cliques_indptr[cliques_n]], cliques_indptr[:cliques_n+1], cliques_n
        
# def as_dense(X):    
#     if isspmatrix(X):
#         return X.toarray()
#     elif isinstance(X, np.matrixlib.defmatrix.matrix):
#         return np.array(X)
#     else:
#         return X

def as_dense_array(X, order=None):
    if isspmatrix(X):
        return X.toarray(order=order)
    else:
        if order is None:
            order = 'K'
        return np.array(X, order=order)

def as_dense_flat(X):
    return as_dense_array(X).reshape(-1)

def cliques_to_csc(cliques, cliques_indptr, cliques_n, n, dtype=np.int32, copy=False):
    return csc_matrix((np.ones(cliques.size, dtype), cliques, cliques_indptr),
                      (n, cliques_n), copy=copy)

def csc_to_cliques_list(cliques):
    return [tuple(sorted(cliques.indices[cliques.indptr[i]:cliques.indptr[i+1]])) for i in range(cliques.shape[1])]

def tuples_to_csc(cliques, n, dtype=None):
    if len(cliques)==0:
        assert dtype is not None, 'Input is empty list of cliques. Must specify dtype for csc_matrix'
        return csc_matrix((n, len(cliques)), dtype=dtype)
    else:
        indices = np.concatenate(cliques)
        indptr = np.append(0, np.cumsum([len(c) for c in cliques]))
        return csc_matrix((np.ones(indices.size, np.int32), indices, indptr),
                          (n, len(cliques)), dtype=dtype)

def adjacency_to_edges_mat(G):
    assert isspmatrix_csc(G)
    
    G = G.copy()
    G.data[G.indices < np.repeat(np.arange(G.shape[1]), G.indptr[1:] - G.indptr[:-1])] = 0
    G.eliminate_zeros()
    
    indices = np.empty(2 * G.indices.size, np.int32)
    indices[::2] = G.indices
    indices[1::2] = np.repeat(np.arange(G.shape[1]), as_dense_flat(G.sum(0)).astype(indices.dtype))
    indptr = np.arange(0, 2 * G.indices.size + 2, 2)
    edges_mat = csc_matrix((np.ones(indices.size, G.dtype), indices, indptr),
                       (G.shape[0], G.indices.size))
    edges_mat.sort_indices()
    return edges_mat
    
def get_largest_clique_covers(cliques, G):    
    # Check symmetry
    assert G.multiply(G.T).sum() == G.sum()
    
    # Take upper-right triangle
    G = G.copy()
    G.data[G.indices < np.repeat(np.arange(G.shape[1]), G.indptr[1:] - G.indptr[:-1])] = 0
    G.sort_indices()
    G.eliminate_zeros()
    
    edges_mat = adjacency_to_edges_mat(G)
    cluster_to_edges = dot(cliques.T, edges_mat)
    cluster_to_edges.data = cluster_to_edges.data == 2
    cluster_to_edges.eliminate_zeros()
    
    cluster_sizes = as_dense_flat(cliques.sum(0))
    covers_to_edges = csc_matrix(cluster_to_edges)
    for i in range(covers_to_edges.shape[1]):
        cluster_idx = covers_to_edges.indices[covers_to_edges.indptr[i] : covers_to_edges.indptr[i+1]]
        cluster_data = covers_to_edges.data[covers_to_edges.indptr[i] : covers_to_edges.indptr[i+1]]
        assert cluster_idx.size > 0, i
        c = cluster_sizes[cluster_idx]
        cluster_data[c < c.max()] = 0
    covers_to_edges.eliminate_zeros()
    
    return (covers_to_edges.sum(1) > 0).nonzero()[0]

def get_largest_cliques(cliques):
    cliques_sizes = as_dense_flat(cliques.sum(0))
    max_clique = cliques_sizes.max()
    return cliques[:, cliques_sizes == max_clique], max_clique

def print_dense(G):
    print G.toarray()

def sparse_str(G):
    G_start, G_end, G_indices = G.indptr[:-1], G.indptr[1:], G.indices
    return [(v, list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(G.shape[0]) if G_end[v] > G_start[v]]
    
def sparse_str_indices(G):
    G_start, G_end, G_indices = G.indptr[:-1], G.indptr[1:], G.indices
    return [(v, G_start[v], G_end[v], list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(G.shape[0]) if G_end[v] > G_start[v]]
    
#def print_sparse(G):
#    print 'G:', sparse_str(G)
    
def print_density(G):
    print 'G density:', G.sum() / float(G.shape[0] * (G.shape[0] -1))    

def format_clique_sizes(X):
    return OrderedDict(sorted(Counter(as_dense_flat(X.sum(0)).astype(np.int32)).items(), key=lambda x:x[0]))
                

def check_cliques(G, cliques=None, order=None, method='igraph'):
    """
    cliques : The cliques found as a nodes-by-cliques matrix
    method : reference method
    """
    
    if method=='numba':
        start = time.time()
        PX = np.arange(k).astype(np.int32)
        ref_cliques, ref_cliques_indptr, ref_cliques_n, _ = BKPivotSparse2_Gnew_wrapper(Gold, Gnew, PX=PX)
        ref_cliques = [tuple(sorted(ref_cliques[ref_cliques_indptr[i]:ref_cliques_indptr[i+1]])) for i in range(ref_cliques_n)]
        print 'Time:', time.time() - start
    elif method=='igraph':
        import igraph
        graph = igraph.Graph(n=G.shape[0],
                             edges=list(set(zip(*G.nonzero()))),
                             directed=False)
        start = time.time()
        m = graph.maximal_cliques()
        print 'Time:', time.time() - start
        ref_cliques = sorted([tuple(sorted(np.array(x))) for x in m])
        
    ref_cliques = tuples_to_csc(ref_cliques, G.shape[0])
    ref_cliques, ref_max_clique = get_largest_cliques(ref_cliques)
    
    if order is not None:
        inv_order = np.empty(G.shape[0], np.int32)
        inv_order[order] = np.arange(order.size)

    if cliques is not None:
        _, max_clique = get_largest_cliques(cliques)
        print 'Found max: %s, ref max: %s' % (max_clique, ref_max_clique)        
        cliques_tuples = sorted(csc_to_cliques_list(cliques))
        ref_cliques_tuples = sorted(csc_to_cliques_list(ref_cliques))
        if cliques_tuples != ref_cliques_tuples:            
            print 'Cliques found but not in ref:', sorted(set(cliques_tuples) - set(ref_cliques_tuples))[:5]
            if order is not None:
                print 'Cliques in ref but not found (reordered indices):', [sorted([inv_order[a] for a in b]) for b in sorted(set(ref_cliques_tuples) - set(cliques_tuples))[:5]]
            else:
                print 'Cliques in ref but not found:', sorted(set(ref_cliques_tuples) - set(cliques_tuples))[:5]
            print 'Ref cliques:', ref_cliques_tuples
            
            assert max_clique == ref_max_clique
            raise Exception('Size of maximum clique agrees, but the actual cliques do not')
    else:
        print 'Ref max clique:', ref_max_clique
        
    return ref_cliques, ref_max_clique

def assert_clique(clique, G):
    clique = np.array(list(clique))
    sub = G[:, clique][clique, :]
    edges = sub.sum() / 2
    total = (clique.size * (clique.size-1)) / 2
    assert edges == total, 'Edges: %s, Possible: %s' % (edges, total)

def subsumption(X, XX=None):
    """
    Returns array sub where sub[i,j]==1 if X[:,i] is a subset of X[:,j], and 0 otherwise.
    """

    X_sizes = as_dense_flat(X.sum(0))
    if XX is None:
        XX = dot(X.T, X)
        
    if issparse(XX):
        # XX = as_dense_array(XX)

        assert isspmatrix_csr(XX)
        raise Exception('To finish')
#        XX.indices
            
        X_min_sizes = np.minimum(X_sizes.reshape(-1,1), X_sizes.reshape(1,-1))
        X_was_min_sizes = X_sizes.reshape(-1,1) >= X_sizes.reshape(1,-1)

        sub = XX == X_min_sizes
        sub[X_was_min_sizes] = 0
        return sub
    else:
        X_min_sizes = np.minimum(X_sizes.reshape(-1,1), X_sizes.reshape(1,-1))
        X_was_min_sizes = X_sizes.reshape(-1,1) >= X_sizes.reshape(1,-1)

        sub = XX == X_min_sizes
        sub[X_was_min_sizes] = 0
        return sub
    
def assert_unique_cliques(cliques):
    if isinstance(cliques, csc_matrix):
        tmp = csc_to_cliques_list(cliques)
    else:
        tmp = cliques
    assert len(tmp)==len(set(tmp))

    
