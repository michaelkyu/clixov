import time
import numpy as np
import scipy.sparse
from scipy.sparse import isspmatrix, isspmatrix_csc, isspmatrix_csr, csc_matrix, csr_matrix, issparse, coo_matrix
from collections import OrderedDict, Counter

from mkl_spgemm import dot, elt_multiply

from numba import jit

from constants import cache
        
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

def adjacency_to_edges_mat(G, upper_right=True):
    assert isspmatrix_csc(G)

    if upper_right:
        G = G.copy()
        G.data[G.indices < np.repeat(np.arange(G.shape[1]), G.indptr[1:] - G.indptr[:-1])] = 0
        G.sort_indices()
        G.eliminate_zeros()
    
    indices = np.empty(2 * G.indices.size, np.int32)
    indices[::2] = G.indices
    indices[1::2] = np.repeat(np.arange(G.shape[1]), as_dense_flat(G.sum(0)).astype(indices.dtype))
    indptr = np.arange(0, 2 * G.indices.size + 2, 2)
    edges_mat = csc_matrix((np.ones(indices.size, G.dtype), indices, indptr),
                           (G.shape[0], G.indices.size))
    edges_mat.sort_indices()
    return edges_mat
    
def get_largest_clique_covers(cliques, G, ret_edges=False, assert_covered=True):
    # Check symmetry
    assert G.multiply(G.T).sum() == G.sum()
    
    edges_mat = adjacency_to_edges_mat(G)
    # print edges_mat.shape
    # print edges_mat.toarray().astype(np.int32)
    cluster_to_edges = dot(cliques.T, edges_mat)
    cluster_to_edges.data = cluster_to_edges.data == 2
    cluster_to_edges.eliminate_zeros()
    
    cluster_sizes = as_dense_flat(cliques.sum(0))
    covers_to_edges = csc_matrix(cluster_to_edges)
    max_covers = np.zeros(covers_to_edges.shape[1], np.int32)
    for i in range(covers_to_edges.shape[1]):
        cluster_idx = covers_to_edges.indices[covers_to_edges.indptr[i] : covers_to_edges.indptr[i+1]]
        cluster_data = covers_to_edges.data[covers_to_edges.indptr[i] : covers_to_edges.indptr[i+1]]
        if assert_covered:
            assert cluster_idx.size > 0, ('No cover for edge index %s:' % i)
        if cluster_idx.size > 0:
            c = cluster_sizes[cluster_idx]
            cluster_data[c < c.max()] = 0
            max_covers[i] = c.max()
        else:
            max_covers[i] = 2
    covers_to_edges.eliminate_zeros()

    if ret_edges:
        # Also return the edges indices and their max covers
        cover_G = coo_matrix((max_covers, (edges_mat.indices[::2], edges_mat.indices[1::2])), shape=G.shape)
        cover_G += cover_G.T
        cover_G = csc_matrix(cover_G)
        #cover_G[edges_mat.indices[::2], edges_mat.indices[1::2]] = max_covers
        return (covers_to_edges.sum(1) > 0).nonzero()[0], cover_G
    else:
        # Just return the indices of cliques to keep as max covers
        return (covers_to_edges.sum(1) > 0).nonzero()[0]

def get_largest_cliques(cliques):
    cliques_sizes = as_dense_flat(cliques.sum(0))
    max_clique = cliques_sizes.max()
    return cliques[:, cliques_sizes == max_clique], max_clique

def print_dense(G):
    print G.toarray()

def sparse_str_I(GI, GS, GE):
    return [(v, list(GI[GS[v] : GE[v]])) for v in np.arange(GS.size) if GE[v] > GS[v]]

def sparse_str(G):
    return sparse_str_I(G.indices, G.indptr[:-1], G.indptr[1:])
    # G_start, G_end, G_indices = 
    # return [(v, list(G_indices[G_start[v] : G_end[v]])) for v in np.arange(G.shape[0]) if G_end[v] > G_start[v]]
    
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
        ref_cliques, ref_cliques_indptr, ref_cliques_n, _ = BK_Gnew_py(Gold, Gnew, PX=PX)
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

def assert_cliques(cliques, G):
    if issparse(G):
        G = G.toarray()
    G = np.ascontiguousarray(G)
    
    for i in range(cliques.shape[1]):
        c = cliques[:,i].nonzero()[0]
        try:
            assert G[c,:][:,c].sum() == c.size * (c.size-1)
        except:
            print 'Not a clique:'
            print c.tolist()
            print G[c,:][:,c]
            raise        
        extra = as_dense_flat(G[c,:].sum(0)==c.size).nonzero()[0]
        try:
            assert extra.size == 0
        except:
            print 'Not a maximal clique:'
            print c.tolist(), 'extra:', extra
            raise
        
@jit(nopython=True, cache=cache)
def remove_diagonal_nb(data, indices, indptr):
    ## Assumes well formatted sparse matrix with sorted indices

    new_data = np.empty(data.size, data.dtype)
    new_indices = np.empty(indices.size, indices.dtype)
    new_indptr = indptr.copy()
    end = 0
    new_end = 0
    offset = 0
    
    n = indptr.size - 1
    for i in range(n):
        for j in range(indptr[i],indptr[i+1]):
            if i==indices[j]:
                new_data[new_end : new_end + j - end] = data[end : j]
                new_indices[new_end : new_end + j - end] = indices[end : j]
                new_end += j - end
                end = j + 1
                offset += 1
                break
        new_indptr[i+1] -= offset
            
    very_end = indptr[n]
    new_data[new_end : new_end + very_end - end] = data[end : very_end]
    new_indices[new_end : new_end + very_end  - end] = indices[end : very_end]
    new_end += very_end - end

    return new_data[:new_end], new_indices[:new_end], new_indptr

def remove_diagonal(X):
    assert isspmatrix_csr(X) or isspmatrix_csc(X)
    X.data, X.indices, X.indptr = remove_diagonal_nb(X.data, X.indices, X.indptr)
            
def fill_diagonal(X, val):
    assert X.shape[0]==X.shape[1]
    if isspmatrix(X):
        n = X.shape[0]
        X[np.arange(n), np.arange(n)] = 0        
        X.eliminate_zeros()
    else:
        np.fill_diagonal(X, val)

def subsumption(X, XX=None):
    """
    Returns array sub where sub[i,j]==1 if X[:,i] is a subset of X[:,j], and 0 otherwise.
    """

    X_sizes = get_clique_sizes(X)
    if XX is None:
        H = dot(X.T, X)
    else:
        H = XX.copy()
        
    if isspmatrix(H):
        assert isspmatrix_csr(H)
        H.data = (H.data == np.repeat(X_sizes, H.indptr[1:]-H.indptr[:-1]))
    else:
        H = H == X_sizes.reshape(1,-1)

    fill_diagonal(H, 0)
    return H
    
def update_subsumption(X, dX, H):
    X_sizes = get_clique_sizes(X)
    dH = dot(X.T, dX)
    bottom_shape = (dX.shape[1], X.shape[1]+dX.shape[1])
    if isspmatrix(dH):
        assert isspmatrix_csr(dH) and isspmatrix(H)        
        dH.data = (dH.data == np.repeat(X_sizes, dH.indptr[1:]-dH.indptr[:-1])).astype(H.dtype)
        dH.eliminate_zeros()
        # print type(dH)
        H = scipy.sparse.hstack([H, dH])
        #H = H.tocsr()
        
        # # Manually add in empty rows
        # assert isspmatrix_csr(H)
        # H.indptr = np.append(H.indptr, np.repeat(H.indptr[-1], dX.shape[1]))
        # H.shape = (H.shape[1], H.shape[1])
        
        H = scipy.sparse.vstack([H,
                                 scipy.sparse.csr_matrix(bottom_shape, dtype=H.dtype)])
        # print type(H)           
    else:
        dH = (dH == X_sizes.reshape(1,-1)).astype(H.dtype)
        H = np.vstack([np.hstack([H, dH]),
                       np.zeros(bottom_shape, H.dtype)])
    return H
    
def assert_unique_cliques(cliques):
    if isinstance(cliques, csc_matrix):
        tmp = csc_to_cliques_list(cliques)
    else:
        tmp = cliques
    assert len(tmp)==len(set(tmp))

def get_clique_sizes(X):
    return as_dense_flat(X.sum(0))

@jit(nopython=True)
def infer_children(parents):
    children = np.zeros(parents.size, parents.dtype)
    depths = np.zeros(parents.size, np.int32)
    depths[:2] = -1
#    print 'asdf'
    for v_i in range(parents.size):
        children[parents[v_i]] += 1
        depths[v_i] = depths[parents[v_i]] + 1
    return children, depths

def print_tree_indent(tree, width=5):
    parents = tree[0,:]
    branches = tree[1,:]
    returns = tree[2,:]
    children, depths = infer_children(parents)

    indent = '-'
    curr_line = ''
    for v_i in range(1, parents.size):
        parent = parents[v_i]
        branch = branches[v_i]
        depth = depths[v_i]
        ret = returns[v_i]

        if ret==0:
            block = ('%s *' % (branch)).ljust(width)
        else:
            block = ('%s' % (branch)).ljust(width)
        
        if parent == v_i - 1:
            curr_line += block
        else:
            print curr_line
            curr_line = ' ' * (width * (depth-2)) + block
    print curr_line
    
def get_unexplained_edges(X, G):
    """Return a gene-by-gene boolean matrix with 1 indicating that the
       gene pair is in a clique.

       X : gene-by-clique matrix
       G : gene-by-gene adjacency matrix
    """
    Y = dot(X, X.T)
    if issparse(Y):
        Y.data = (Y.data > 0).astype(X.dtype)
        # fill_diagonal(Y, 0)
        remove_diagonal(Y)
    else:
        Y = (Y > 0).astype(X.dtype)
        fill_diagonal(Y, 0)

    Y = G - elt_multiply(G, Y)
    return Y
