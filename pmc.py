import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import time
import os

from clixov_utils import as_dense_flat, csc_to_cliques_list, assert_clique

lib = ctypes.cdll.LoadLibrary("/cellar/users/mikeyu/clixov/pmc/libpmc.so")

fun = lib.max_clique
#call C function
fun.restype = np.int32
fun.argtypes = [ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),ctypes.c_int32,
              ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]

def pmc(G, threads=1, verbose=False): #ei, ej is edge list whose index starts from 0
    ei, ej = G.nonzero()
    nnodes = G.shape[0]
    nnedges = ei.size

#    degrees = as_dense_flat(G.sum(0))
    # if maxd is None:
    #     maxd = np.int32(G.sum(0).max())
    tmp = ej < ei
    new_ei, new_ej = ei[tmp], ej[tmp]

    os.environ['OMP_THREAD_LIMIT']=str(threads)
        
#    maxd = int(max(degrees))
    offset = 0
    new_ei = np.array(new_ei,dtype = np.int32)
    new_ej = np.array(new_ej,dtype = np.int32)
#    outsize = maxd + 1
    outsize = nnodes
    output = np.zeros(outsize,dtype = np.int32)
    
    start = time.time()
    clique_size = fun(len(new_ei),new_ei,new_ej,offset,outsize,output)
    if verbose:
        print 'PMC time:', time.time() - start
    max_clique = np.empty(clique_size,dtype = np.int32)
    max_clique[:]=[output[i] for i in range(clique_size)]

    return max_clique

def check_pmc(G, cliques, check_members=True, verbose=False):
    found = csc_to_cliques_list(cliques)
    for c in found:
        assert_clique(c, G)
    pmc_clique = pmc(G.copy(), verbose=True)
    if verbose:
        print 'PMC:', pmc_clique.size, np.sort(pmc_clique)

    assert_clique(pmc_clique, G)
    assert pmc_clique.size == len(found[0]), 'PMC clique size: %s, Found clique size: %s' % (pmc_clique.size, len(found[0]))
    if check_members:
        assert tuple(sorted(pmc_clique)) in found

    return pmc_clique
