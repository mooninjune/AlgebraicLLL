import sys
from time import perf_counter
import time
from l2_ import *

from gen_lat import gen_ntru_instance
import contextlib

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

load("hawk.sage")

def descend_L( B, depth=1 ):
    """
    Descends B to the subfield of relative degree 2^depth.
    param B: matrix to descend.
    param depth: the depth of descend.
    """
    if depth==0:
        return B
    n, m = B.nrows(), B.ncols()
    K = B[0,0].parent()
    z = K.gen()
    d = K.degree()
    L.<t> = CyclotomicField( d )

    M = matrix( L, 2*n, 2*m )
    for i in range( n ):
        for k in range(2):
            b = z^k * B[i]
            for j in range( m ):
                M[2*i+k,2*j]   = L( b[j].list()[0:d:2] )
                M[2*i+k,2*j+1] = L( b[j].list()[1:d:2] )
    return descend_L( M,depth-1 )

def run_experiment(d=128, sigma_kg=1.5, sigma_sig=1, sigma_ver=1.1, seed=0):
    manual_descend = 1
    K.<z> = CyclotomicField( 2*d )
    Sig = SignatureScheme(d, sigma_kg, sigma_sig, sigma_ver)

    sk, pk = Sig.KGen()
    f, g, F, G = sk
    q00, q10, q11 = pk

    B = matrix( K, [
        [ q00, q10.conjugate() ],
        [ q10, q11           ]
    ] )
    n, m = B.nrows(), B.ncols()

    print( f"Size reduction over {K}..." )
    LLL = L2( B,LLL_params() )
    then = perf_counter()
    U = LLL.size_unit_reduce()  #size and unit reducing
    print(f"Size reduction done in: {perf_counter()-then}")

    print("Eucledian Norms:")
    print("      Before      |     After ")
    for i in range(n):
        print( enorm_vector_over_numfield(B[i]).n(40)^0.5, '|', enorm_vector_over_numfield(LLL.B[i]).n(40)^0.5 )

    print("Descending...")
    B = descend_L( B,depth=manual_descend )
    L = B[0,0].parent()
    n, m = B.nrows(), B.ncols()

    strat = LLL_params( bkz_beta=36, svp_oracle_threshold=512, first_block_beta=42 )
    strat["debug"] = debug_flags.verbose_anomalies#|debug_flags.dump_gcdbad
    global_variables.log_basis_degradation_factor = 6.0
    LLL = L2( B,strat )

    then = perf_counter()
    U = LLL.lll( )
    total_time = perf_counter()-then
    print(f"All done in: {total_time}")
    B_red = LLL.B

    answer = vector( list(f)+list(g) )
    ansnrm = norm( answer ).n()
    shvec = vector( list(B_red[0,0]) + list(B[0,1]) )
    shnrm = norm( shvec ).n()

    print( f"Answer: {answer}" )
    print( f"Found: {shvec}" )
    print( f"Their: {ansnrm} vs my: {shnrm}" )

run_experiment(d=128, sigma_kg=1.5, sigma_sig=1, sigma_ver=1.1, seed=0)
