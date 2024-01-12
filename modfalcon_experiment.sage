import os, time
from fpylll import *
from time import perf_counter
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

from fpylll import *
from fpylll import BKZ as BKZ_FPYLLL
from fpylll import LLL as LLL_FPYLLL
from fpylll import GSO, IntegerMatrix, FPLLL, Enumeration, EnumerationError, EvaluatorStrategy
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
from itertools import chain
import time
from time import perf_counter
from common_params import *
from copy import deepcopy

from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
from l2_ import *
from  numpy import mean
from time import perf_counter
from gen_lat import keygen_modfalcon, balancemod

from gen_lat import gen_ntru_instance
import contextlib

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle, io

def flatter_interface( fpylllB ):
    #import os
    basis = '[' + fpylllB.__str__() + ']'
    #print(basis)
    seed = randrange(2**32)
    filename = f"lat{seed}.txt"
    filename_out = f"redlat{seed}.txt"
    with open(filename, 'w') as file:
        file.write( "["+fpylllB.__str__()+"]" )

    out = os.system( "flatter " + filename + " > " + filename_out )
    time.sleep(float(0.05))
    os.remove( filename )

    B = IntegerMatrix.from_file( filename_out )
    os.remove( filename_out )
    return B

def svp_attempt(B, tar_nrm, bkz_beta=50):
    nrm = RR( norm( vector(B[0]) ) )
    if nrm <= tar_nrm:
        return True, vector(B[0])

    B = flatter_interface( B )
    try:
        G = GSO.Mat( B, float_type="dd" )
    except:
        G = GSO.Mat( B, float_type="mpfr" )
    G.update_gso()

    then = perf_counter()
    lll = LLL.Reduction(G)
    lll()
    print( f"LLL done in {perf_counter()-then} r00 = {G.r()[0]**0.5}" )
    if norm( vector(lll.M.B[0]) ) <= tar_nrm:
        return True, vector(lll.M.B[0])

    G = lll.M
    flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.GH_BND
    bkz = BKZReduction(G)
    for beta in range(4,bkz_beta+1,1):    #BKZ reduce the basis
        par = BKZ_FPYLLL.Param(beta,
                               max_loops=8,
                               flags=flags,
                               strategies=BKZ_FPYLLL.DEFAULT_STRATEGY
                               )
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        print( f"BKZ-{beta} done in {round_time} r00={bkz.M.r()[0]**0.5}" )
        if norm( vector(lll.M.B[0]) ) <= tar_nrm:
            print( f"Sucsess!" )
            return True, RR( norm( vector(bkz.M.B[0]) ) )
    return False, RR( norm( vector(bkz.M.B[0]) ) )

def in_lattice( B,v, use_custom_solver=False ):
    #checks if v in lattice defined by B
    try:
        if use_custom_solver:
            u = solve_left( B,v )
        else:
            u = B.solve_left( v )
    except ValueError:
        return False

    for uu in u:
        for uuu in uu:
            if not uuu in ZZ:
                return False
    return True

def solve_left( B, v ):
    """
    Replaces malfunctioning B.solve_left(v).
    param B: matrix over pow-of-2 CyclotomicField
    param v: vector over pow-of-2 CyclotomicField
    """
    K = B[0,0].parent()
    d = K.degree()
    n, m = B.nrows(), B.ncols()

    BB = embed_Q( B, scale_for_bkz=False )
    vv = []
    for v_ in v:
        vv += v_.list()
    vv = vector( vv )

    uu = BB.solve_left( vv )
    u = vector( [ K(uu[i*d:(i+1)*d].list()) for i in range(len(uu)/d) ] )
    return u

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


def run_experiment( conductor,q,k,manual_descend=0, beta=35, seed=0 ):
    path = "modfalcon_folder/"
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
    strq = str(q) if q<10**14 else str(q.n(40)) #filename overflow fix
    filename = path + f"MODFALC_f{conductor}_q{strq}_k{k}_desc{manual_descend}_seed{seed}.txt"
    try:
        with open(filename, 'w') as file:
            with contextlib.redirect_stdout(file):
                K.<z> = CyclotomicField( conductor )
                d = K.degree()

                stddev = min( ( q^(1/(k+1)) / sqrt( d*(k+2) ) ).n() , 0.5 )
                print( f"stddev: {stddev}" )

                B, F, g, h = keygen_modfalcon(K,q,k,seed, sigmaf=stddev, sigmag=stddev)

                Bfg = matrix( K,k,k+1 )  #unfinished matrix
                gb = g #balancemod( K,g,q )
                for i in range(k):
                    Bfg[i,0] = gb[i]
                for i in range(k):
                    for j in range(1,k+1):
                        Bfg[i,j] = -F[i,j-1].lift()

                # - - - check - - -
                Rq = Integers(q)[x].quotient_ring(K.polynomial())

                h_ = vector( Rq(hh) for hh in h )
                Finv = matrix([[Rq(F[i,j]) for j in range(k)] for i in range(k)])^-1
                g_ = vector([ Rq(gg) for gg in g] )
                check = Finv*g_ - h_
                assert all( c==0 for c in check ), "h is incorrect!"


                for uv in B:
                    u = uv[0]
                    v = uv[1:]
                    tmp = u + v.dot_product( h )
                    assert Rq(tmp) == 0, f"Not an NTRU lattice"

                print( "Checking that Bfg is sublattice of B..." )
                for bfg in Bfg:
                    # print( solve_left( B, bfg ) )
                    # print( in_lattice( B, bfg, use_custom_solver=True ) )
                    assert in_lattice( B, bfg, use_custom_solver=False ) , f"Dense sublattice constructed incorrectly."
                print()

                # - - - END check - - -

                targetnorm = min( enorm_vector_over_numfield(b)^0.5 for b in Bfg )
                print(f"Target norm: {targetnorm}")
                gsnorm = ( q^(1/(1+k)) ).n(80)
                print(f"Expected hsvp_factor: {enorm_vector_over_numfield(Bfg[0]).n()^0.5 / gsnorm}")

                if manual_descend:
                    B = descend_L( B,manual_descend )

                lllpar = LLL_params.overstretched_NTRU( 2*d,q,descend_number=0, beta=beta )
                lllpar["debug"] = debug_flags.verbose_anomalies #|debug_flags.dump_gcdbad
                LLL_params.set_precision(Prec_user=100, fplllPrec_user=100, bkz_scaling_factor_user=32)
                global_variables.log_basis_degradation_factor = 144.0

                alglll = L2(B, strategy=lllpar)
                then = perf_counter()
                alglll.lll()
                t = perf_counter() - then
                print(f"LLL done in: {t}")

                print("Eucledean norm before:")
                for b in B:
                    print( enorm_vector_over_numfield(b)^0.5 )

                print("Eucledean norm after:")
                minnrm = Infinity
                for b in alglll.B:
                    curnorm = enorm_vector_over_numfield(b)^0.5
                    if curnorm < minnrm:
                        minnrm = curnorm
                    print( curnorm )
                approx_fact = minnrm / targetnorm
                print(f"approximation factor: {approx_fact}")
                hsvp_factor = minnrm / ( q^(1/(k+1)) ).n()

                flag_old = any( in_lattice( Bfg, b, use_custom_solver=True ) for b in alglll.B )
                print(f"Short vector in BFg: {flag_old}")

                key_recovered = False
                tar_nrm = None
                B = alglll.B
                threshold_dim = 2
                threshold_dim = min( threshold_dim, k+1 )
                flag_dim2, flag = False , False
                flag = any( [ vector(b) in Bfg ] )
                for bfg in Bfg:
                    flag_dim2 |= in_lattice( B[:threshold_dim], bfg )
                    if flag_dim2:
                        tar_nrm = enorm_vector_over_numfield(bfg)**0.5
                        break

                print(f"Answer in B[:{threshold_dim}]: {flag_dim2}")
                """
                if flag_dim2:
                    print( f"Launching partial key recovery attack for rank-{threshold_dim}..." )
                    then = perf_counter()
                    key_recovered, minnrm = svp_attempt(IntegerMatrix.from_matrix(embed_Q(B[:threshold_dim], scale_for_bkz=False).change_ring(ZZ)), tar_nrm)
                    dt = perf_counter() - then
                    approx_fact = minnrm / targetnorm
                    print( f"key_recovered for rank-{threshold_dim}: {key_recovered} in {dt}" )
                    t+=dt
                """

                if not flag_dim2 and k>1: #if the answer not in dim 2 sublattice and overall dim is > 3
                    threshold_dim = min( 3, k )
                    for bfg in Bfg:
                        flag |= in_lattice( B[:threshold_dim], bfg )
                        if flag:
                            tar_nrm = enorm_vector_over_numfield(bfg)**0.5
                            break


                    print(f"Answer in B[:{threshold_dim}]: {flag}")
                    """
                    if flag:
                        print( f"Launching partial key recovery attack for rank-{threshold_dim}..." )
                        then = perf_counter()
                        key_recovered, minnrm  = svp_attempt(IntegerMatrix.from_matrix(embed_Q(B[:threshold_dim], scale_for_bkz=False).change_ring(ZZ)), tar_nrm)
                        dt = perf_counter() - then
                        approx_fact = norm( minnrm ) / targetnorm
                        print( f"key_recovered for rank-{threshold_dim}: {key_recovered} in {dt}" )
                    """

    except Exception as err:
        print(f"seed:{seed}, {err}")
        raise(err)
        return None
    print( f"seed {seed} done" )
    return( (q,approx_fact,hsvp_factor,t,flag) )

def process_output( output ):
    d = {}
    for o in output:
        if o is None:
            continue
        if not o[0] in d.keys():
            d[o[0]] = [ [ o[1] ], [ o[2] ], [ o[3] ], 0 ]
            if o[4]:
                d[o[0]][3] += 1
        else:
            d[o[0]][0] += [ o[1] ]
            d[o[0]][1] += [ o[2] ]
            d[o[0]][2] += [ o[3] ]
            if o[4]:
                d[o[0]][3] += 1

    print(d)

    print( f"q  |  approx  |  hsvp_factor  |  time | in sublat" )
    for key in d.keys():
        print( f"{key}: {mean(d[key][0])}, {mean(d[key][1])}, {mean(d[key][2])}, {d[key][3]} " )


nthreads = 3
tests_per_q = 2
"""
k=2
qs = [ next_prime(round(2**p)) for p in [12.0+0.2*i for i in range(6)] ] * tests_per_q
f=128
manual_descend = 0
beta=30

output = []
pool = Pool(processes = nthreads )
tasks = []

seed = 0

print( f"f={f}, qs={qs}, k={k}" )
for q in qs:
    tasks.append( pool.apply_async(
    run_experiment, (f,q,k,manual_descend,beta,seed)
    ) )
    seed += 1

for t in tasks:
    output.append( t.get() )

pool.close() #closing processes in order to avoid crashing
process_output( output )

sys.stdout.flush()
"""
# - - - k=3 - - -

k=3
qs = [ next_prime(round(2**p)) for p in [20.0+i for i in range(2)] ] * tests_per_q
f=128
k=3
beta=25
manual_descend = 0

output = []
pool = Pool(processes = nthreads )
tasks = []

seed = 0

print( f"f={f}, qs={qs}, k={k}" )
for q in qs:
    tasks.append( pool.apply_async(
    run_experiment, (f,q,k,manual_descend,beta,seed)
    ) )
    seed += 1

for t in tasks:
    output.append( t.get() )

pool.close() #closing processes in order to avoid crashing
process_output( output )
