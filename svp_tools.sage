import sys

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

FPLLL.set_precision(global_variables.fplllPrec)


def dsd_check( l, threshold=0.01 ):
    l = [ log(ll) for ll in l ]
    n = len(l)
    return sum(l[:n//2]) < (1-threshold)*sum(l[n//2:])

from time import perf_counter
import pickle

from time import perf_counter
import pickle

def pip_solver( a,b,seed=None ):  #if aOK+bOK is principal, finds g: gOK = aOK+bOK
    """
    Solves equation g*O_K = a*O_K + b*O_K. For pow-of-2 CyclotomicField of conductor <= 256.
    For 512-th field we invoke GentrySzydlo which often fails.
    param a: CyclotomicField element
    param b: CyclotomicField element
    """
    K = a.parent()
    d = K.degree()
    z = K.gen()
    print(d)

    g = []
    I = ideal(a,b)
    if d < 32: #if the field is small enough, we can solve PIP manually
        g = I.gens_reduced( proof=False )
    elif d in [32,64]:   #if we managed to compute bnfinit for a certain fields, we just solve it with sage
        filename = f"f{2*d}bnf.pkl"
        print(f"Reading {filename}")
        pari( r"\p 58" )
        with open( filename, "rb" ) as file_:
            K._pari_bnf = pickle.load( file_ )
        I._pari_bnf = K._pari_bnf
        then = perf_counter()
        try:
            alarm( int(100) ) #timeout for gens reduced
            g = I.gens_reduced( proof=False )
            cancel_alarm()  #stop timeout
        except AlarmInterrupt as e:
            raise RuntimeError( "PIP interrupted!" )
        except:
            cancel_alarm()  #stop timeout if something goes wrong
        print( f"PIP done in {perf_counter()-then}" )
    else:
        #else we go for a Gentry Szydlo
        print(f"Solving pip for f={2*d}")

        varname = str(K.gen())

        pari( f"n={2*d}" )
        pari( "g0="+str(a).replace(varname,"t") )   #prepeare input for pari gp
        pari( "g1="+str(b).replace(varname,"t") )
        pari( "seed=\"" + str(seed) + "\"")

        then = perf_counter()
        pari(r"\r GenRec.gp")
        print(f"PIP solved in {perf_counter()-then}")

        B = matrix( pari("BKZin") )

        B = bkz_reduce( B, block_size=24 )*B

        for j in range(d//2,-1,-1):
            res = B[0,0]  + sum( (z^i-z^(d-i))*B[0,i] for i in range(1,d/2) )
            md = K( pari("Mod(1+GSgen,PolC)") )

            tmp = res/md
            nrm = int( norm(res/md) / norm(I) )
            if (nrm != 0) and (nrm & (nrm-1) == 0):
                return tmp

        B = bkz_reduce( B, block_size=32 )*B
        for j in range(d//2,-1,-1):
            res = B[0,0]  + sum( (z^i-z^(d-i))*B[0,i] for i in range(1,d/2) )
            md = K( pari("Mod(1+GSgen,PolC)") )

            tmp = res/md
            nrm = int( norm(res/md) / norm(I) )
            if (nrm != 0) and (nrm & (nrm-1) == 0):
                return tmp

        return tmp


    assert len(g) == 1, f"Ideal is not principal or defined over too large field. PIP cannot be solved!"
    return g[0]


    assert len(g) == 1, f"Ideal is not principal or defined over too large field. PIP cannot be solved!"
    return g[0]



def bkz_reduce(B, block_size, verbose=False, task_id=None, sort=True, bkz_r00_abort=False, force_ld=False):
    """
    BKZ reduces the integral lattice defined by B.
    param B: matrix over ZZ with coefficients < 512 bit.
    param block_size: block size for BKZ.
    param verbose: flag showing if the information being verbosed.
    param task_id: task number. Useful in multithreading.
    param sort: if True the vectors are sorted according to their length so the U[0]*B is the shortest among the vectors in U*B.
    param bkz_DSD_trick: if True, does bkz untill the DSD event supposedly happens, then bkz continues of the first half or the matrix.
    param bkz_r00_abort: abort BKZ if r00 decreased by a factor of 2^log_bkz_abort_factor.
    Returns U, such that U*B is ~BKZ-block_size reduced.
    """
    #print(f"bkz_r00_abort: {bkz_r00_abort}")
    n, m = B.nrows(), B.ncols()

    B = IntegerMatrix.from_matrix(B)

    #BKZ
    RRf = RealField( 30 )

    if n>global_variables.mpfr_usage_threshold_dim or global_variables.bkz_scaling_factor>global_variables.mpfr_usage_threshold_prec or force_ld:
        GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='mpfr')
        print(f"Reducing lattice of dimension {n}. Using: mpfr")
    else:
        if global_variables.bkz_scaling_factor<=global_variables.ld_usage_threshold_prec and n<=global_variables.ld_usage_threshold_dim:
            GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='ld')
            print(f"Reducing lattice of dimension {n}. Using: ld")
        elif qd_avaliable:
            GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='qd')   #'qd'
            print(f"Reducing lattice of dimension {n}. Using: qd")
        else:
            GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='mpfr')
            print(f"Reducing lattice of dimension {n}. Using: mpfr")
    GSO_M.update_gso()

    old_log_r00 = log( GSO_M.get_r(0,0),2 )/2

    lll_red = LLL_FPYLLL.Reduction(GSO_M,delta=0.95, eta=0.53)
    if verbose:
        print(f"Launching LLL... initial log r00 = {RRf(old_log_r00)}")
        then = time.perf_counter()
    lll_red()

    if verbose:
        try:
            tmp = basis_quality(GSO_M)["/"]
        except:
            tmp = None
        print(f"lll done in {time.perf_counter()-then} slope: {tmp}, log r00: {RRf(log( lll_red.M.get_r(0,0),2 )/2)}")

    then=time.perf_counter()

    flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.GH_BND

    then=time.perf_counter()

    block_sizes = [i for i in range(4,int(min(n,m,block_size)),2)]
    block_sizes += [int(min(n,block_size))]

    bkz = BKZReduction(GSO_M)
    sys.stdout.flush()    #flush after the LLL

    dsd_happened_flag = False
    for beta in block_sizes:    #BKZ reduce the basis
        par = BKZ_FPYLLL.Param(beta,
                               max_loops=global_variables.bkz_max_loops,
                               flags=flags
                               )
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        if bkz_r00_abort:
            bkz.M.update_gso()
            new_log_r00 = log( bkz.M.get_r(0,0),2 )/2
            if new_log_r00+log_bkz_abort_factor < old_log_r00:
                print("r00 decreased. Aborting BKZ.")
                break
        if verbose:
            print('bkz for beta=',beta,' done in:', round_time, 'slope:', basis_quality(GSO_M)["/"], 'log r00:', RRf( log( bkz.M.get_r(0,0),2 )/2 ), 'task_id = ', task_id)
            sys.stdout.flush()  #flush after the BKZ call
    if verbose:
        gh = gaussian_heuristic([GSO_M.get_r(i,i) for i in range(n)])
        print('gh:', log(gh), 'true len:', RRf(log(GSO_M.get_r(0,0))))

    dt=time.perf_counter()-then

    print('All BKZ computed in',dt, 'sec')
    U =  matrix(ZZ, GSO_M.U)

    if dsd_happened_flag and bkz_DSD_trick:
        print("Doing SVP in the dense sublattice.")
        B_sub = matrix( ZZ, bkz.M.B[0:n//2] )
        U_sub = bkz_reduce(B_sub, block_size, verbose=verbose, task_id=None, sort=False)
        U_ = matrix.block(ZZ, [
            [ U_sub,             matrix.zero(n//2) ],
            [ matrix.zero(n//2), matrix.identity(n//2) ]
        ])
        U = U_*U

    if sort:
        U = matrix( [x for _, x in sorted(zip(GSO_M.B,U), key=lambda pair: norm(pair[0]))] )

    return(U)

def bkz_reduce_ntru(B, block_size, verbose=False, task_id=None, sort=True, bkz_r00_abort=False):
    """
    BKZ reduces the integral lattice defined by B.
    param B: matrix over ZZ with coefficients < 512 bit.
    param block_size: block size for BKZ.
    param verbose: flag showing if the information being verbosed.
    param task_id: task number. Useful in multithreading.
    param sort: if True the vectors are sorted according to their length so the U[0]*B is the shortest among the vectors in U*B.
    param bkz_DSD_trick: if True, does bkz untill the DSD event supposedly happens, then bkz continues of the first half or the matrix.
    param bkz_r00_abort: abort BKZ if r00 decreased by a factor of 2^log_bkz_abort_factor.
    Returns U, such that U*B is ~BKZ-block_size reduced.
    """
    #print(f"bkz_r00_abort: {bkz_r00_abort}")
    n, m = B.nrows(), B.ncols()

    B = IntegerMatrix.from_matrix(B)

    #BKZ

    if n>global_variables.mpfr_usage_threshold_dim or global_variables.bkz_scaling_factor>global_variables.mpfr_usage_threshold_prec:
        GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='mpfr')
        print(f"Reducing lattice of dimension {n}. Using: mpfr")
    else:
        if global_variables.bkz_scaling_factor<=global_variables.ld_usage_threshold_prec and n<=global_variables.ld_usage_threshold_dim:
            GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='ld')
            print(f"Reducing lattice of dimension {n}. Using: ld")
        elif qd_avaliable:
            GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='qd')   #'qd'
            print(f"Reducing lattice of dimension {n}. Using: qd")
        else:
            GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type='mpfr')
            print(f"Reducing lattice of dimension {n}. Using: mpfr")
    GSO_M.update_gso()

    old_log_r00 = log( GSO_M.get_r(0,0),2 )/2

    lll_red = LLL_FPYLLL.Reduction(GSO_M,delta=0.95, eta=0.53)
    if verbose:
        print(f"Launching LLL... initial log r00 = {RRf(old_log_r00)}")
        then = time.perf_counter()
    lll_red()

    if verbose:
        try:
            tmp = basis_quality(GSO_M)["/"]
        except:
            tmp = None
        print(f"lll done in {time.perf_counter()-then} slope: {tmp}, log r00: {log( lll_red.M.get_r(0,0),2 )/2}")

    then=time.perf_counter()

    flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.GH_BND

    then=time.perf_counter()

    block_sizes = [i for i in range(4,int(min(n,m,block_size)),2)]
    block_sizes += [int(min(n,block_size))]

    bkz = BKZReduction(GSO_M)
    sys.stdout.flush()    #flush after the LLL

    dsd_happened_flag = False
    dsd_counter = 0
    for beta in block_sizes:    #BKZ reduce the basis
        if dsd_counter>=4:
            break
        par = BKZ_FPYLLL.Param(beta,
                               max_loops=global_variables.bkz_max_loops,
                               flags=flags
                               )
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        if bkz_r00_abort:
            bkz.M.update_gso()
            new_log_r00 = log( bkz.M.get_r(0,0),2 )/2
            if new_log_r00+log_bkz_abort_factor < old_log_r00:
                print("r00 decreased. Aborting BKZ.")
                break

        profile = [ bkz.M.get_r(t,t) for t in range(n) ]
        if dsd_check( profile ) or basis_quality(GSO_M)["/"]>=-0.005:        #if DSD is supposedly happened, we starting counter to quit
            dsd_happened_flag = True
            print("DSD happened!")
            dsd_counter+=1
        if verbose:
            print('bkz for beta=',beta,' done in:', round_time, 'slope:', basis_quality(GSO_M)["/"], 'log r00:', log( bkz.M.get_r(0,0),2 )/2, 'task_id = ', task_id)
            sys.stdout.flush()  #flush after the BKZ call
    if verbose:
        gh = gaussian_heuristic([GSO_M.get_r(i,i) for i in range(n)])
        print('gh:', log(gh), 'true len:', log(GSO_M.get_r(0,0)))

    dt=time.perf_counter()-then

    print('All BKZ computed in',dt, 'sec')
    U =  matrix(ZZ, GSO_M.U)

    if dsd_happened_flag:
        print("Doing SVP in the dense sublattice.")
        B_sub = matrix( ZZ, bkz.M.B[0:n//2] )
        U_sub = bkz_reduce(B_sub, block_size, verbose=verbose, task_id=None, sort=False)
        U_ = matrix.block(ZZ, [
            [ U_sub,             matrix.zero(n//2) ],
            [ matrix.zero(n//2), matrix.identity(n//2) ]
        ])
        U = U_*U

    if sort:
        U = matrix( [x for _, x in sorted(zip(GSO_M.B,U), key=lambda pair: norm(pair[0]))] )

    return(U)
