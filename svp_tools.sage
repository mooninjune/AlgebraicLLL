import sys
from sys import stdout

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
import os
import subprocess, re, shutil

import numpy as np

FPLLL.set_precision(global_variables.fplllPrec)


def dsd_check( l, threshold=0.01 ):
    l = [ log(ll) for ll in l ]
    n = len(l)
    return sum(l[:n//2]) < (1-threshold)*sum(l[n//2:])

from time import perf_counter
import pickle

from time import perf_counter
import pickle

if g6k_avaliable:
    from pump import my_pump_n_jump_bkz_tour
    from g6k import Siever, SieverParams
    from g6k.algorithms.bkz import pump_n_jump_bkz_tour
    from g6k.utils.stats import dummy_tracer

def flatter_interface( fpylllB, do_timeout=True ):
    """
    If the flatter lib (https://github.com/keeganryan/flatter) is installed, outputs flatter-reduced basis of fpylllB.
    Else returns fpylllB.
    """
    flatter_is_installed = os.system( "flatter -h > /dev/null" ) == 0
    n = fpylllB.nrows

    if flatter_is_installed:
        fpylllB = IntegerMatrix.from_matrix( [ b for b in sorted(fpylllB, key= lambda t : norm(t)) ] )
        basis = '[' + fpylllB.__str__() + ']'
        seed = randrange(2**32)
        filename = f"lat{seed}.txt"
        # filename_out = f"redlat{seed}.txt"
        while os.path.exists(filename):
            filename = f"lat{seed}_{randrange(2**20)}.txt"
        # while os.path.exists(filename_out):
        #     filename = f"redlat{seed}_{randrange(1024)}.txt"

        with open(filename, 'w') as file:
            file.write( "["+fpylllB.__str__()+"]" )

        # out = os.system( "flatter " + filename + " > " + filename_out )
        command = ["flatter", filename ]
        try:
            # Run the command and capture its output
            alarm( int(2*n) ) #flatter can freeze
            out = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            cancel_alarm()
            # Process the output as needed
        except subprocess.CalledProcessError as e:
            # Handle any errors, e.g., print the error message
            os.remove( filename )
            print(f"Error: {e.returncode} - {e.output}")
            return fpylllB
        except AlarmInterrupt as e:
            print( "flatter interrupted!" )
            return fpylllB

        elements = out.split()
        # Initialize an empty string to store the modified output
        output_string = ""

        # Iterate through the elements
        for element in elements:
            if element.endswith(']'):
                output_string += element + ","
            else:
                output_string += element + " "

        time.sleep(float(0.05))
        os.remove( filename )
        output_string = re.sub(r'(\d)\s', r'\1, ', out)
        output_string = re.sub(r']\s', '],', output_string)[:-1]
        out = eval( output_string )
        B = IntegerMatrix.from_matrix( out )
        return B
    return fpylllB

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
    print(a, ", ", b)
    stdout.flush()
    if d < 32: #if the field is small enough, we can solve PIP manually
        g = I.gens_reduced( proof=False )
    elif d in [32]:   # 64 if we managed to compute bnfinit for a certain fields, we just solve it with sage
        filename = f"f{2*d}bnf.pkl"
        print(f"Reading {filename}")
        pari( r"\p 100" )
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
    Uflatter = matrix.identity(B.nrows)

    T = GSO.Mat(B, float_type="ld")
    T.update_gso()
    print(f"Invoking flatter... r00={(log(T.get_r(0,0),2)/2).n()}")
    then = time.perf_counter()
    BB = flatter_interface(B)
    print(f"flatter done in {time.perf_counter()-then}")
    then =time.perf_counter()
    # Uflatter = matrix( [matrix(RR,B).solve_left(b) for b in matrix(RR,BB)] ).change_ring(ZZ) #we need Uflatter

    Uflatter = matrix(RR,B).solve_left( matrix(RR,BB) )
    Uflatter= matrix( ZZ, [
        [ round(Uflatter[i,j]) for j in range(n) ] for i in range(n)
    ] )

    # this is faster, but crashes
    # B0, B1 = np.matrix(matrix(B).transpose()), np.matrix(matrix(BB).transpose())
    # Uflatter = np.linalg.lstsq( B0, B1, rcond=None )[0]
    # Uflatter= matrix( ZZ, [
    #     [ round(Uflatter[i,j]) for j in range(n) ] for i in range(n)
    # ] ).transpose()
    # Uflatter = matrix(ZZ,Uflatter)

    T = GSO.Mat(B, float_type="dd")
    T.update_gso()
    # Uflatter = matrix( ZZ, [ [ round(t) for t in T.babai(b) ] for b in BB ] )
    Uflatter = IntegerMatrix.from_matrix( Uflatter )
    print(f"flatter Uflatter done in {time.perf_counter()-then}")
    B = BB

    #BKZ
    RRf = RealField( 30 )

    if n>global_variables.mpfr_usage_threshold_dim or global_variables.bkz_scaling_factor>global_variables.mpfr_usage_threshold_prec or force_ld:
        GSO_M = GSO.Mat(B, U=Uflatter, float_type='mpfr')
        print(f"Reducing lattice of dimension {n}. Using: mpfr")
    else:
        if global_variables.bkz_scaling_factor<=global_variables.ld_usage_threshold_prec and n<=global_variables.ld_usage_threshold_dim:
            GSO_M = GSO.Mat(B, U=Uflatter, float_type='ld')
            print(f"Reducing lattice of dimension {n}. Using: ld")
        elif qd_avaliable:
            GSO_M = GSO.Mat(B, U=Uflatter, float_type='dd')   #dd is avaliable iff qd is avaliable
            print(f"Reducing lattice of dimension {n}. Using: dd")
        else:
            GSO_M = GSO.Mat(B, U=Uflatter, float_type='mpfr')
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

    # block_sizes = [i for i in range(4,int(min(n,m,block_size)),2)]
    # block_sizes += [int(min(n,block_size))]
    block_sizes = [ i for i in range(4,min(n,m,block_size)+1) ]

    bkz = BKZReduction(GSO_M)
    sys.stdout.flush()    #flush after the LLL

    dsd_happened_flag = False
    for beta in block_sizes:    #BKZ reduce the basis
        par = BKZ_FPYLLL.Param(beta,
                               max_loops=global_variables.bkz_max_loops,
                               flags=flags,
                               strategies=BKZ_FPYLLL.DEFAULT_STRATEGY
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
        float_type='mpfr'
        print(f"Reducing lattice of dimension {n}. Using: mpfr")
    else:
        if global_variables.bkz_scaling_factor<=global_variables.ld_usage_threshold_prec and n<=global_variables.ld_usage_threshold_dim:
            float_type='ld'
            print(f"Reducing lattice of dimension {n}. Using: ld")
        elif qd_avaliable:
            float_type='qd'
            print(f"Reducing lattice of dimension {n}. Using: qd")
        else:
            float_type='mpfr'
            print(f"Reducing lattice of dimension {n}. Using: mpfr")

    GSO_M = GSO.Mat(B, U=IntegerMatrix.identity(B.nrows), float_type=float_type)
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
                               strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
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

def g6k_reduce(B, block_size, verbose=True, task_id=None, sort=True):
    B = IntegerMatrix.from_matrix(B)
    n = B.nrows

    if n>global_variables.mpfr_usage_threshold_dim or global_variables.bkz_scaling_factor>global_variables.mpfr_usage_threshold_prec:
        float_type='mpfr'
        print(f"Reducing lattice of dimension {n}. Using: mpfr")
    else:
        if global_variables.bkz_scaling_factor<=global_variables.ld_usage_threshold_prec and n<=global_variables.ld_usage_threshold_dim:
            float_type='ld'
            print(f"Reducing lattice of dimension {n}. Using: ld")
        elif qd_avaliable:
            float_type='qd'
            print(f"Reducing lattice of dimension {n}. Using: qd")
        else:
            float_type='mpfr'
            print(f"Reducing lattice of dimension {n}. Using: mpfr")

    M = GSO.Mat(B, float_type=float_type,
                    U=IntegerMatrix.identity(B.nrows, int_type=B.int_type),
                    UinvT=IntegerMatrix.identity(B.nrows, int_type=B.int_type))
    M.update_gso()

    param_sieve = SieverParams()
    param_sieve['threads'] = global_variables.sieve_threads
    param_sieve['default_sieve'] = global_variables.sieve_for_bkz #"bgj1"
    g6k = Siever(M, param_sieve)
    then = time.perf_counter()

    for blocksize in range( 60,block_size+1 ):
        for t in range(global_variables.bkz_max_loops):
                    then_round=time.perf_counter()
                    my_pump_n_jump_bkz_tour(g6k, dummy_tracer, blocksize, jump=1,
										 filename="devnull", seed=1,
										 dim4free_fun="default_dim4free_fun",
										 extra_dim4free=0,
										 pump_params={'down_sieve': False},
										 verbose=verbose)
                    round_time = time.perf_counter()-then_round
                    if verbose:
                        print('tour ', t, ' bkz for beta=',blocksize,' done in:', round_time, 'slope:', basis_quality(M)["/"], 'log r00:', float( log( g6k.M.get_r(0,0),2 )/2 ), 'task_id = ', task_id)
                        sys.stdout.flush()  #flush after the BKZ call
    if verbose:
        gh = gaussian_heuristic([M.get_r(i,i) for i in range(n)])
        print('gh:', log(gh), 'true len:', (log(M.get_r(0,0))))

    dt=time.perf_counter()-then

    print('All BKZ computed in',dt, 'sec')
    U =  matrix(ZZ, M.U)

    if sort:
        U = matrix( [x for _, x in sorted(zip(M.B,U), key=lambda pair: norm(pair[0]))] )

    return(U)
