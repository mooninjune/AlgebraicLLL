import sys
import os
from time import perf_counter
import time
from gen_lat import keygen_modfalcon, balancemod
import contextlib

from fpylll import*
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll import BKZ as BKZ_FPYLLL
from utils import enorm_vector_over_numfield, embed_Q
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import numpy as np

def in_lattice( G, t ):
    w = vector( G.B.multiply_left( G.babai( t ) ) )
    if vector(t)-w == 0:
        return True
    return False

def run_experiment( f=256,q=next_prime(ceil(2^16.98)),k=2, beta=4, seed=0 ):
    filename = f"lllbkz_folder/LLLBKZ_MODFALC_f{f}_k{k}_q{q}_b{beta}_seed{seed}.txt"
    print( f"Seed: {seed} launched.")
    with open(filename, 'w') as file:
        with contextlib.redirect_stdout(file):
            flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.GH_BND  #|BKZ_FPYLLL.VERBOSE
            K.<z> = CyclotomicField(f)
            d=K.degree()

            # stddev = max( ( q^(1/(k+1)) / sqrt( d*(k+2) ) ).n() , 0.3 )
            # print( f"stddev: {stddev}" )

            B, F, g, h = keygen_modfalcon(K,q,k,seed,sigmaf=0.66, sigmag=0.66)
            Bfg = matrix( K,k,k+1 )  #unfinished matrix
            gb = balancemod( K,g,q )
            for i in range(k):
                Bfg[i,0] = gb[i]
            for i in range(1,k):
                for j in range(0,k):
                    fi = balancemod( K,F[i],q )
                    Bfg[i,j+1] = -fi[j]
            Bfg = embed_Q(Bfg)
            Bfg = IntegerMatrix.from_matrix( Bfg.change_ring( ZZ ) )
            Gfg = GSO.Mat( Bfg, float_type="ld" )
            Gfg.update_gso()

            gsnorm = ( q^(1/(1+k)) ).n(80)
            print(f"Expected gs_slack: {enorm_vector_over_numfield(Bfg[0]).n()^0.5 / gsnorm}")

            targetnorm = min( enorm_vector_over_numfield(b)^0.5 for b in Bfg )
            print(f"Target norm: {targetnorm}")

            B_red = IntegerMatrix.from_matrix( matrix(ZZ,embed_Q( B )) )
            if d*(k+1)<=400 and q<2^22:
                G = GSO.Mat( B_red,float_type='ld' )
            elif d*(k+1)<=768:
                FPLLL.set_precision(144)
                G = GSO.Mat( B_red,float_type='mpfr' )
            else:
                FPLLL.set_precision(250)
                G = GSO.Mat( B_red,float_type='mpfr' )
            G.update_gso()
            print(f"Initial r00: {G.get_r(0,0)**0.5}")
            sys.stdout.flush()

            then = perf_counter()
            lll_obj = LLL.Reduction(G,delta=0.98, eta=0.51)
            lll_obj()
            lll_t = perf_counter()-then
            len_lll = lll_obj.M.get_r(0,0)**0.5
            print(f"LLL done in {lll_t} r_00={len_lll}")
            sys.stdout.flush()

            M = lll_obj.M
            f1, g1 =  K( list(M.B[0])[:d] ), K( list(M.B[0])[d:] )

            len_lll = enorm_vector_over_numfield( vector(f1,g1) )^0.5
            if len_lll < gsnorm:
                print(f"SKR event after lll: norm: {len_lll}")
                print( f"len_lll/targetnorm, len_lll/gsnorm = {len_lll/targetnorm, len_lll/gsnorm}" )
                return q, 2, len_lll/targetnorm, None, len_lll/gsnorm, None, True, lll_t, 0

            bkz_obj = BKZReduction(G)
            thenbkz = perf_counter()
            DSD_BKZ = None
            for beta_counter in range(4,beta+1,1):
                then = perf_counter()
                par = BKZ_FPYLLL.Param(beta_counter,
                                           max_loops=10,
                                           flags=flags,
                                           strategies=BKZ_FPYLLL.DEFAULT_STRATEGY
                                           )
                bkz_obj(par)
                print(f"BKZ done in {perf_counter()-then}; beta={beta_counter}, r_00={bkz_obj.M.get_r(0,0)**0.5}")
                M = bkz_obj.M
                f1, g1 =  K( list(M.B[0])[:d] ), K( list(M.B[0])[d:] )
                len_bkz = bkz_obj.M.get_r(0,0)**0.5 #enorm_vector_over_numfield( vector(f1,g1) )^0.5
                flag = any( [ in_lattice(Gfg,b) for b in M.B] )
                if len_bkz < gsnorm or flag:
                    print( f"flag: {flag}" )
                    bkz_t = perf_counter() - thenbkz
                    print(f"SKR event: {len_bkz}")
                    print( f"len_bkz/targetnorm, len_bkz/gsnorm = {len_bkz/targetnorm, len_bkz/gsnorm}" )
                    return q, beta_counter, len_lll/targetnorm, len_bkz/targetnorm, len_lll/gsnorm, len_bkz/gsnorm, True, lll_t, bkz_t
                sys.stdout.flush()

            bkz_t = perf_counter() - thenbkz
            print( f"All done in {bkz_t}" )

            len_bkz = bkz_obj.M.get_r(0,0)^0.5

            print(f"SKR event: {len_bkz}")
            print( f"len_bkz/targetnorm, len_bkz/gsnorm = {len_bkz/targetnorm, len_bkz/gsnorm}" )
            return q, beta, len_lll/targetnorm, len_bkz/targetnorm, len_lll/gsnorm, len_bkz/gsnorm, False, lll_t, bkz_t
    return None

path = "lllbkz_folder/"
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)

nthreads = 10
tests_per_q = 10 #
dump_public_key = False

# - - - k=2
f=128  #the conductor of a number field
k=2    #rank of module - 1
qs = [ next_prime( ceil(2**tmp) ) for tmp in [13] ] * tests_per_q
beta = 47

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, k={k} qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    run_experiment, (f,q,k, beta, init_seed)
    ) )
    init_seed += 1

for t in tasks:
    output.append( t.get() )

pool.close() #closing processes in order to avoid crashing

print(output)

print( "q | beta | len_lll/targetnorm | len_bkz/targetnorm | len_lll/gsnorm | len_bkz/gsnorm | succ_num | walltime |" )
d = {}
for o in output:
    if o is None:
        continue
    if not o[0] in d.keys():
        d[o[0]] = [ [ o[1] ], [ o[2] ], [ o[3] ] if o[3] is not None else [], [ o[4] ], [ o[5] ] if o[5] is not None else [], 0, [ o[7] ], [ o[8] ] ]
        if o[6]:
            d[o[0]][5] += 1
    else:
        d[o[0]][0] += [ o[1] ]
        d[o[0]][1] += [ o[2] ]
        if o[3] is not None:
            d[o[0]][2] += [ o[3] ]
        d[o[0]][3] += [ o[4] ]
        if o[5] is not None:
            d[o[0]][4] += [ o[5] ]
        if o[6]:
            d[o[0]][5] += 1

for k in d.keys():
    print( f"{k} {np.mean(d[k][0]): .4f}     {np.mean(d[k][1]): .4f}      {np.mean(d[k][2]): .4f}      {np.mean(d[k][3]): .4f}      {np.mean(d[k][4]): .4f}      {d[k][5]}      {np.mean(d[k][6])+np.mean(d[k][7]): .4f}" )

# - - - k=3 - - -

f=128  #the conductor of a number field
k=3    #rank of module - 1
qs = [ next_prime( ceil(2**tmp) ) for tmp in [20] ] * tests_per_q
beta = 47

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, k={k} qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    run_experiment, (f,q,k, beta, init_seed)
    ) )
    init_seed += 1

for t in tasks:
    output.append( t.get() )

pool.close() #closing processes in order to avoid crashing

print(output)

print( "q | beta | len_lll/targetnorm | len_bkz/targetnorm | len_lll/gsnorm | len_bkz/gsnorm | succ_num | walltime |" )
d = {}
for o in output:
    if o is None:
        continue
    if not o[0] in d.keys():
        d[o[0]] = [ [ o[1] ], [ o[2] ], [ o[3] ] if o[3] is not None else [], [ o[4] ], [ o[5] ] if o[5] is not None else [], 0, [ o[7] ], [ o[8] ] ]
        if o[6]:
            d[o[0]][5] += 1
    else:
        d[o[0]][0] += [ o[1] ]
        d[o[0]][1] += [ o[2] ]
        if o[3] is not None:
            d[o[0]][2] += [ o[3] ]
        d[o[0]][3] += [ o[4] ]
        if o[5] is not None:
            d[o[0]][4] += [ o[5] ]
        if o[6]:
            d[o[0]][5] += 1

for k in d.keys():
    print( f"{k} {np.mean(d[k][0]): .4f}     {np.mean(d[k][1]): .4f}      {np.mean(d[k][2]): .4f}      {np.mean(d[k][3]): .4f}      {np.mean(d[k][4]): .4f}      {d[k][5]}      {np.mean(d[k][6])+np.mean(d[k][7]): .4f}" )
