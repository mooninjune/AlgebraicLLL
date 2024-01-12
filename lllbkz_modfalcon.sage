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

def dot_product(v,w):
    """
    Implements dot product of two vectors over CyclotomicField that actually works (see https://trac.sagemath.org/ticket/34597 for the issue with the in-built one).
    param v: vector over CyclotomicField
    param u: vector over CyclotomicField
    Outputs v.dot_product(u), but computes it correctly.
    """
    return(sum(v[i]*w[i] for i in range(len(v))))

def bkz_red( B, bkz_beta=40 ):
    then = perf_counter()
    try:
        B = flatter_interface(B)
    except Exception as err:
        pass
    try:
        G = GSO.Mat(B, float_type="dd")
    except:
        G = GSO.Mat(B, float_type="dd")
    G.update_gso()
    lll = LLL.Reduction( G )
    lll()
    print( f"LLL done in: {perf_counter()-then}" )

    G = lll.M
    flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.GH_BND
    bkz = BKZReduction(G)
    for beta in range(4,bkz_beta+1,1):    #BKZ reduce the basis
        par = BKZ_FPYLLL.Param(beta,
                               max_loops=2,
                               flags=flags,
                               strategies=BKZ_FPYLLL.DEFAULT_STRATEGY
                               )
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        print( f"BKZ-{beta} done in {round_time} r00={bkz.M.r()[0]**0.5}" )

    return bkz.M

def in_lattice( B,v, use_custom_solver=False ):
    #checks if v in lattice defined by B
    try:
        if use_custom_solver:
            u = solve_left( B,v )
        else:
            u = B.solve_left( v )
    except ValueError:
        return False

    w = vector( [round(ww) for ww in u] )
    if not u == w:
        return False

    return True

def in_lattice_alg( B,v, use_custom_solver=False ):
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
    param B: matrix over QQ
    param v: vector over QQ
    """
    K = B[0,0].parent()
    d = K.degree()
    n, m = B.nrows(), B.ncols()

    u = B.solve_left( v )
    return u

def dsd_happened( B, Bfg ):
#     K = B[0,0].parent()
    subdet = det(Bfg*Bfg.transpose())**(1/2)
    B = list(B)
    m, threshold = len(B), Bfg.nrows()
    vctrs = []
    j = 0
    flag = False
    for i in range(m):
        b = B[i]
        if in_lattice( Bfg, b ):
            vctrs.append(b)
            j+=1
        if j>=threshold:
            flag = True
            break
    if flag:
        Cfg = matrix(vctrs)
        ovdet = det(Cfg*Cfg.transpose())**(1/2)
        # print( [ B.index(vv) for vv in vctrs ] )
        return ovdet / subdet
    return 0

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
            gb = g #balancemod( K,g,q )
            # for i in range(k):
            #     Bfg[i,0] = gb[i]
            # for i in range(1,k):
            #     for j in range(0,k):
            #         fi = F[i] #balancemod( K,F[i],q )
            #         Bfg[i,j+1] = -fi[j]
            for i in range(k):
                Bfg[i,0] = gb[i]
            for i in range(k):
                for j in range(1,k+1):
                    Bfg[i,j] = -F[i,j-1].lift()

            Bfg = embed_Q(Bfg)
            Gfg = GSO.Mat( IntegerMatrix.from_matrix( Bfg.change_ring(ZZ) ) )
            Gfg.update_gso()
            ldetBfg = sum( [ log(rr) for rr in Gfg.r() ] )/2
            # print(f"debug: {Bfg.nrows(), Bfg.ncols()}")

            gsnorm = ( q^(1/(1+k)) ).n(80)
            print(f"Expected gs_slack: {enorm_vector_over_numfield(Bfg[0]).n()^0.5 / gsnorm}")

            targetnorm = min( enorm_vector_over_numfield(b)^0.5 for b in Bfg )
            print(f"Target norm: {targetnorm}")

            B_red = IntegerMatrix.from_matrix( matrix(ZZ,embed_Q( B )) )

            i = 0
            for b in Bfg:
                assert in_lattice(matrix(B_red),vector(b)) , f"Dense sublattice constructed incorrectly. {i}"
                i+=1

            if d*(k+1)<=250 and q<2^22:
                G = GSO.Mat( B_red,float_type='ld' )
            elif d*(k+1)<=500:
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

            len_lll = RR( norm( vector(M.B[0]) ) ) #enorm_vector_over_numfield( vector(f1,g1) )^0.5
            # if dsd_happened( matrix(M.B).change_ring(ZZ), Bfg ):
            if sum( [log(rr) for rr in M.r()[:k*d]] )/2 <= 1.0001*ldetBfg:
                print(f"DSD event after lll: norm: {len_lll}")
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
                len_bkz = RR( bkz_obj.M.get_r(0,0)**0.5 ) #enorm_vector_over_numfield( vector(f1,g1) )^0.5
                flag_skr = any( [ vector(b) in Bfg for b in bkz_obj.M.B ] )
                print( sum( [log(rr) for rr in bkz_obj.M.r()[:k*d]] )/2 , 1.0001*ldetBfg )
                flag_dsd =  sum( [log(rr) for rr in bkz_obj.M.r()[:k*d]] )/2 <= 1.0001*ldetBfg #dsd_happened( matrix(bkz_obj.M.B).change_ring(ZZ), Bfg )
                if flag_dsd or flag_skr:
                    print( f"DSD: {flag_dsd}; SKR: {flag_skr}" )
                    bkz_t = perf_counter() - thenbkz
                    if flag_dsd:
                        print(f"DSD event: {len_bkz}")
                    else:
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

nthreads = 40
tests_per_q = 20 #
dump_public_key = False

# - - -
k=2
f=128  #the conductor of a number field
k=2    #rank of module - 1
qs = [ next_prime( ceil(2**tmp) ) for tmp in [12 + 0.2*i for i in range(6)] ] * tests_per_q
beta = 30

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
qs = [ next_prime( ceil(2**tmp) ) for tmp in [20.0+i for i in range(5)] ] * tests_per_q
beta = 55

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
