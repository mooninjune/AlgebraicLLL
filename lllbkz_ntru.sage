import sys
import os
from time import perf_counter
import time
from gen_lat import gen_ntru_instance
import contextlib

from fpylll import*
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll import BKZ as BKZ_FPYLLL
from utils import enorm_vector_over_numfield, embed_Q
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from svp_tools import flatter_interface
import numpy as np

def projection(basis, projectLeft):
    """
    On input of a basis matrix B = (B_1||B_2) and projectLeft = True,
    this function projects B_1 orthogonally against the all one vector.
    Then scales the matrix, such that it is integral.
    If projectLeft = False, then the projection is applied to B_2.
    """
    d = int( len(basis[0]) /2)

    if not projectLeft:
        basis = swapLeftRight(basis)

    for i, v in enumerate(basis):
        v_left = v[:d]
        v_right = v[d:]

    sum_left = sum(v_left)

    for j in range(d):
        v_left[j] = d*v_left[j] - sum_left
        v_right[j] = d*v_right[j]

    basis[i] = v_left + v_right

    if not projectLeft:
        basis = swapLeftRight(basis)

    return basis

"""
  Inverts the projection map.
"""
def projected(v):
    v = np.array( list(v) )
    d = int( len(v)/2 )

    ones_left = np.array( [1]*d + [0]*d )
    ones_right = np.array( [0]*d + [1]*d )
    all_zero = np.array( 2*d*[0] )

    foundPreImage = False
    i = 0

    while not foundPreImage and i < d:
        j = 0
        while not foundPreImage and j < d:
            candidate = v + i*ones_left + j*ones_right

            if np.array_equal(candidate % d, all_zero):
                foundPreImage = True
                v = (candidate/d).astype(int).tolist()
            else:
                j += 1

            i += 1
    return v

def dsd_check( l ):
    l = [ log(ll) for ll in l ]
    n = len(l)
    return sum(l[:n//2]) < sum(l[n//2:])

def run_experiment( f=256,q=next_prime(ceil(2^16.98)),beta=4,seed=randrange(2^32) ):
    """
    returns q, gamma_lll, gamma_bkz, lll_time, bkz_time, bool lll_dsd, int bkz_dsd
    """
    path = "lllbkz_folder/"
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)

    filename = f"lllbkz_folder/LLLBKZ_f{f}_q{q}_b{beta}_seed{seed}.txt"
    print( f"Seed: {seed} launched.")
    try:
        with open(filename, 'w') as file:
            with contextlib.redirect_stdout(file):
                flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.VERBOSE
                K.<z> = CyclotomicField(f)
                d=K.degree()
                f_, g_, h_ = gen_ntru_instance( K,q,seed )
                exp_len = enorm_vector_over_numfield( vector([f_,g_]) )
                B = matrix( K, [        #computing NTRU lattice
                    [K(q), 0],
                    [h_,  1]
                ])

                then = perf_counter()
                B_red = IntegerMatrix.from_matrix( matrix(ZZ,[b.list() for b in embed_Q( B )]) )
                B_red = flatter_interface( B_red )
                flattime = perf_counter()-then
                print(f"flatter done in {flattime} ")
                tmp = B_red #projection(B_red,True)
                B_red = matrix( ZZ,tmp )
                B_red = IntegerMatrix.from_matrix( B_red )
                if d<=100:
                    G = GSO.Mat( B_red,float_type='ld' )
                else:
                    FPLLL.set_precision(144)
                    G = GSO.Mat( B_red,float_type='mpfr' )

                then = perf_counter()
                lll_obj = LLL.Reduction(G,delta=0.98, eta=0.51)
                lll_obj()
                lll_t = perf_counter()-then
                len_lll = lll_obj.M.get_r(0,0)
                print(f"LLL done in {lll_t} r_00^2={len_lll}")
                sys.stdout.flush()

                M = lll_obj.M
                w = vector( projected( matrix(G.B)[0] ) )
                f1, g1 =  K( list(w)[:d] ), K( list(w)[d:] )

                multiple = False
                profile = False
                if f1/f_-g1/g_ == 0:
                    print( f"DSD event over field K={K}, q={q}, after LLL (f,g) is a multiple" )
                    multiple = True
                if dsd_check([lll_obj.M.get_r(i,i) for i in range(2*d)]):
                    print( f"DSD event over field K={K}, q={q}, after LLL profile check" )
                    profile = True
                if multiple or profile:
                    return q, len_lll/exp_len, len_lll/exp_len, lll_t+flattime, 0, True, 2, multiple+2*profile

                Mproj = matrix( ZZ,lll_obj.M.B  )
                Mproj = Mproj[ :2*d-Mproj.nullity() ]  #resolve linear dependency

                B_red = IntegerMatrix.from_matrix( Mproj )
                G = GSO.Mat( B_red,float_type='dd' )
                G.update_gso()

                bkz_obj = BKZReduction(G)
                thenbkz = perf_counter()
                DSD_BKZ = None
                for beta_counter in range(4,beta+1,2):
                    then = perf_counter()
                    par = BKZ_FPYLLL.Param(beta_counter,
                                               max_loops=14,
                                               strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
                                               flags=flags
                                               )
                    bkz_obj(par)
                    print(f"BKZ done in {perf_counter()-then}; beta={beta_counter}, r_00^2={bkz_obj.M.get_r(0,0)}")
                    sys.stdout.flush()

                    w = vector( projected( matrix(G.B)[0] ) )
                    f1, g1 =  K( list(w)[:d] ), K( list(w)[d:] )

                    if f1/f_-g1/g_ == 0:
                        print( f"DSD event over field K={K}, q={q}, after BKZ_{beta_counter}" )
                        print("- - - MULTIPLE - - -")
                        DSD_BKZ = beta_counter
                        multiple=True

                    if dsd_check([bkz_obj.M.get_r(j,j) for j in range(2*d)]):
                        print( f"DSD event over field K={K}, q={q}, after BKZ_{beta_counter}" )
                        print("- - - PROFILE DSD - - -")
                        DSD_BKZ = beta_counter
                        profile=True
                    if profile or multiple:
                        break

                bkz_t = perf_counter() - thenbkz
                len_bkz = bkz_obj.M.get_r(0,0)

                print( f"|f,g|={exp_len}, len_bkz={len_bkz}, len_lll={len_lll}" )
                print( f"len_lll/|f,g|={len_lll/exp_len}, len_bkz/|f,g|={len_bkz/exp_len}, lll_time={lll_t}, bkz_time={bkz_t}" )
                return q, len_lll/exp_len, len_bkz/exp_len, lll_t, bkz_t, False, DSD_BKZ, multiple+2*profile
    except util.ReductionError as err:
        print(err)
        return None

def process_output( output ):
    output = [ (x, a,b,c,d,e,f,g ) for x, a,b,c,d,e,f,g  in sorted(output, key=lambda tpl: (tpl[0])) ]
    print( "q's and outcomes of the experiments:" )
    print(output)

    d = dict()
    for item in output:
        if item is None:
            continue
        if item[0] in d.keys():
            d[item[0]][0]+=1    #increase number
            d[item[0]][1]+=item[1]  #add lll factor
            d[item[0]][2]+=item[2]  #add bkz factor
            d[item[0]][3]+=item[3] #add running time
            d[item[0]][4]+=item[4] #add running time
            d[item[0]][5]+= 1 if item[5] else 0
            d[item[0]][6]+= item[6] if not( item[6] is None ) else 0  #add bkz_beta
            d[item[0]][7]+= 1 if not( item[6] is None ) else 0        #if bkz sucsessful, increase counter
            d[item[0]][8]+= 1 if item[7] in [1,2] else 0              #if dsd happened only in one check, we increase the counter
        else:
            d[item[0]] = [1,0,0,0,0,0,0,0,0]  #exp number, gamma_lll, gamma_bkz, lll_t, lll_time, bkz_time, lll_dsd times, bkz_dsd beta, amount of sucsesses, amount of multiple!=dsd
            d[item[0]][1]+=item[1]  #add lll factor
            d[item[0]][2]+=item[2]  #add bkz factor
            d[item[0]][3]+=item[3] #add running time
            d[item[0]][4]+=item[4] #add running time
            d[item[0]][5]+= 1 if item[5] else 0  #dsd lll
            d[item[0]][6]+= item[6] if not( item[6] is None ) else 0  #add bkz_beta  #dsd bkz
            d[item[0]][7]+= 1 if not( item[6] is None ) else 0
            d[item[0]][8]+= 1 if item[7] in [1,2] else 0

    print(d)

    RR = RealField(20)
    print( "   q   |test_num| avg-γ-lll |  avg-γ-bkz   | avg_time_lll | avg_time_bkz | dsd_lll_times | dsd_bkz_avg | dsd!=multiple" )
    for q in d.keys():
        print( f"{q : ^8}   {d[q][0] : ^5} {RR(d[q][1])^0.5/max(1,d[q][0]) : ^5}    {RR(d[q][2])^0.5/max(1,d[q][0]) : ^8}   {RR(d[q][3]/max(1,d[q][0])) : ^8}  {RR(d[q][4]/max(1,d[q][0])) : ^10} {d[q][5]} {RR(d[q][6]/max(1,d[q][7]))} {d[q][8]}  " )

    time.sleep( float(0.02) )  #give a time for the program to dump everything to the disc

# - - - f=128 - - -

nthreads = 15
tests_per_q = 20
dump_public_key = False

f=128
qs = [ next_prime( ceil(2^tmp) ) for tmp in [9.0+i*0.5 for i in range(3)] ] * tests_per_q
beta=40

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    run_experiment, (f,q,beta,init_seed)
    ) )
    init_seed += 1

for t in tasks:
    o = t.get()
    if not o is None:
        output.append( o )

pool.close() #closing processes in order to avoid crashing

print( process_output(output) )

# - - - f=256 - - -

nthreads = 15
tests_per_q = 20
dump_public_key = False

f=256
qs = [ next_prime( ceil(2^tmp) ) for tmp in [13.0+i*0.1 for i in range(6)] ] * tests_per_q
beta=40

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    run_experiment, (f,q,beta,init_seed)
    ) )
    init_seed += 1

for t in tasks:
    o = t.get()
    if not o is None:
        output.append( o )

pool.close() #closing processes in order to avoid crashing

print( process_output(output) )

# - - - f=512 - - -

nthreads = 15
tests_per_q = 20
dump_public_key = False

f=512
qs = [ next_prime( ceil(2^tmp) ) for tmp in [17.0+i*0.1 for i in range(6)] ] * tests_per_q
beta=50

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    run_experiment, (f,q,beta,init_seed)
    ) )
    init_seed += 1

for t in tasks:
    o = t.get()
    if not o is None:
        output.append( o )

pool.close() #closing processes in order to avoid crashing

print( process_output(output) )
