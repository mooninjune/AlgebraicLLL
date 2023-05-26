"""
This file tests how accurate the BKZ as an algebraic approxSVP oracle is.
"""

from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
FPLLL.set_precision(80)
import time
from time import perf_counter
from copy import deepcopy
import pickle

from arakelov import arakelov_rand_walk

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool


from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

def numfield_elem_to_matrix(a):
    """
    Represents a*OK as a ZZ module.
    """
    K = a.parent().fraction_field()
    z = K.gen()
    d = K.degree()
    rows = []
    z.list()
    for i in range(d):
        rows.append( (a*z^i).list() )
    return matrix(QQ, [
        rows[i] for i in range(d)
    ])

def enorm_numfield(a):
    #returnd squared euclidean norm of a numfield element after coefficient embedding
    tmp = a.list()
    return sum(abs(t)^2 for t in tmp)

def short_lattice_vectors(B, nr_solutions=1):
    """
    Finds <=nr_solutions short vectors of lattice defined by B.
    """
    n, m = B.nrows(), B.ncols()
    B = B.LLL(pr="ld")

    Mint = IntegerMatrix(n, m)

    denom = B.denominator()
    for i in range(n):
        for j in range(m):
            Mint[i,j] = int( B[i,j]*denom )

    #BKZ
    GSO_M = GSO.Mat(Mint, float_type='ld')
    GSO_M.update_gso()
    then=time.perf_counter()


    flags = BKZ.AUTO_ABORT|BKZ.GH_BND|BKZ.MAX_LOOPS
    then=time.perf_counter()
    for beta in range(4,B.nrows(),2):
        #print(f"beta={beta}")
        par = BKZ.Param(block_size=beta, flags=flags, max_loops=25)
        bkz = BKZ2(GSO_M)
        DONE = bkz(par)
    par = BKZ.Param(block_size=B.nrows(), flags=flags, max_loops=25)
    bkz = BKZ2(GSO_M)

    dt=time.perf_counter()-then
    print('BKZ done in',dt, 'sec')

    R = GSO_M.get_r(0, 0)*( 1.01 if n>35 else 1.44)

    #pruning = Pruning.run(R, unknown_parameter, GSO_M.r(), gauss_factor, float_type="mpfr")
    then = time.perf_counter()
    enum = Enumeration(GSO_M, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, nr_solutions=int( nr_solutions ) )

    #res = enum.enumerate( 0, n, R, 0 ,pruning=pruning.coefficients  )
    res = enum.enumerate( 0, n, R, 0   )
    print(f"Enumeration done in {time.perf_counter()-then}, {len(res)} vectors found.")
    return res, matrix(QQ,Mint)/denom


def test_ideal(g, seed=0):
    I = ideal(g)
    K=I.number_field()
    d = K.degree()

    is_principal = False
    while not is_principal:
        set_random_seed( seed )
        s = K( [D() for i in range(d)] )
        I = ideal(K,s)
        # I._pari_bnf = K._pari_bnf
        then = perf_counter()
        I = arakelov_rand_walk( I, B=8*10**3, s=0.85, N=2, normalize=False  )
        print(f"Arakelov done in {perf_counter()-then}")
        #I *= I.denominator()
        nrm = norm( I )
        #print( f"ideal norm: {nrm}~{nrm.n(40)}" )
        then = perf_counter()
        g = I.gens_reduced( proof=False )
        is_principal = len(g)==1
        #is_principal = ( I.is_principal( proof=False ) )
        print(f"PIP done in {perf_counter()-then}; result:{is_principal}, seed = {seed} ")
    g = g[0]
    M = [
      list(tmp) for tmp in I.basis()
    ]
    try:
        V, B = short_lattice_vectors(matrix( M ),d+1)

        B = matrix(B)

        s = vector( ZZ, min( V, key=lambda x:x[0] )[1] )
        s = s*B
        s = K(s.list())
        #print(s)

        J = ideal( s )

        flag = (J==I)
        #print(flag,', N(I)=', norm(I), 'N(J)=', norm(J))
        #print( f"s in I: {s in I}")
        print(f"Sucsess: {flag}, {norm(s)/norm(g)}")
        print()

        return flag, enorm_numfield(g)^0.5
    except Exception as err:
        print( err )
        return None

test_num= 20
nthreads = 20

# --- f32

f = 32
K.<z> = CyclotomicField( f )
print( K )
d = K.degree()
D = DiscreteGaussianDistributionIntegerSampler( sigma = 100 )

if d in [32,64]:
    filename = f"f{2*d}bnf.pkl"
    print(f"Reading {filename}")
    with open( filename, "rb" ) as file_:
        K._pari_bnf = pickle.load( file_ )

output = []
pool = Pool( processes = nthreads )
tasks = []

res = []
init_seed = 0
for t in range(test_num):
    set_random_seed( init_seed+t )
    g = K( [D() for i in range(d)] )
    tasks.append( pool.apply_async( test_ideal, (g,t) ) )
    #res.append( test_ideal( I ) )

i = 0
for t in tasks:
    print(f"Processing experiment No. {i}")
    output.append( t.get() )
    i+=1

print(output)
pool.close()

o=[]
for oo in output:
    if not oo is None:
        o.append(oo)

print(f"Sucsess percentage for f={f}: {( sum( 1 if oo[0] else 0 for oo in o ) / len(o) ).n()}")

# --- f32

f = 64
K.<z> = CyclotomicField( f )
print( K )
d = K.degree()
D = DiscreteGaussianDistributionIntegerSampler( sigma = 100 )

if d in [32,64]:
    filename = f"f{2*d}bnf.pkl"
    print(f"Reading {filename}")
    with open( filename, "rb" ) as file_:
        K._pari_bnf = pickle.load( file_ )

output = []
pool = Pool( processes = nthreads )
tasks = []

res = []
init_seed = 0
for t in range(test_num):
    set_random_seed( init_seed+t )
    g = K( [D() for i in range(d)] )
    tasks.append( pool.apply_async( test_ideal, (g,t) ) )
    #res.append( test_ideal( I ) )

i = 0
for t in tasks:
    print(f"Processing experiment No. {i}")
    output.append( t.get() )
    i+=1

print(output)
pool.close()

o=[]
for oo in output:
    if not oo is None:
        o.append(oo)

print(f"Sucsess percentage for f={f}: {( sum( 1 if oo[0] else 0 for oo in o ) / len(o) ).n()}")
