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

def rev_bit_order( l,m ):
    if m==0:
        return l
    k = l[0:len(l):2] + l[1:len(l):2]
    return( rev_bit_order(k,m-1) )

def ntru_experiment( f=512,q=next_prime(ceil(2^16.98)),beta=40,descend_number=0,seed=randrange(2^32), manual_descend=1, dump_public_key=False, first_block_beta=30, early_abort_niters=False ):
    """
    Constructs NTRU lattice, size and unit reduces it, descends to the subfied of index 2 and calls LLL.
    After that checks if the dense submodule is found.

    param f: CyclotomicField conductor.
    param q: modulus, must be prime
    param beta: block size for BKZ
    param descend_number: amount of descends in LLL
    param seed: seed
    """
    print(f"Launching ntru_experiment with seed={seed}")

    path = "ntruexp_folder/"
    isExist = os.path.exists(path)
    if not isExist:
        try:
            os.makedirs(path)
        except:
            pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.

    try:
        strq = str(q) if q<10**14 else str(q.n(40)) #filename overflow fix
        filename = path + f"NTRU_f{f}_q{strq}_b{beta}_desc{descend_number}_seed{seed}.txt"
        with open(filename, 'w') as file:
            with contextlib.redirect_stdout(file):
                K.<z> = CyclotomicField(f)
                d=K.degree()
                L.<t> = CyclotomicField(d)
                f_, g_, h_ = gen_ntru_instance( K,q,seed )

                if dump_public_key:
                    filename = f"H_f{f}_q{q}_seed{seed}.txt"
                    with open(filename, 'w') as file_dump:
                        file_dump.write( str(h_.list()) )

                B = matrix( K, [        #computing NTRU lattice
                    [K(q), 0],
                    [h_,  1]
                ])

                print( "Expected vector length: ", enorm_vector_over_numfield(vector((f_,g_))).n(40)^0.5 )
                print(f"q = {q}")

                n, m = B.nrows(), B.ncols()
                print( f"Size reduction over {K}..." )
                LLL = L2( B,LLL_params() )
                then = perf_counter()
                U = LLL.size_unit_reduce()  #size and unit reducing
                print(f"Size reduction done in: {perf_counter()-then}")
                #B = matrix([ [dot_product(U[i],B.column(j)) for j in range(m)] for i in range(n) ])

                print("Eucledian Norms:")
                print("      Before      |     After ")
                for i in range(n):
                    print( enorm_vector_over_numfield(B[i]).n(40)^0.5, '|', enorm_vector_over_numfield(LLL.B[i]).n(40)^0.5 )

                print("Descending...")

                # B = matrix(L,[      #descending to the subfield
                #     descend(L,B[0,0]) + descend(L,B[0,1]),
                #     descend(L,z*B[0,0]) + descend(L,z*B[0,1]),
                #     descend(L,B[1,0]) + descend(L,B[1,1]),
                #     descend(L,z*B[1,0]) + descend(L,z*B[1,1]),
                # ])
                B = descend_L( B,depth=manual_descend )
                L = B[0,0].parent()

                n, m = B.nrows(), B.ncols()
                strat = LLL_params.overstretched_NTRU(d,q,descend_number=descend_number, beta=beta, first_block_beta=first_block_beta, early_abort_niters=early_abort_niters)
                strat["rho"] = early_abort_niters
                strat["rho_sub"] = 12
                strat["debug"] = debug_flags.verbose_anomalies|debug_flags.dump_gcdbad
                LLL = L2( B,strat )     #LLL reducing

                print(f"Running full LLL... with descend num:", strat["descend_number"])

                then = perf_counter()
                U = LLL.lll( )
                total_time = perf_counter()-then
                print(f"All done in: {total_time}")

                print("Eucledian Norms:")
                print("      Before      |     After ")
                for i in range(n):
                    print( enorm_vector_over_numfield(B[i]).n(40)^0.5, '|', enorm_vector_over_numfield(LLL.B[i]).n(40)^0.5 )
                print()
                print("approx_fact")
                l = enorm_vector_over_numfield(vector((f_,g_)))^0.5
                minlen = min(enorm_vector_over_numfield(b)^0.5 for b in LLL.B)

                print((minlen/l).n(40))
                print(f"||q||/minlen = {q/minlen.n()}")

                #FG = ascend(K,B[0][:2]), ascend(K,B[0][2:])

                FG = LLL.B[0]
                depth = manual_descend
                FIs = [ K ]
                dd = d
                while dd > 2*L.degree():    #prepare tower field
                    M = CyclotomicField( dd )
                    dd/=2
                    FIs.append(M)

                FIs = [ i for i in reversed(FIs) ]

                for  i in range(len(FIs)):  #ascend the vector in the basis to the top
                    tmp = []
                    for j in range(len(FG)/2):
                        tmp.append( ascend( FIs[i], FG[2*j:2*j+2] )  )
                    FG = tmp
                    tmp = []


                D = FG[0] / f_, FG[1] / g_

                if D[0]!= D[1]:
                    print( "Short vector not in (f_,g_)*OK !" )
                    return q,False,total_time

                print(f"Sucsess!")
                return q,True, total_time
    except Exception as err:
        print(f"seed:{seed}, {err}")
        print(err)
    return None

def process_output( output ):
    i=0
    while i< len(output):
        if isinstance( output[i], Exception ):
            output.pop(i)
        i+=1
    if i>0:
        print(f"{i} experiments omitted due to the problems.")

    output = [ (x, a,b ) for x,a,b in sorted(output, key=lambda triple: (triple[0])) ]
    print( "q's and outcomes of the experiments:" )
    print(output)

    d = dict()
    for item in output:
        if not isinstance(item,(tuple,list)):
            continue
        if item[0] in d.keys():
            d[item[0]][0]+=1    #increase number
            if item[1]:
                d[item[0]][1]+=1
            d[item[0]][2]+=item[2] #add running time
        else:
            d[item[0]]= [1,0,0]
            if item[1]:
                d[item[0]][1]+=1
            d[item[0]][2]+=item[2] #add running time

    RR = RealField(40)
    print( "   q   | test_num | succ_num |  avg_time" )
    for q in d.keys():
        print( f"{q : ^8} {d[q][0] : ^10} {d[q][1] : ^10}" + str( RR(d[q][2]/d[q][0] ) ) )

    time.sleep( float(0.02) )  #give a time for the program to dump everything to the disc

nthreads = 20
tests_per_q = 20
dump_public_key = False

descend_number = 0
manual_descend = 1
early_abort_niters = False

# - - - NTRU 128

f=128
qs = [ next_prime( ceil(2^tmp) ) for tmp in [9.0,9.5,10.0] ] * tests_per_q
beta=20
first_block_beta = 25

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    ntru_experiment, (f,q,beta,descend_number,init_seed,manual_descend,dump_public_key, first_block_beta, early_abort_niters)
    ) )
    init_seed += 1

for t in tasks:
    output.append( t.get() )

pool.close() #closing processes in order to avoid crashing
process_output( output )

print("NTRU 128 Done")
# - - - NTRU 256

f=256
qs = [ next_prime( ceil(2^tmp) ) for tmp in [13.0+i*0.1 for i in range(6)] ] * tests_per_q
beta=40
first_block_beta = 46

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    ntru_experiment, (f,q,beta,descend_number,init_seed,manual_descend,dump_public_key, first_block_beta, early_abort_niters)
    ) )
    init_seed += 1

for t in tasks:
    output.append( t.get() )

pool.close() #closing processes in order to avoid crashing
process_output( output )
print("NTRU 256 Done")

# - - - NTRU 512

f=512
qs = [ next_prime( ceil(2^tmp) ) for tmp in [17.0 + 0.4*i for i in range(6)] ] * tests_per_q
beta= 40
first_block_beta = 50

output = []
pool = Pool(processes = nthreads )
tasks = []

i=0
init_seed = 0
print( f"Launching {len(qs)} experiments on {nthreads} threads." )
print( f"f={f}, qs={qs}, beta={beta}" )
for q in qs:
    tasks.append( pool.apply_async(
    ntru_experiment, (f,q,beta,descend_number,init_seed,manual_descend,dump_public_key, first_block_beta, early_abort_niters)
    ) )
    init_seed += 1

for t in tasks:
    output.append( t.get() )

pool.close() #closing processes in order to avoid crashing
process_output( output )
print("NTRU 512 Done")
