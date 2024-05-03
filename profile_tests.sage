"""
This file containsthe experiments regarding the alebraic profile of a lattice.
They show how far from the prediction the basis is.
"""

import sys
import os
from time import perf_counter
import time
from gen_lat import gen_LWE
import contextlib

from fpylll import*
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll import BKZ as BKZ_FPYLLL
from utils import enorm_vector_over_numfield, embed_Q
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from l2_ import *

import numpy as np
import pickle

def run_experiment_LWE( f=256,q=next_prime(ceil(2^16.98)),k=2, beta=4, seed=0 ):
    path = "profile_folder/"
    isExist = os.path.exists(path)
    try:
        os.makedirs(path)
    except:
        pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.

    B = gen_LWE( f,q,k,seed=seed )

    filename = f"profile_folder/PROFLWE_f{f}_k{k}_q{q}_b{beta}_seed{seed}.txt"
    print( f"Seed: {seed} launched.")

    try:
        with open(filename, 'w') as file:
            with contextlib.redirect_stdout(file):

              K = B[0,0].parent()
              d = K.degree()
              n = B.nrows()

              B_old = [ nf_vect( [minkowski_embedding(bij) for bij in B[i]] ) for i in range(n) ]
              G = GSO_obj(B_old)
              G.compute_GSO(start=0, end=n)
              alg_profile_old = [ log( G.Rr[i][i].alg_norm(),2 ).n(50)/2 for i in range(n) ]

              strat = LLL_params.LWE( f,q,descend_number=0,beta=beta )
              gamma = 1
              strat["gamma"] = gamma
              lll = L2( B,strat )

              print("Running full LLL...")

              then = perf_counter()
              try:
                  U = lll.lll( )
              except Exception as e:
                  print(f"Fail at seed {seed}: {e}")
                  print(e)
                  return None
              print(f"All done in: {perf_counter()-then}")

              print("Norms:")
              print("      Before      |     After ")
              for i in range(n):
                  print( enorm_vector_over_numfield(B[i]).n(40)^0.5, '|', enorm_vector_over_numfield(lll.B[i]).n(40)^0.5 )
              print()

              B_red = [ nf_vect( [minkowski_embedding(bij) for bij in lll.B[i]] ) for i in range(n) ]
              G = GSO_obj(B_red)
              G.compute_GSO(start=0, end=n)
              alg_profile = [ log( G.Rr[i][i].alg_norm(),2 ).n(50)/2 for i in range(n) ]
              log_alpha = ((d*(log(d,2)+1) + 2*d*log(gamma,2)))

              perfect_value = (n-1)/2 * log_alpha
              dr = [ alg_profile[i]-alg_profile[i+1] for i in range(n-1) ]
              log_slope = np.mean( dr )
              observed_value = log_slope #sum( (n-i-1)*dr[i] for i in range(n-1) ) / n

              print(f"Perfect: {perfect_value} vs ours: {observed_value}")

              print(f"Perfect: {log_alpha} vs ours: {log_slope}")

              print( f"foolchk: {sum(alg_profile_old)} | {sum(alg_profile)}" )

              return (q,k), perfect_value, observed_value, log_alpha, log_slope, vector(alg_profile_old), vector(alg_profile)
    except Exception as e:
        print( e )
        return None

def process_output( output, expnum, f ):
    d = {}
    for o in output:
        if o is None:
            continue
        if not o[0] in d.keys():
            d[o[0]] = list( o[1:] )
        else:
            d[o[0]][0] += o[1]
            d[o[0]][1] += o[2]
            d[o[0]][2] += o[3]
            d[o[0]][3] += o[4]
            d[o[0]][4] += o[5]
            d[o[0]][5] += o[6]
    for key in d.keys():
        d[key] = [ oo / expnum for oo in d[key] ]

    print("Note: perfect_value should be >= observed_value and log_alpha >= log_slope")
    print( "q  | perfect_value | observed_value | log_alpha | log_slope" )
    for key in d.keys():
        o = d[key]
        print( f"{key[0]}{o[0]} {o[1]} {o[2]} {o[3]}" )
        list_plot( o[4],title=f"{key}: old profile").save_image( f"f{f}_{key}_oldprofile.png" )
        list_plot( o[5],title=f"{key}: new profile").save_image( f"f{f}_{key}_newprofile.png" )
        with open( f"{f}_{key}_newprofile.pkl", "wb" ) as file:
            pickle.dump( o[5],file )

    time.sleep( float(0.1) )

if __name__ == "__main__":

    nthreads = 6
    tests_per_q = 5
    dump_public_key = False

    # - - - f=small

    f=32
    k=6
    qs = [ next_prime( ceil(2^tmp) ) for tmp in [12,14,16] ] * tests_per_q
    beta=35

    output = []
    pool = Pool(processes = nthreads )
    tasks = []

    i=0
    init_seed = 0
    print( f"Launching {len(qs)} experiments on {nthreads} threads." )
    print( f"f={f}, k={k} qs={qs}, beta={beta}" )
    for q in qs:
        tasks.append( pool.apply_async(
        run_experiment_LWE, (f,q,k, beta, init_seed)
        ) )
        init_seed += 1

    for t in tasks:
        output.append( t.get() )

    pool.close() #closing processes in order to avoid crashing
    process_output( output, tests_per_q, f )

    # --- f=64
    #
    # f=64
    # k=6
    # qs = [ next_prime( ceil(2^tmp) ) for tmp in [32,48,64] ] * tests_per_q
    # beta=35
    #
    # output = []
    # pool = Pool(processes = nthreads )
    # tasks = []
    #
    # i=0
    # init_seed = 0
    # print( f"Launching {len(qs)} experiments on {nthreads} threads." )
    # print( f"f={f}, k={k} qs={qs}, beta={beta}" )
    # for q in qs:
    #     tasks.append( pool.apply_async(
    #     run_experiment_LWE, (f,q,k, beta, init_seed)
    #     ) )
    #     init_seed += 1
    #
    # for t in tasks:
    #     output.append( t.get() )
    #
    # pool.close() #closing processes in order to avoid crashing
    # process_output( output, tests_per_q, f )


    # --- f=128

    # f=128
    # k=4
    # qs = [ next_prime( ceil(2^tmp) ) for tmp in [25,27.5,30] ] * tests_per_q
    # beta=35
    #
    # output = []
    # pool = Pool(processes = nthreads )
    # tasks = []
    #
    # i=0
    # init_seed = 0
    # print( f"Launching {len(qs)} experiments on {nthreads} threads." )
    # print( f"f={f}, k={k} qs={qs}, beta={beta}" )
    # for q in qs:
    #     tasks.append( pool.apply_async(
    #     run_experiment_LWE, (f,q,k, beta, init_seed)
    #     ) )
    #     init_seed += 1
    #
    # for t in tasks:
    #     output.append( t.get() )
    #
    # pool.close() #closing processes in order to avoid crashing
    # process_output( output, tests_per_q, f )
