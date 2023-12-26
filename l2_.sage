"""
Main file. Contains class L2, which implements LLL algorithm.
Also contains function generate_FieldInfos that precomputes the Log-unit
lattices (LUL) for various fields.
To precompute LUL for fields of conductor up to 2**11 run generate_FieldInfos( 11 ).
"""

from keflll_wrapper import ascend, descend, BezTransform
from time import perf_counter
from utils import minkowski_embedding, inv_minkowski_embedding, svp_coeff, canonical_embedding_for_fpylll, embed_Q, enorm_vector_over_numfield, enorm_numfield
from LLL_params import LLL_params
from svp_tools import pip_solver
from copy import deepcopy
import warnings
import pickle
from fpylll import GSO, IntegerMatrix
import numpy as np

from datetime import datetime

from sys import stdout
from common_params import *
from util_l2 import *

def dot_product(v,w):
    """
    Implements dot product of two vectors over CyclotomicField that actually works (see https://trac.sagemath.org/ticket/34597 for the issue with the in-built one).
    param v: vector over CyclotomicField
    param u: vector over CyclotomicField
    Outputs v.dot_product(u), but computes it correctly.
    """
    return(sum(v[i]*w[i] for i in range(len(v))))

def lll_fft(
        B, K=None, FIs=None, rho=9, rho_sub=9, gamma=0.98, gamma_sub=0.5, bkz_beta=8, svp_oracle_threshold = 32, use_coeff_embedding=False,
        debug=0, verbose=True, early_abort_niters=False, early_abort_r00_decrease=False, dump_intermediate_basis=False, bkz_r00_abort=False,
        use_custom_idealaddtoone=True, first_block_beta=0, use_pip_solver=False, min_runs_amount = 0, experiment_name=None):
    """
    Perform algebraic LLL reduction à la KEF-LLL on a free algebraic module defined by the row-vectors of B.
    param B: basis of a lattice over CyclotomicField (must consist of nf_vect) in the fft domain
    param FIs: set of FieldInfo objects corresponding to the fields in the field tower. If None, those will be computed and later reused.
    param rho: amount of Lovàsz condition checks (possible amount of approxSVP oracle calls)
    param rho_sub: amount of Lovàsz condition checks in the recursive calls of lll_fft.
    param gamma: expected approx factor of SVP oracle.
    param gamma_sub: expected approx factor of SVP oracle in recursive lll_fft call.
    param bkz_beta: block size of BKZ algorithm used in SVP oracle.
    param svp_oracle_threshold: maxinal degree of a field we're allowed to call BKZ on.
    param use_coeff_embedding: if True, use coefficient embedding for the oracle, else - use canonical.
    param debug: computes internal info.
    param verbose: verboses internal info.
    param early_abort_niters: stop reducing after rho iterations
    param early_abort_r00_decrease: exits LLL as soon as norm(r_i_i) decreased.
    param dump_intermediate_basis: dumps current basis after each tour
    param bkz_r00_abort: aborts bkz as soon as r_0_0 decreased.
    param use_custom_idealaddtoone: use our implenentation of pari gp's idealaddtoone.
    param first_block_beta: first approxSVP call will use specified block size.
    param use_pip_solver: use Principal Ideal Problem solver.
    param experiment_name: part of a name of the file intermediate basis will be dumped to
    """
    assert gamma>0, "gamma must be > 0"
    if debug:
        print("DEBUG ENABLED")
    print(f"Using PIP solver: {use_pip_solver}")
    n,m = len(B), len(B[0])

    assert not( K is None and FIs is None ), f"{bcolors.FAIL}Provide Field, of a FieldInfo object!"

    z = K.gen()
    f = 2*K.degree()
    assert (f & (f-1) == 0) and f != 0, f"{bcolors.FAIL}Only cyclotomic of pow 2 supported!"
    d = K.degree()
    if dump_intermediate_basis:
        if experiment_name is None:
            experiment_name = randrange(2**32)
        filename = str( experiment_name ) + f"{n}_{m}_{d}" + ".pkl"
    """
    Since K is cyclotomic pow-of-2 number field, disc(K) = 2^(d log(d)).
    gamma is approximation factor of approxSVP oracle.
    In [2019-1305] alpha_K = gamma^(2d)*2^d*dics(K), so log(alpha_K) =  2d*log(gamma) + (d (log(d)+1))
    As an input we are given gamma that we expect to be achieved by our oracle. Note that since
    the formula is loose, we call with gamma below 1 due to the fact that this is
    approx factor w.r.t. the determinant of the lattice.
    """
    #alpha_K = 2^( (d*(log(d,2)+1) + 2*d*log(gamma,2)) )
    log_alpha_K_sq = 2*((d*(log(d,2)+1) + 2*d*log(gamma,2)))

    lfi = max(FIs.keys())+1
    FI = FIs[lfi-1]

    # create the identity matrix
    zero_elem = vector(CC, [ 0 for ii in range(d/2) ])
    U = [ nf_vect([zero_elem]*n) for ii in range(n) ]
    one = minkowski_embedding(K(1))
    for ii in range(n):
        U[ii].v[ii]=one

    #compute Gram matrix of the basis
    G = GSO_obj(B)

    if verbose:
        G.compute_GSO(start=0, end=n)
        old_alg_profile = [ log( G.Rr[i][i].alg_norm(),2 ).n(50) for i in range(n) ]
        old_eucl_profile = [ B[i].canon_norm().n(50) for i in range(n) ]
    B_star = [None for i in range(n)]

    zfft = [ nf_elem( minkowski_embedding(z^ii) ) for ii in range(d) ]  # powers of z in fft domain
    iota,num_idle_Lovacz=0,0
    tour_num = 0
    tour_timer = perf_counter()

    good_blocks = [ True for i in range(n-1) ]  #indicates if the SVP on block i improved the basis
    #(if bkz does not improve on block i, it will not improve on it later, until the block gets updated - we set good_blocks[i]=False)
    while num_idle_Lovacz<n-1:
        if early_abort_niters:
            print(f"Early abort after {tour_num} iterations!")
            break
        i=iota%(n-1)
        if i==0:
            num_idle_Lovacz = 0  #zeroize at the start of a new pass through the basis
            tour_num+=1
            if verbose and iota!=0:
                print(f"Tour {tour_num} done in {perf_counter()-tour_timer}")
                tour_timer = perf_counter()
        print(f"kef i={i} iota={iota}")

        #we need to compute all Mu[i][:] and all Rr[i][:] along with Rr[i+1][i+1] (the latter to check Lovasz’ condition) that's why end = i+2
        #end could be i+1, but then in "if log_t0>=log_t1_  and good_blocks[i]": we need to compute log_t1_
        G.compute_GSO(start=i, end=i+2)
        G.unit_reduce( FI,B,U, start=i, end=i+1, debug=debug )
        G.size_reduce( FI,B,U,start=i,end=i+1, debug=debug )  #this updates U
        t0, t1 = abs( G.Rr[i][i].alg_norm() ) , abs( G.Rr[i+1][i+1].alg_norm() )

        log_t0 = log(t0.n(),2)
        log_t1_ = log(t1.n(),2)+log_alpha_K_sq #t1_ = alpha_K^2*t1
        if verbose:
            print( f"Rr[{i},{i}]  >=? alpha_K*Rr[{i+1},{i+1}]:", "{:0.2f}".format(log_t0), " >=? {:0.2f}".format(log_t1_)  )

        B_star[i] = B[i] - sum( G.Mu[i][k]*B_star[k] for k in range(i) )

        if debug&debug_flags.check_consistency:
            G.compute_GSO(start=i, end=i+2)
            B_star[i+1] = B[i+1] - sum( G.Mu[i+1][k]*B_star[k] for k in range(i+1) )
            print(f"Checking that the GSO object is consistent up until {i+1}...")
            G.check( B,debug,B_star, start=0, end=i+2 )

        if not good_blocks[i] and verbose:
            print(f"{bcolors.WARNING}BKZ at block {i,i+1} has already been called and the block isn't randomized. Skipping!{bcolors.ENDC}")
        if log_t0>=log_t1_ and good_blocks[i]:  #if anti-Lovasz is triggered and the block is 'good', we call SVP
            G.unit_reduce(FI,B,U, start=i+1, end=i+2,debug=debug)
            G.size_reduce(FI,B,U, start=i+1, end=i+2,debug=debug)
            btmp = B[i+1] - sum( G.Mu[i+1][k]*B_star[k] for k in range(i) )

            save_norm = log( G.Rr[i][i].alg_norm(),2 )

            # 2 X n projected matrix
            M = [
                B_star[i],
                btmp #Mu[i+1][i]*B_star[i] + B_star[i+1]
            ]

            bkz_beta_ = bkz_beta if first_block_beta==0 else first_block_beta
            first_block_beta = 0
            if d<=svp_oracle_threshold: #call SVP
                if use_coeff_embedding:
                    M[0], M[1] = M[0].to_number_field(K), M[1].to_number_field(K)
                    M = matrix(K, M )

                    P = embed_Q(M)
                    us = svp_coeff(P,K, bkz_beta=bkz_beta_, verbose=verbose, bkz_r00_abort=bkz_r00_abort,bkz_DSD_trick=(first_block_beta>0))    #if no DSD tweak, we just call SVP oracle
                else:
                    M_ = [
                        zfft[ii]*M[0] for ii in range(d)
                    ] + [
                        zfft[ii]*M[1] for ii in range(d)
                    ]

                    M_ = matrix(CC, [mm.to_long_CC_vector() for mm in M_] )
                    P  = canonical_embedding_for_fpylll( M_ )
                    us = svp_coeff(P,K, bkz_beta=bkz_beta_, verbose=verbose, bkz_r00_abort=bkz_r00_abort,bkz_DSD_trick=(first_block_beta>0))    #if no DSD tweak, we just call SVP oracle
            else: #recurse
                L = FIs[lfi-2].Field
                print(f"{bcolors.OKGREEN}Descend {d}->{d/2} {bcolors.ENDC}")

                m0, m1 = M[0], M[1]
                M_ = [
                    m0,
                    zfft[1]*m0,
                    m1,
                    zfft[1]*m1
                ]
                M_ = [ m_.descend_fft() for m_ in M_ ]

                us = lll_fft(M_, K=FIs[lfi-2].Field ,FIs=dict( (k,FIs[k]) for k in range(min(FIs.keys()),lfi-1) ), rho=rho_sub,
                        rho_sub=rho_sub, gamma=gamma_sub, gamma_sub=gamma_sub, bkz_beta=bkz_beta, svp_oracle_threshold=svp_oracle_threshold,
                        use_coeff_embedding=use_coeff_embedding, early_abort_niters=early_abort_niters, early_abort_r00_decrease=early_abort_r00_decrease,
                        dump_intermediate_basis=False, bkz_r00_abort=bkz_r00_abort, use_custom_idealaddtoone=use_custom_idealaddtoone, first_block_beta=first_block_beta)

                for ii in range(4):
                    tmp = us[ii].to_number_field_round( L )
                    tmp0, tmp1 = ascend( K,tmp[:2] ), ascend( K,tmp[2:] )
                    us[ii] = vector( [tmp0,tmp1] )
                print(f"{bcolors.OKGREEN}Ascend {d/2}->{d} {bcolors.ENDC}")
            tested_us = 0
            for u in us[:global_variables.max_insertion_attempts]:
                try:
                    stdout.flush()
                    s0, s1 = u[0], u[1]
                    ns0, ns1 = norm(s0),norm(s1)
                    gcdss = ZZ( gcd( ns0,ns1 ) )
                    print(f"gcdss: {gcdss, gcd( ns0,ns1 )}")
                    pip_per_svp = 14 #TODO: if we include pip, that'll be a new global variable
                    ssOK = ideal(s0,s1)
                    is_OK = ssOK==ideal(K(1))
                    if verbose and gcdss!=1:
                        print(f"Is O_K: {is_OK} | {str(ssOK)}")
                    if not is_OK and not( ns0==0 or ns1==0 or gcdss & (gcdss-1) in ZZ  ) and not tested_us>pip_per_svp and use_pip_solver:
                        if (debug&debug_flags.dump_gcdbad):
                            filedump = f"gcd{float(gcdss): .5f}.txt"
                            with open(filedump, 'a') as file:
                                file.write(f"{gcdss}\n")
                                file.write(f"{str(s0)}\n")
                                file.write(f"{str(s1)}\n")
                                file.write(f"\n")
                        stdout.flush()
                        g = None
                        try:
                            print(f"Launching pip on gcd={gcdss}, N(a)={ns0}, N(b)={ns1}")
                            g = pip_solver( s0,s1 )  #solving pip
                            gg = nf_elem( minkowski_embedding(g) )
                            # g *= gg.get_close_unit(FI,for_sqrt=False).to_number_field_round(K) #shortening the generator
                            # nerr = pari("norm_err")
                            #print(f"DEBUG: N(g/answer)={nerr}") #printing debug info

                            s0, s1 = s0/g, s1/g
                            ns0, ns1 = norm(s0),norm(s1)
                            gns  = gcd(ns0,ns1)
                            assert gns in ZZ, f"Bad gcd after pip: {gns}"
                            print(f"gcd after pip: {gns} | s0+OK + s1*OK = {ideal(s0,s1)}")
                            u = vector( (s0,s1) )
                        except Exception as e:
                            print(f"Error in pip:{e}")
                            if (debug&debug_flags.dump_gcdbad):
                                filedump = f"gcd{gcdss}.txt"
                                with open(filedump, 'a') as file:
                                    file.write(f"{e}\n")
                                    file.write(f"{str(g)}\n")
                                    file.write(f"\n")
                            tested_us+=1
                            continue

                    u0 = [ nf_elem(minkowski_embedding(uu)) for uu in u ]
                    vi = M[0]*u0[0] + M[1]*u0[1]
                    nvi = log(vi.alg_norm(),2)

                    """
                    DEBUG_MX = M[0].to_number_field(K), M[1].to_number_field(K)
                    DEBUG_VC = DEBUG_MX[0]*u[0] + DEBUG_MX[1]*u[1]
                    print( f"DEBUG: {(log((DEBUG_VC.hermitian_inner_product(DEBUG_VC)).trace(),2)/2).n(50)}" )
                    print(f"DEBUG: m0: {log(M[0].canon_norm(),2).n(30)} m1: {log(M[1].canon_norm(),2).n(30)} vi: {log(vi.canon_norm(),2).n(30)}")
                    """

                    if nvi >= global_variables.log_basis_degradation_factor+(save_norm):
                        if debug&debug_flags.verbose_anomalies:
                            print(f"{bcolors.FAIL} Ayyy, Caramba! {nvi.n(50)} >= {save_norm.n(50)}{bcolors.ENDC} with a margin {global_variables.log_basis_degradation_factor}")
                        tested_us+=1
                        continue

                    U_ = BezTransform(FIs, lfi-1, u , debug=True, use_custom_idealaddtoone=use_custom_idealaddtoone)
                    if U_ != matrix.identity(2): #if bkz gave a good vector
                        if i > 0:
                            good_blocks[i-1] = True #the adjoint two blocks are updated, hence worth considering in the next tour
                        if i < n-2:
                            good_blocks[i+1] = True
                    else: #if it's not, we terminate this SVP call since it's unlikely to find better vector
                        if verbose:
                            print("Identity matrix!")
                        good_blocks[i] = False
                        tested_us+=1
                        continue

                    u0, u1 = [ nf_elem(minkowski_embedding(uu)) for uu in U_[0] ], [ nf_elem(minkowski_embedding(uu)) for uu in U_[1] ]

                    #save changes
                    # vi = M[0]*u0[0] + M[1]*u0[1]
                    # nvi = log(vi.alg_norm(),2)
                    # if nvi >= global_variables.log_basis_degradation_factor+abs(save_norm):
                    #     if debug&debug_flags.verbose_anomalies:
                    #         print(f"{bcolors.FAIL} Ayyy, Caramba! {nvi.n(50)} >= {save_norm.n(50)}{bcolors.ENDC}")
                    #     tested_us+=1
                    #     continue
                    b     = B[i]
                    B[i]  = u0[0]*B[i]+u0[1]*B[i+1]
                    B[i+1]= u1[0]*b+u1[1]*B[i+1]

                    utmp  = U[i]
                    U[i]  = u0[0]*U[i]+u0[1]*U[i+1]
                    U[i+1]= u1[0]*utmp+u1[1]*U[i+1]

                    # - - -
                    G.update_after_svp_oracle_rank_2([u0,u1],i)
                    G.compute_GSO(start=i, end=i+1) #we update Mu and GS-vectors for the i-th basis vector
                    G.size_reduce( FI,B,U,start=i,end=i+1 )  #we size reduce i-th basis vector, we assume, the unit_reduce is not needed since we've called the SVP oracle
                    B_star[i] = B[i] - sum( G.Mu[i][k]*B_star[k] for k in range(i) )

                    npre = save_norm.n(50)
                    #npost = log(G.Rr[i][i].alg_norm(),2).n(50)

                    #print( f"deboogie-woogie: {nvi}, {npost}" )
                    if nvi <= npre-global_variables.log_basis_degradation_factor:
                        print(f"{bcolors.OKGREEN}Something done, norm Rr[{i},{i}]: {npre}- - ->{nvi.n(50)}{bcolors.ENDC}")
                        if early_abort_r00_decrease and i==0:
                            print("early_abort_r00_decrease triggered")
                    # elif npost >= npre+global_variables.log_basis_degradation_factor:    #normally shouldn't be triggered at all
                    #     print(f"{bcolors.FAIL}Something bad done, norm Rr[{i},{i}]: {npre}- - ->{npost}{bcolors.ENDC}")
                    #     num_idle_Lovacz+=1   #Little to nothing done, so we increase num_idle_Lovacz
                    else:
                        print(f"{bcolors.WARNING}Little to nothing done, norm Rr[{i},{i}]: {npre}- - ->{nvi.n(50)}{bcolors.ENDC}")
                        num_idle_Lovacz+=1   #Nothing done, so we increase num_idle_Lovacz
                except ValueError as e:
                    tested_us+=1
                    print(e)
                    continue
                except ZeroDivisionError as e:
                    tested_us+=1
                    print(e)
                    continue
                except AssertionError as e:
                    tested_us+=1
                    print(e)
                    continue
                except PariError as e:
                    tested_us+=1
                    print(e)
                    continue
                except RuntimeError as e:
                    tested_us+=1
                    print(e)
                    continue
                break

            if tested_us>= min( len(us), global_variables.max_insertion_attempts ):
                num_idle_Lovacz+=1   #Nothing done, so we increase num_idle_Lovacz
                good_blocks[i] = False
        else:
            num_idle_Lovacz+=1   #if Lovàsz condition is satisfied, nothing is done here. Thus, num_idle_Lovacz is increased.

        if verbose:
            new_alg_profile = [ log( G.Rr[ii][ii].alg_norm(),2 ) for ii in [i,i+1] ]
            ak = new_alg_profile[0]-new_alg_profile[1]
            logg = (ak/2 - d*(log(d,2)+1)) / (2*d)
            print(f"Gamma required to trigger non-Lovasz condition again is: {(2^logg).n()}")
        if iota < min_runs_amount*(n-1):
            num_idle_Lovacz=0
        if i==n-2:
            if dump_intermediate_basis:
                with open(filename, "wb") as f:
                    pickle.dump( B,f )
            if verbose:
                print(" - - - - - - - - - - - - - - -Done with the tour #", tour_num, "- - - - - - - - - - - - - - -")
                report_status(G,B,old_alg_profile)
                print(" - - - - - - - - - - - - - - - - - - - < - >  - - - - - - - - - - - - - - - - - - -")
        iota+=1

    if (num_idle_Lovacz>=n-1):
        print("Terminating LLL due to little progress.")
    """
    Size and Unit reduce everything that is left ahead (usually the last block). Note, we no longer need to update GS vectors.
    We start at the position i+1 (which is always left unreduced after the svp) and end at the end of the basis.
    Since we may not finish the last tour (if the short vector is found by svp call somewhere at the begining of the basis) in order
    to save a few approxSVP calls, we size and unit reduce all the consequent vectors to make them shorter.
    """
    G.compute_GSO(start=i+1, end=n)
    G.unit_reduce(FI,B,U, start=i+1,end=n,debug=debug)
    G.size_reduce( FI,B,U,start=min(i+1,n-1),end=n,debug=debug )  #we size reduce them

    U = [x for _, x in sorted(zip(B,U), key=lambda pair: (pair[0].alg_norm()).n())] #we sort the basis according to the Euclidean length

    if verbose:
        report_summary( G,B,K,log_alpha_K_sq,gamma,old_eucl_profile,old_alg_profile )

    return U

def report_status( G,B,old_alg_profile ):
    """
    Assuming G is valid. Reports the slope of algebraic profile and norm( Rr[0][0] ).
    param G: valid Gram matrix of the basis.
    param B: valod basis consisting of nf_vect.
    """
    n = len( G.Rr )
    new_alg_profile = [ log( G.Rr[i][i].alg_norm(),2 ) for i in range(n) ]
    dr = [ new_alg_profile[i+1]-new_alg_profile[i] for i in range(n-1) ]
    log_slope = np.mean( dr )
    print(f"log_slope = {log_slope.n(24)}, log(Rr[0,0])={new_alg_profile[0].n(24)}")

def report_summary( G,B,K,log_alpha_K_sq,gamma,old_eucl_profile,old_alg_profile ):
    """
    Reports algebraic profile, basis vectors eucledean norm and compares the former to the estimation based on the
    definition of the algebraic LLL-reduced bases.
    param G: valid Gram matrix of the basis.
    param B: valod basis consisting of nf_vect.
    param K: CyclotomicField the lattice is defined over.
    param alpha_K: alpha_K the lll_fft was invoked with.
    param old_eucl_profile: lengths of vectors in the initial basis.
    param old_alg_profile: log(Rr[i][i]) of the initial basis.
    """
    n = len( G.Rr )
    new_alg_profile = [ log( G.Rr[i][i].alg_norm(),2 )for i in range(n) ]
    new_eucl_profile = [ B[i].canon_norm() for i in range(n) ]
    d = K.degree()

    print('---------- Summary report -----------')

    print("Alg_profile:")
    print("      old          |  new ")
    for i in range(n):
        print( f"{old_alg_profile[i].n(50) }, {new_alg_profile[i].n(50)}" )
    print("Euclidean lengths of basis vectors:")
    print("      old          |  new ")
    for i in range(n):
        print( f"{old_eucl_profile[i].n(50)}, {new_eucl_profile[i].n(50)}" )

    tmp_log = log_alpha_K_sq*(n-1)+log(G.det_squared().alg_norm(),2) / n
    tmp_log = log(G.Rr[0][0].alg_norm(), 2) - tmp_log
    if tmp_log>0:
        print(f"{bcolors.FAIL}r_00 is {tmp_log.n(50)} times worse then predicted on the log scale.{bcolors.ENDC}" )
    else:
         print( f"{bcolors.OKGREEN}r_00 is {-tmp_log.n(50)} times better then predicted on the log scale.{bcolors.ENDC}" )

    log_alpha_K_sq_1 = 2*((d*(log(d,2)+1))) + log(gamma,2)
    tmp_log = 0.5*log_alpha_K_sq_1*(n-1)+log(G.det_squared().alg_norm(),2) / n
    tmp_log = log(G.Rr[0][0].alg_norm(), 2) - tmp_log
    if tmp_log>0:
        print(f"{bcolors.FAIL}r_00 is {tmp_log.n(50)} times worse then predicted on the log scale for gamma=1.{bcolors.ENDC}" )
    else:
         print( f"{bcolors.OKGREEN}r_00 is {-tmp_log.n(50)} times better then predicted on the log scale for gamma=1.{bcolors.ENDC}" )

    print( f"N(Rr[0,0])={log(G.Rr[0][0].alg_norm().n(50),2)}")
    print('---------- End summary report -----------')

class L2:

    def __init__(self,B,strategy=None, FIs=None):
        """
        param B: basis of a lattice over CyclotomicField
        param FIs: set of FieldInfo objects corresponding to the fields in the field tower. If None, those will be computed manually (or unpickled) and reused.
        """
        if strategy is None:
            strategy = LLL_params()
        self.strategy = strategy

        if qd_avaliable:
            print("QD avaliable. Using it when appropriate.")
        else:
            print(f"{bcolors.WARNING}QD not found! Using mpfr when appropriate... {bcolors.ENDC}")

        K = B[0,0].parent()
        f = 2*K.degree()
        self.f = f
        self.K = K
        self.nrows, self.ncols = B.nrows(), B.ncols()
        max_log_conductor = log(f,2)
        if FIs is None:
            with open("Field_infos.pkl", "rb") as f:    #load precomputed data
                FIs, scale_factor_log_unit_pickled = pickle.load(f)
                assert scale_factor_log_unit_pickled == global_variables.scale_factor_log_unit, "scale_factor_log_unit changed since the log unit generation. Regenerate Field_infos.pkl or set scale_factor_log_unit={scale_factor_log_unit_pickled}"
                for key in FIs.keys():
                    if key>max_log_conductor:  #if we loaded enough, stop
                        break
                    FIs[key].LLL_GSO = GSO.Mat( FIs[key].LLL_GSO, float_type="mpfr" )
                    FIs[key].LLL_GSO.update_gso()
            FIs.update( dict( (k,FieldInfo( k )) for k in range(max([key for key in FIs.keys()])+1,max_log_conductor+1 ) ) )  #if something is not precomputed, compute it
        self.FieldInfoDict = dict( (k,FIs[k]) for k in range(2,max_log_conductor+1 ) )
        self.B = B

    def lll(self, strategy=None, modify_basis=True):
        """
        Launch algebraic LLL reduction à la KEF-LLL. Returns the transformation matrix.
        param strategy: LLL_params object.
        param modify_basis: if True, we update self.B
        """

        if strategy is None:
            strategy=self.strategy

        if self.strategy["verbose"]:
            now = datetime.now()    #algorithm can work days on 1024-th field. It's nice to have info when we started.
            nowstr = now.strftime("%Y-%m-%d %H:%M:%S")

            print('------ LLL parameters summary ------')
            print(f"{bcolors.OKCYAN}{nowstr}{bcolors.ENDC} LLL is launched on a module of rank {self.nrows}")
            if strategy["use_coeff_embedding"]: print('LLL uses coefficient embedding')
            else: print('LLL uses canonical embedding')

            if strategy["use_custom_idealaddtoone"]: print('LLL uses custom idealaddtoone')
            else: print('LLL uses custom idealaddtoone')
            print('Does LLL use early abort if R[0,0] reduced by a factor', log_bkz_abort_factor, '?:', strategy["bkz_r00_abort"])

            print('LLL uses gamma=', strategy["gamma"], 'to check Lovàsz condition in non-recursive calls ')
            print('LLL uses gamma_sub = ', strategy["gamma_sub"], 'to check Lovàsz condition in all recursive calls ')
            print('-----------------------------------')


        # Main call to lll_fft. The basis should be in the fft domain
        B_ = [ nf_vect( [minkowski_embedding(bij) for bij in self.B[i]] ) for i in range(self.nrows) ]
        U = lll_fft(
                B_, self.K, FIs=self.FieldInfoDict,
                rho=strategy["rho"], rho_sub=strategy["rho_sub"], gamma=strategy["gamma"], gamma_sub=strategy["gamma_sub"],
                bkz_beta=strategy["bkz_beta"], svp_oracle_threshold = strategy["svp_oracle_threshold"],
                use_coeff_embedding=strategy["use_coeff_embedding"], debug=strategy["debug"], verbose=strategy["verbose"],
                early_abort_niters=strategy["early_abort_niters"], early_abort_r00_decrease=strategy["early_abort_r00_decrease"],
                dump_intermediate_basis=strategy["dump_intermediate_basis"], bkz_r00_abort=strategy["bkz_r00_abort"],
                use_custom_idealaddtoone=strategy["use_custom_idealaddtoone"], first_block_beta=strategy["first_block_beta"], use_pip_solver=strategy["use_pip_solver"],
                experiment_name=strategy["experiment_name"]
            )

        U = matrix( [
            u.to_number_field_round(self.K) for u in U
        ] )

        print( "We are finished with det U:", norm(det(U)) )

        if modify_basis:
            self.B = matrix([ [dot_product(U[i],self.B.column(j)) for j in range(self.ncols)] for i in range(self.nrows) ])

        return matrix(self.K,U)

    def size_unit_reduce(self,start=0,end=-1,modify_basis=True):
        """
        Size and unit reduces the basis. Returns the transformation matrix.
        param modify_basis: if True, we update self.B
        """
        FI = self.FieldInfoDict[log(self.f,2)]
        then = perf_counter()
        U = matrix( size_unit_reduce_internal( self.B,FI,start=start, end=end ) )
        if self.strategy["verbose"]:
            print(f"SU-reduction done in: {perf_counter()-then}")
        if modify_basis:
            self.B = matrix([ [dot_product(U[i],self.B.column(j)) for j in range(self.ncols)] for i in range(self.nrows) ])
        return U

def generate_FieldInfos( logf ):
    then = perf_counter()
    FIs = [ (k, FieldInfo(k)) for k in range(2,logf+1) ]
    print( f"Done in {perf_counter()-then}" )

    D = dict(FIs)
    for key in D.keys():
        D[key].LLL_GSO  = D[key].LLL_GSO.B

    with open("Field_infos.pkl", "wb") as f:
        pickle.dump( (D,global_variables.scale_factor_log_unit),f )
