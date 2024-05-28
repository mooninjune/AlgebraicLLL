"""
This file contains default constants and parameters an L2 object refers to.
"""

import pickle
from time import perf_counter
from fpylll import *

global Prec
global fplllPrec
global bkz_scaling_factor
global scale_factor_log_unit
global threshold
global log_distortion
global log_basis_degradation_factor
global early_abort
global mpfr_usage_threshold_prec
global mpfr_usage_threshold_dim
global ld_usage_threshold_prec
global ld_usage_threshold_dim
global bkz_max_loops
global parisizemax
global idealaddtoone_threshold
global sieve_threads
global sieve_for_bkz
global max_insertion_attempts

class global_variables:
    Prec = 250                          #Precision we are storing nf_elem and nf_vect with.
    fplllPrec = 144                     #Fpylll's mpfr precision
    bkz_scaling_factor = 48           #Maximum magnitude of matrix over ZZ that is allowed to be passed to fpylll BKZ (in bits). < 512
    scale_factor_log_unit = 180         #Scale factor for Log Unit lattice if changed, one must regenerate the lattices (default: 180)
    threshold = 10^-14                  #threshold for assertions error when comparing equal values (e.g., norms)
    log_distortion = 10^-10             #bound on inaccuracy of log_unit_projection
    log_basis_degradation_factor = 50.0
    """
    Let L be a proper subfield of K. Let M be a OK-module. Let V be a vector of M. Let M' be descend of M to L - the OL-module. It contains vector v = descend(V).
    Suppose, approxSVP oracle called on M' returned w such that N_L(w)=N_L(v)^(1-ε) for 0<ε<1/2. Let W be ascend of w to K. Then it might be that N_K(W)=N_K(V)^(1+γ) for some γ>=0.
    If log(γ)<=log_basis_degradation_factor we allow that longer (wrt alg. norm) vector to be inserted for the sake of the basis randomization.
    """
    early_abort = None              #haven't decided how to implement it yet
    mpfr_usage_threshold_prec = 144      #minimal precision to start using mpfr
    mpfr_usage_threshold_dim = 385      #minimal dimension to start using mpfr
    ld_usage_threshold_prec = 80       #maximal precision for using ld
    ld_usage_threshold_dim = 128 #128        #maximal dimension for using ld

    bkz_max_loops = 2                  #amount of tours in BKZ algorithm called during algebraic LLL
    parisizemax = 6442713088            #amount of memory allocated by pari gp (in Bytes)
    idealaddtoone_threshold = 256
    max_insertion_attempts = 10     #amount of times LLL tries to insert a vector after SVP
    """
    idealaddtoone_threshold is a limit for the degree of a number field Pari GP's idealaddtoone is allowed
    to be called over. For degree 512 idealaddtoone seems to crash due to the fragmentation error and overall
    is really slow. If the dimension exceeds this limit, we use GEuclide to reduce the dimension until it
    reaches this value.
    """

    sieve_threads = 5                   #amount of threads utilized by the BKZ
    sieve_for_bkz = "bgj1"              #the sieve that BKZ uses

log_bkz_abort_factor = 1.1          #log base 2 of r00 decrease required to abort bkz (applicable only with flag bkz_r00_abort)

RealNumber = RealField(global_variables.Prec)
ComplexNumber = ComplexField(global_variables.Prec)
RR = RealField(global_variables.Prec)
CC = ComplexField(global_variables.Prec)

euler_const = RR(e)
pi_const = RR(pi)

class debug_flags(object):
    check_integrity         = 1
    check_consistency       = 2
    check_gs_vect           = 4
    verbose_unit            = 8
    verbose_size_reduction  = 16
    verbose_summary         = 32
    verbose_anomalies       = 64
    dump_gcdbad             = 128

    def __init__(self,flag):
        self.flag = flag

    def __str__(self):
        s = ""
        if self.flag & self.check_integrity:
            s+="check_integrity "
        if self.flag & self.check_consistency:
            s+="check_consistency "
        if self.flag & self.check_gs_vect:
            s+="check_gs_vect"
        return s

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# - - - Precomputed constants - - -

try:
    GSO_M = GSO.Mat(IntegerMatrix.identity(2), float_type='qd')
    #print("QD avaliable. Using it when appropriate.")
    qd_avaliable = True
except Exception as err:
    print(err)
    #print(f"{bcolors.WARNING}QD not found! Using mpfr... {bcolors.ENDC}")
    qd_avaliable = False

try:
    import g6k
    g6k_avaliable = True
except ModuleNotFoundError:
    print(f"g6k is not avaliable")
    g6k_avaliable = False
