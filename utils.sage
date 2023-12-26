from fpylll import *
from fpylll import BKZ as BKZ_FPYLLL
from fpylll import LLL as LLL_FPYLLL
from fpylll import GSO, IntegerMatrix, FPLLL, Enumeration, EnumerationError, EvaluatorStrategy
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
from itertools import chain
import time
from common_params import *

from svp_tools import bkz_reduce, bkz_reduce_ntru, g6k_reduce

FPLLL.set_precision(global_variables.fplllPrec)
from copy import deepcopy

def enorm_numfield(a):
    """
    Returns squared euclidean norm of a numfield element after coefficient embedding.
    param a: CyclotomicField element.
    """
    tmp = a.list()
    return sum(abs(t)^2 for t in tmp)

def enorm_vector_over_numfield(v):
    """
    Returns squared euclidean norm of a vector over numfield after coefficient embedding.
    param v: vector over CyclotomicField
    """
    return sum( enorm_numfield(t) for t in v )

def embed_Q(B, scale_for_bkz=True):
    """
    Embeds matrix B over CyclotomicField to QQ using coefficient embedding.
    param B: matrix over CyclotomicField
    param scale_for_bkz: scale input so that its coefficients are no longer then bkz_scaling_factor
    """
    n,m = B.nrows(), B.ncols()
    d = B[0,0].parent().degree()
    M = [[None for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            Bij = matrix.circulant( B[i,j].list() )
            for k in range(d):
                for l in range(k):
                    Bij[k,l]*=-1
            M[i][j] = Bij

    M = matrix.block(M)
    if scale_for_bkz:
        ceiling = log(max( [abs(t) for t in M.coefficients()]).n(), 2 )
        p = max( 0, ceil((ceiling-global_variables.bkz_scaling_factor)) )

        for i in range(M.nrows()):
            for j in range(M.ncols()):
                M[i,j] = M[i,j]>>p
    return M

def solve_left( B, v ):
    """
    Replaces malfunctioning B.solve_left(v).
    param B: matrix over pow-of-2 CyclotomicField
    param v: vector over pow-of-2 CyclotomicField
    """
    K = B[0,0].parent()
    d = K.degree()
    n, m = B.nrows(), B.ncols()

    BB = embed_Q( B, scale_for_bkz=False )
    vv = []
    for v_ in v:
        vv += v_.list()
    vv = vector( vv )

    uu = BB.solve_left( vv )
    u = vector( [ K(uu[i*d:(i+1)*d].list()) for i in range(m-1) ] )
    return u

def roundoff(a):
    """
    Rounds element of a number field. Used in util_l2.
    param a: CyclotomicField element.
    """
    return a.parent()( [round(t) for t in a] )

def butterfly(v_,s):
    """
    Butterfly step of fft.ы
    param v_: vector over CC.
    param s: either 1 or -1.
    """
    v=[t for t in v_]
    n = len(v_)
    if n>1:
        vi = butterfly(v_[1:n:2],s)
        vp = butterfly(v_[0:n:2],s)

        zeta=(exp(-2.n(global_variables.Prec)*I*pi/n*s)).n(global_variables.Prec)
        mu=1
        for i in range(n/2):
            t=mu*vi[i]
            v_[i+n/2]=vp[i]-t
            v_[i]=vp[i]+t
            mu*=zeta
    return v_ if isinstance(v,list) else [v_] #force to return list

def ifft(v):
    """
    Subroutine for inverse minkowski.
    param v: vector over CC.
    """
    d=len(v)
    z=(euler_const**(-1.n(global_variables.Prec)*pi*I/d)).n(global_variables.Prec)
    z=CC(z)
    v=butterfly(v,1)
    for i in range(d):
        v[i]*=(z^i)
    v = [t[0]/d for t in v]
    return v

def minkowski_embedding(a):
    """
    Given a in CyclotomicField, returns its minkowski embedding.
    param a: CyclotomicField element.
    We have real coefficients so only half the embeddings are enough.
    Applicable only to 2^h cyclotomics where h>=2.
    """

    sigmas = a.parent().fraction_field().embeddings(CC)
    return vector( CC, [s(a) for s in sigmas[:len(sigmas)/2]] )

def minkowski_embedding_blind(a,K):
    """
    Given nf_elem a, returns its minkowski embedding.
    param a: nf_elem
    param K : CyclotomicField, (a) was defined over.
    We have real coefficients so only half the embeddings are enough.
    Applicable only to 2^h cyclotomics where h>=2.
    """

    sigmas = K.embeddings(ComplexField(global_variables.Prec))
    return vector( CC, [s(a) for s in sigmas[:len(sigmas)/2]] )

def inv_minkowski_embedding(s):
    """
    Given s in freq domain, computes inverse to minkowski_embedding.
    param s: vector over CC.
    """
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    ls = len(s)
    tmp = list( s[:] ) + [0]*ls
    l = len(tmp)
    for i in range(ls-1,-1,-1):   #TODO use chain here
        tmp[l-1-i] = s[i].conjugate()
    return vector( ifft(tmp) )

def canonical_embedding_for_fpylll(M_):
    """
    Given M_ - a (n x m) matrix over CC, returns its embedding to RR and rounds it to ZZ.
    param M: matrix over CC.
    """
    n,m = M_.nrows(), M_.ncols()
    M = matrix([
      chain.from_iterable( [ [M_[i,j].real(),M_[i,j].imag()] for j in range(m) ]) for i in range(n)
    ])
    cmax = floor( log( max( abs(c) for c in M.coefficients() ),2 ) )
    p = global_variables.bkz_scaling_factor-cmax

    M*=2^p
    M = matrix(ZZ,[
      [ round(b) for b in bb ] for bb in M
    ])

    ceiling = log(max( [abs(t) for t in M.coefficients()]).n(), 2 )
    p = max( 0, ceil((ceiling-global_variables.bkz_scaling_factor)) )

    for i in range(M.nrows()):
        for j in range(M.ncols()):
            M[i,j] = M[i,j]>>p

    return M


def svp_coeff(M,K, bkz_beta=14, verbose=False, bkz_r00_abort=False, bkz_DSD_trick=False):
    """
    Given module, solves approxSVP and outputs corresponding coeffitients of the short vectors w.r.t the basis M.
    param M: nf_vect list defining the basis of the module (no linear dependency allowed!).
    param K: CyclotomicField M is defined over.
    param bkz_beta: block size for BKZ.
    param bkz_r00_abort: see bkz_r00_abort in bkz_reduce.
    """
    if bkz_DSD_trick:
        U_ = bkz_reduce_ntru(M, block_size=min(60,bkz_beta), verbose=verbose, bkz_r00_abort=bkz_r00_abort)
    else:
        U_ = bkz_reduce(M, block_size=min(60,bkz_beta), verbose=verbose, bkz_r00_abort=bkz_r00_abort)
        # DEBUG_MX = U_*embed_Q(M)
        # print( f"DEBUG: svp_coeff: {(log(norm(DEBUG_MX[0]),2)/2).n()}" )

    if bkz_beta>=60 and g6k_avaliable:
        M = IntegerMatrix.from_matrix( (U_*matrix(M)).change_ring(ZZ) )
        U2_ = g6k_reduce(M, bkz_beta, verbose=True, task_id=None, sort=True)
        U_ = U2_ * U_

    U = [None for i in range(U_.nrows())]
    d=K.degree()
    for i in range(U_.nrows()):
        u = U_[i]
        U[i] = vector(K, [ K(u[j*d:(j+1)*d].list()) for j in range(U_.ncols()/d) ] )
    return U
