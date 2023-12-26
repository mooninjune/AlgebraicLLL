"""
Utils implemented from developing the ideas of Kirshner, Espitau and Fouque.
This file implements:
    ○ Ascending / Descending;
    ○ Log embedding;
    ○ Log unit lattice decoding;
    ○ Efficient Bezout equation solver.
"""

from sage.all_cmdline import *   # import sage library

import time
import copy
import numpy
from numpy import fft, array
import random as rnd
from fpylll import LLL as LLL_FPYLLL
from fpylll import IntegerMatrix, GSO, FPLLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from utils_wrapper import butterfly, ifft, minkowski_embedding, inv_minkowski_embedding, roundoff, embed_Q
from util_l2 import FieldInfo, nf_elem, round_babai, compute_log_unit_lattice, size_unit_reduce_internal
import sys

from common_params import *

# - - -

def dot_product(v,w):
    """
    Implemenst dot product of two vectors over CyclotomicField, but actually works (see https://trac.sagemath.org/ticket/34597).
    param v: vector over CyclotomicField
    param u: vector over CyclotomicField
    Outputs v.dot_product(u), but computes it correctly.
    """
    return(sum(v[i]*w[i] for i in range(len(v))))

def fast_mult(a,b, fft_domain=False):
    """
    Returns the approximate value of a*b for number field elements a and b. Checked.
    If fft_domain: returns the vector over CC so you can add it as an element.
    """
    K, L = a.parent().fraction_field(), b.parent().fraction_field()
    if not K==L:
        if K.is_subring(L):
            K=L
            a=K(a)
        elif L.is_subring(K):
            b=K(b)
        else:
            print("Numfield incompatible!")
    if K.degree()<=4 :
        return a*b

    aa, bb = minkowski_embedding(a), minkowski_embedding(b)
    cc = [aa[i]*bb[i] for i in range(len(a.list())/2 )]
    if fft_domain:
        return vector(cc)
    return K( inv_minkowski_embedding(cc) )

def fast_inv(a, fft_domain=False):
    """
    Approximate inverse of a, but using the Minkowski embedding.
    paran a: CyclotomicField element
    param fft_domain: return result in frequency domain
    """
    K = a.parent().fraction_field()
    if K.degree()<=4 :
        return 1 /a
    aa = minkowski_embedding(a)
    try:
        aa = [1 /t if t!=0  else 0  for t in aa]
    except ZeroDivisionError:
        raise ZeroDivisionError("FFT inverse doesn't exist!")
    if fft_domain:
        return vector(aa)
    return K( inv_minkowski_embedding(aa) )

def fast_sqrt(a, fft_domain=False):
    """
    The square root of a, but using the Minkowski embedding.
    paran a: CyclotomicField element
    param fft_domain: return result in frequency domain
    """

    if a in RR or a in QQ:
        return QQ( sqrt(abs(a.n()) ))
    K = a.parent()
    tmp = minkowski_embedding(a)

    tmp = [sqrt(t.n()) for t in tmp]
    if fft_domain:
        return vector(tmp)
    return K( inv_minkowski_embedding(tmp) )

def fast_hermitian_inner_product(u,v, fft_domain=False):
    """
    Computes hermitian inner product of u and v.
    paran u: vector over CyclotomicField
    paran v: vector over CyclotomicField
    param fft_domain: return result in frequency domain
    """
    if u[0].parent().fraction_field().degree()<=4 :
        return u.hermitian_inner_product(v)
    if fft_domain:
        return sum( fast_mult( v[i], u[i].conjugate(), fft_domain=True ) for i in range(len(u)) )
    return sum( [fast_mult( v[i], u[i].conjugate() ) for i in range(len(u))] )

# - - -

def ascend(K,v):
    """
    Ascends vector v to element of the field K. Checked.
    param K: CyclotomicField
    param v: vector over CyclotomicField
    """

    qh = len(v)
    d_ = v[0 ].parent().degree()
    d=d_*qh
    z_=K.gen()

    v_z = [0 ]*qh*d_

    for i in range(qh):
        for j in range(d_):
            v_z[j*qh+i] = v[i][j]

    out = K(v_z)
    return(out)

def descend(K,a):   #only for K - cyclotomic of power 2
    """
    Descends number field element a to the vector space over subfield.
    param K: CyclotomicField
    param a: CyclotomicField element
    """
    d_ = a.parent().degree()

    out = [0 ,0 ]
    for i in range(2 ):
        out[i] =  K(a.list()[i:d_:2 ])
    return out

def invertibles(f):
    """
    Returns all 0<=i<f that are coprime with f.
    param f: integer.
    """
    out=[0  for i in range(euler_phi(f))]

    t=0
    for i in range(f):
        if gcd(i,f)==1 :
            out[t]=i
            t+=1
    return out

def log_embedding(a):
    """
    Performs the Log-embedding on number field element a.
    param a: CyclotomicField element
    """
    ac = minkowski_embedding(a)
    return 2 *vector(RR, [ln(abs(t)) for t in ac])

def inv_log_embedding(s,K):
    """
    Inverse of the Log embedding.
    param s: vector over CC
    param K: CyclotomicField
    """
    tmp = [euler_const**(t/2 ) for t in s]
    a = K( inv_minkowski_embedding(tmp) )
    return a


def GEuclide(L, Lptr,a,b, debug=False, use_custom_idealaddtoone=True):
    """
    Outputs mu, nu such that a*mu + b*nu is unit.
    param L: dictionary of FieldInfo objects.
    param Lptr: position of FieldInfo object in L.
    param a: CyclotomicField element
    param b: CyclotomicField element
    param debug: reserved for debug
    param use_custom_idealaddtoone: see GEuclide_base_case
    """

    K = a.parent().fraction_field()

    if K.degree()<=global_variables.idealaddtoone_threshold :     #if the field is small enough, we can use pari gp to solve Bezout equation exactly.
        return GEuclide_base_case(L, Lptr, a, b, use_custom_idealaddtoone=use_custom_idealaddtoone)
    try:
        assert K.degree()==1  or K==L[Lptr].Field   #assert if the pointer to the field is correct
    except AssertionError:
        raise AssertionError("Wrong pointer in GEuclide")
    subK = L[Lptr-1 ].Field

    Na, Nb = a.norm(subK), b.norm(subK)
    mu, nu = GEuclide(L,Lptr-1 ,Na,Nb, debug=debug, use_custom_idealaddtoone=use_custom_idealaddtoone)
    #After this step norm(a.norm(subK)*mu+b.norm(subK)*nu) is 1.

    mu_, nu_ =  mu * a**-1  * Na , nu * b**-1  * Nb   #now norm(a*mu_+b*nu_) == 1. Same procedure as in [KEF17]

    U_bezout = matrix(K,[
        [a,b],
        [nu_,mu_]
    ])

    U = size_unit_reduce(U_bezout,L[Lptr],start=1,end=2) #We don't need to reduce (a,b) since those lead us to a short vector. We just need to curb the length of (nu,mu).
    U_bezout = matrix([ [dot_product(U[i],U_bezout.column(j)) for j in range(2)] for i in range(2) ])
    return(U_bezout)

def GEuclide_base_case(L, Lptr,a,b,use_custom_idealaddtoone=True):
    """
    Given a, b s.t. (a) is coprime to (b) returns mu, nu s. t. a/mu-b/nu = 1. Uses pari gp.
    param L: FieldInfo dictionary
    param Lptr: position of K - the field a and b are defined over
    param a: nf_elem defined over K
    param b: nf_elem defined over K
    param use_custom_idealaddtoone: use custom implementation of idealaddtoone.
    """
    na, nb = norm(a), norm(b)
    g=gcd(na,nb)
    AB = ideal(a,b)
    assert AB==ideal(a.parent()(1)) , f"GEuclide non-coprime ideals. gcd={g} ideal: {str(AB)} "

    if na==0 or nb==0:
        if a==0  and abs(nb) == 1:
            return 0 , 1 /b
        if b==0  and abs(na)==1:
            return 1 /a, 0

    if not use_custom_idealaddtoone:
        print("Using pari gp idealaddtoone. Might be slow for large fields.")
        K = a.parent().fraction_field()
        A = Ideal(K,a)
        B = Ideal(K, b)

        pari.default("parisizemax",global_variables.parisizemax)
        t0, t1 = pari.idealaddtoone(K,A,B)
        t0, t1 = K(t0), K(t1)
    else:
        t0, t1 = idealaddtoone_custom( a,b )
    mu, nu = t0/a,  t1/b
    assert abs(norm(t0+t1))==1, f"Bad Bezout before unit reduce!"
    return mu, nu

def idealaddtoone_custom(a,b):
    """
    Finds aa in a*OK and bb in b*OK s.t aa+bb=1.
    param a: CyclotomicField element.
    param b: CyclotomicField element.
    """
    A = ideal(a)
    B = ideal(b)
    K = a.parent()
    d = K.degree()

    #Extended Euclid in Dedekind Domains (Cohen H. - Advanced Topics in Computational Number Theory, Algorithm 1.3.2)
    AA, BB = matrix( vector(t.list()) for t in A.integral_basis() ), matrix( vector(t.list()) for t in B.integral_basis() )

    W = matrix( AA.rows()+BB.rows() ).change_ring(ZZ)
    W = pari(W.transpose())
    M,U = pari.mathnf(W,flag=1)  #flag = 1 is used to retrieve the transformation matrix
    U = matrix(U).transpose()
    X = U[d][:d]
    aa = K( (X*AA).list() )
    bb = 1-aa
    #assert( aa in A and bb in B), "idealaddtoone_custom error!"

    t = roundoff( aa/(a*b) )*a*b  #we don't use Algorithm 1.4.13 from Cohen, we reduce the answer using roundoff.
    aa -= t
    bb += t
    return aa, bb

def BezTransform(L,Lptr,v,debug=False, use_custom_idealaddtoone=True):
    """
    Given 2-vector over 2^(n)th CyclotomicField, obtains the unimodular 2x2 matrix over (2^n)-th CyclotomicField
    with first vector being ascend(v). If no such vector is found, throws an exception.
    param L: FieldInfo dictionary
    param Lptr: position of the 2^(n)-th CyclotomicField FieldInfo object in L
    param v: 4-vector to be ascended to the transformation
    param use_custom_idealaddtoone: see use_custom_idealaddtoone in GEuclide_base_case
    """
    K = L[Lptr].Field; (z,) = K._first_ngens(1)
    d = K.degree()
    a, b = K(v[0]), K(v[1])
    Na, Nb = abs(norm(a)), abs(norm(b))
    Ngcd = ZZ( gcd( Na,Nb ) )

    if (Ngcd & (Ngcd-1) == 0):  #if power of 2
        print(f"pow of 2!: {Ngcd}")
        pow_of_2_elems = [                      #subset of these elements have norms 1,2,4,8,16,32 (starting with 64th CyclotomicField all of them have pow-of-2 norm)
            1,                                  #and Na or Nb is guranteed to be divisible by pow_of_2_elems[i] for i=log(Ngcd,2)
            z + 1,
            -z^10 - 1,
            -z^3 - z^2 - z - 1,
            z^4 - 1,
            -z^5 - z^4 - z - 1
        ]
        e = 1
        p = log( Ngcd,2 )

        while p>4:  #instead of dividing by (z+1)^p, we iteratively divide by unit*(z+1)^5 = (-z^5 - z^4 - z - 1)
            e*=pow_of_2_elems[5]
            p-=5
        e *= pow_of_2_elems[ p ]
        a, b = a/e, b/e
        Na, Nb = norm(a), norm(b)
        Ngcd = gcd(Na,Nb)
    elif min(Na,Nb)==0:   #if a or b is 0, but the other one is not unit after our trick with pow-of-2 elements, we failed
        raise ValueError(f"min{Na,Nb}=0 !, {a, b}")

    mu, nu = GEuclide(L,Lptr,a,-b)  #returns mu, nu such that a*mu - b*nu = 1

    U_bezout = matrix(K,[
        [a,b],
        [nu,mu]
    ])

    testdet = norm( det(U_bezout) )
    assert testdet == 1, f"Non unimodularity in Bezout! {testdet}"

    return(U_bezout)


# - - - for test.sage - - -

def fast_mat_mult(A,B):
    """
    The matrix multiplication over the number field, but coefficients are multiplied using FFT.
    param A: matrix over CyclotomicField
    param B: matrix over CyclotomicField
    """

    C = matrix(A[0 ,0 ].parent(), A.nrows(), B.ncols())
    for i in range(A.nrows()):
        for j in range(B.ncols()):
            C[i,j] = sum( [fast_mult(A[i,k], B[k,j]) for k in range(A.ncols())] )
    return(C)
