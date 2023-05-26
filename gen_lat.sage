"""
This file implements construction for several lattices.
Namely, it can generate:
    ○ Unimodular matrtices;
    ○ U*B*W matrix for unimodular U and W;
    ○ NTRU lattices with ternary secret;
    ○ LWE matrices;
    ○ Kannan-like mattrices;
    ○ ModFalcon (ModNTRU) lattices.
"""

from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

def gen_unimod_matrix_fast( K,n, sigma=0.75, density=0.2 ):
    #Generates the matrix U such that its determinant is unit.
    z=K.gen()
    d = K.degree()
    #OK = K.fraction_field().ring_of_integers()
    D = DiscreteGaussianDistributionIntegerSampler(sigma=sigma)

    units = [z**((1 -(i))/2 ) * (1 -z**i)/(1 -z) for i in range(1,K.degree(),2) ] if K.degree()>2  else [1 ,z]

    L = matrix.identity(K,n)
    U = matrix.identity(K,n)
    for i in range(n):
        for j in range(n):
            if i<j:
                if uniform(0 ,1 )<density:
                    L[i,j]=K([ D() for counter in range(d) ])
            elif j<i:
                if uniform(0 ,1 )<density:
                    U[n-j-1 ,n-i-1 ]=K([ D() for counter in range(d) ])
            else:
                U[i,i]*=prod([ units[randrange(len(units))] for i in range(6) ])
                L[i,j]*=prod([ units[randrange(len(units))] for i in range(6) ])

    #We cannot multiply two mattrices with sage's inline function because of the bug [ see: https://trac.sagemath.org/ticket/34597 ]
    return matrix( [[L[i].dot_product(U.transpose()[j]) for j in range(n)] for i in range(n)])

def gen_ntru_instance(K,q, seed=None):
    y = var('y')
    ZqZ = Integers(q)
    Zxq = ZqZ[y]
    Rq.<t>=Zxq.quotient_ring(K.polynomial())
    if not seed is None:
        set_random_seed( seed )
    while True:
        try:
            g_ = Rq( [randrange(-1,2) for i in range(K.degree())] )
            f_ = Rq( [randrange(-1,2) for i in range(K.degree())] )
            h_ = f_/g_
        except Exception as ex:
            print(ex)
            continue
        break

    tmp = len(list(h_))
    h_ = K([(int(h_[i])+1)%q -1 for i in range(tmp)])
    f_ = K([(int(f_[i])+1)%q -1 for i in range(tmp)])
    g_ = K([(int(g_[i])+1)%q -1 for i in range(tmp)])

    return f_, g_, h_

def gen_NTRU( f,q,seed=None ):
    """
    f - field conductor
    q - modulus
    """
    assert is_prime(q), f"{q} is not prime!"
    K.<z> = CyclotomicField(f)
    if not seed is None:
        set_random_seed( seed )
    f_, g_, h_ = gen_ntru_instance( K,q,seed=seed )

    B = matrix( K, [
        [K(q), 0],
        [h_,  1]
    ])

    return B, f_, g_

def gen_LWE( f,q,k,seed=None ):
    """
    f - field conductor
    q - modulus
    k - half the rank
    """
    n = 2*k
    q = ceil(q)
    K.<z> = CyclotomicField(f)
    if not seed is None:
        set_random_seed( seed )
    d = K.degree()
    A = matrix([
        [ K([randrange(-q,q) for l in range(d)]) for j in range(k) ] for i in range(k)
    ])

    B = matrix.block( [[q*matrix.identity(k), matrix.zero(k)],[A, matrix.identity(k)]] )
    return B

def gen_UBW( f,q,n,m, sigma=0.45,seed=None ):
    """
    f - field conductor
    q - matrix coeffs ampltude depends on it
    n - rank of module
    m - dmension of ambient space
    sigma - Disc Gauss dispersion (larger => more noised the matrix is)
    """
    q = round(q)
    K.<z> = CyclotomicField(f)
    d = K.degree()

    if not seed is None:
        set_random_seed( seed )
    B = matrix(K,[
     [ K([randrange(-q,q) for k in range(d)]) for j in range(m) ] for i in range(n)
    ])

    U = gen_unimod_matrix_fast( K, n, density=1.0, sigma=sigma )
    W = gen_unimod_matrix_fast( K, m, density=1.0, sigma=sigma )
    B = U*B*W

    return B

def dot_product(v,w):
    """
    Implemenst dot product of two vectors over CyclotomicField, but actually works (see https://trac.sagemath.org/ticket/34597).
    param v: vector over CyclotomicField
    param u: vector over CyclotomicField
    Outputs v.dot_product(u), but computes it correctly.
    """
    return(sum(v[i]*w[i] for i in range(len(v))))

def Kannan_like( f,q,k, sigma_c=0.8, sigma_err=1.8, seed=None, T=1/4 ):
    M = gen_LWE( f,q,k,seed=seed )
    K = M[0,0].parent()
    d = K.degree()
    n,m = M.nrows(), M.ncols()

    Dc = DiscreteGaussianDistributionIntegerSampler( sigma_c )
    Derr = DiscreteGaussianDistributionIntegerSampler( sigma_err )
    s = vector( [ K( [Dc() for k in range(d)] ) for i in range(k) ] )
    e = vector( [ K( [Derr() for k in range(d)] ) for i in range(k) ] )

    BB = M[k:2*k,:k]
    v = -vector([ dot_product(s,BB.column(j)) for j in range(k) ])
    v = vector( [K([vvv%q for vvv in vv]) for vv in v] )
    verr = v+e

    B = matrix(K,n+1,m+1)
    for i in range(n):
        for j in range(m):
            B[i,j] = M[i,j]
    for j in range(k):
        B[n,j]=verr[j]
    t =  K( T )
    B[n,m] = t

    return B,s,e,t



def balancemod( K,g,q ):
    #checked
    return vector( [ K( [(ZZ(ggg)+q//2)%q-q//2 for ggg in gg.lift()] ) for gg in g ] )

def keygen_modfalcon(K,q,k,seed=0,sigmaf=0.65,sigmag=0.65):
    DF = DiscreteGaussianDistributionIntegerSampler( sigma=sigmaf )
    Dg = DiscreteGaussianDistributionIntegerSampler( sigma=sigmag )

    d = K.degree()
    phix = K.polynomial()

    set_random_seed( seed )
    #Rq = K.quotient_ring( q )
    Rq = Integers(q)[x].quotient_ring(K.polynomial())

    inv_flag=False
    while not inv_flag:
        F = matrix( [
            [ Rq( K( [DF() for l in range(d)] ) ) for j in range(k) ] for i in range(k)
        ] )
        inv_flag = F.is_invertible()
    g = vector( [Rq( [Dg() for l in range(d)] ) for i in range(k)] )

    Finv =  F^-1 #(1/det(F) * F.adjugate())
    h = ( Finv*g )
    h = balancemod( K,h,q )

    BNTRU = matrix( K,k+1,k+1 )
    for i in range(k):
        BNTRU[i,0] = -h[i]
    for i in range(0,k):
        BNTRU[i,i+1] = 1
    BNTRU[k,0] = K(q)
    return BNTRU, matrix( [balancemod(K,f,q) for f in F] ), balancemod( K,g,q ), h
