"""
This is an implementation of Arakelov random walks described in:
[dBDPMW20] - oen de Boer, L´eo Ducas, Alice Pellet-Mary, and Benjamin Wesolowski.
Random self-reducibility of ideal-SVP via Arakelov random walks. In
Advances in Cryptology - CRYPTO 2020, pages 243–273, 2020
"""

#import sys, os, time
import sage.misc.randstate as randstate

Prec = 60
max_counter=10
RealNumber = RealField( Prec )
ComplexNumber = ComplexField( Prec )
RR = RealField( Prec )
CC = ComplexField( Prec )

def butterfly(v_,s):
    #butterfly step of fft

    v=[t for t in v_]
    n = len(v_)
    if n>1:
        vp = v_[0:n:2]
        vi = v_[1:n:2]
        vi = butterfly(vi,s)
        vp = butterfly(vp,s)

        zeta=(exp(-2.*I*pi/n*s)).n(Prec)
        mu=1
        for i in range(n/2):
            t=mu*vi[i]
            v_[i+n/2]=vp[i]-t
            v_[i]=vp[i]+t
            mu*=zeta
    return v_ if isinstance(v,list) else [v_] #force to return list

def ifft(v,real_value=True):
    #subroutine for inverse minkowsky

    d=len(v)
    z=(e**(-1.*pi*I/d)).n(Prec)
    z=CC(z)

    v = list(v)
    v=butterfly(v,1)

    for i in range(len(v)):
        v[i]*=(z^i)

    v = [CC(t)/d for t in v] if not real_value else [t[0]/d for t in v]

    a_= [QQ(RR(t)) for t in v]

    return a_

def inv_minkowski_embedding(s):
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    f = 4*len(s)
    K.<z> = CyclotomicField(f)
    tmp = list( s[:] ) + [0]*len(s)
    for i in range(len(s)-1,-1,-1):
        tmp[len(tmp)-1-i] = s[i].conjugate()

    return K( ifft(tmp) )

def log_embedding(a,truncated=True):
    ac = minkowski_embedding(a,truncated)
    return 2*vector(RealField(Prec), [ln(abs(t).n(Prec)) for t in ac])

def inv_log_embedding(s):
    tmp = [e.n()^(t/2) for t in s]
    a = inv_minkowski_embedding(tmp)
    return a

def roundoff(a):
  OK = a.parent().fraction_field().ring_of_integers()
  return OK( [round(t) for t in a] )

def gen_fact_base(B=5):
    P = []
    p=1
    while p< B:
        p = next_prime(p)
        P.append(p)
    return P

def nearest_P_smooth_number(N,P=None):
    p_max = None
    if P is None:
        p_max = max( 3, round( log(log(N,2),2) ) )
        P = gen_fact_base( p_max )
    if abs(ln(N)) < 10:
        scale = round( 10 + 2*log(N,2) )
    else: scale = round( sqrt(len(P))*log(N,2)^4 )
    n = len(P)
    M = [None]*(n+1)
    logn = log(N,2)
    for i in range(n):
        M[i] = [0]*i + [1] + [0]*(n-i-1) + [round( log(P[i],2) * scale )]
    M[n] = [0]*n + [round(logn * scale)]
    M = matrix(ZZ,M)
    #print(M)
    M = M.LLL()

    min_diff = Infinity
    found_v = []
    found_C = 0
    for v in M:
        C = ( prod(P[i]^v[i] for i in range(len(P))) )
        if C < 1:  #if algorithm returned inverse of ~N, make it 1/C
            C = 1/C
        diff = abs(ln(C)-ln(N))

        diff, min_diff = RR(diff),RR(min_diff) #to prevent overflow
        if diff < min_diff:
            min_diff = diff
            found_v = v
            found_C = C
    if abs(ln(found_C)-ln(N))>ln( sqrt(2).n() ):  #if it didn't work, we found garbage and give up
        return 2^( round(log(N,2)) )
    return(found_C)

def gen_smooth_nearly_unit_ideal( K, n=12, B=10^10 ):
    scale=4*log(B,2)^2

    P = [ rand_p_ideal( K, B) for i in range(n) ]
    Norms = [ ln(p.norm()) for p in P ]

    B = matrix(ZZ, [
        i*[0]+[1]+(n-i-1)*[0]+[round(Norms[i]*scale)] for i in range(n)
    ])

    B = B.LLL()

    min_norm = Infinity
    min_i = None
    for v in B:
        if abs(v[-1])<min_norm:
            min_norm = abs(v[-1])
            min_v = v
    return(prod( P[i]^min_v[i] for i in range(n) ))

# - - - The funtions - - -

rounding_factor = 2**80
log_rounding_factor = round( log(rounding_factor,2) )
factor_base = [2, 3, 5, 7, 11]

def gen_gauss_sum_to_zero(n,s):
    #Samples vector from gauss distridution with deviation s and makes the sum of its coefficients equal to zero.
    D = RealDistribution('gaussian', s)
    l = [ CC(D.get_random_element(),D.get_random_element()) for i in range(n) ]
    l[-1] -= sum(l)
    return(l)

def steps_num(d):
    """
    Given the degree of field, returns number of Arakelov jump steps required for the randomization.
    See Corollary 3.5 in https://eprint.iacr.org/2020/297.pdf
    """
    N = d/( 2*log(d,2) ) * ( 8*log(log(d,2),2)/log(d,2) + 1/2 )
    return ceil(N)

def bound_on_B(d):
    """
    Given the degree of field, returns ~ bound on the prime ideals required for the randomization.
    We omit wavy-O notation in the Corollary 3.5 in https://eprint.iacr.org/2020/297.pdf
    """
    return ceil(d^(2+2*log(d,2)))


def rand_p_ideal(K, B):
    #returns ring of integer's prime ideal of norm < B
    d = K.degree()
    assert B>=3, f"Wrong B!"

    p = 4
    counter = 0
    for t in range(10): #ideally, it's while True, but we have to wait for too long
        counter += 1
        randstate.set_random_seed(counter+(hash(p)%2**31))  #rerandomize
        p = ZZ( randrange(2,B) )
        while not ( p.is_prime() ):
            p = ZZ( randrange(2,B) )    #sample prime p
        #print(f"Prime: {p}")
        F = Ideal( K(p) ).factor()      #factor (p) into product of prime ideals
        I_norm = Infinity
        counter=0
        while I_norm >B and counter<=max_counter:   #while among factors the ideal of norm < B not found (and num of steps is not too high)
            I = F[ randrange(len(F)) ][0]   #choose new one
            I_norm = norm( I )
            counter+=1
        if I_norm<B:
            break
    return(I)

def arakelov_crawl(II, s, smooth_unit=False):
    #II is ideal
    if s==0:
        return II

    d = II.ring().degree()
    v = vector(gen_gauss_sum_to_zero(d//4,s))
    assert abs( sum([t for t in v]) ) < 10**-3, f"Not a unit after log-embedding!"

    if smooth_unit:
        U = gen_smooth_nearly_unit_ideal( II.ring() )
        assert abs( ln( abs(norm(U)) ) )<0.2, f"Not a unit! {abs( ln( abs(norm(U)) ) ).n()}"
        return U*II
    else:
        u = ( inv_log_embedding(v) )
        u_stash = u

        denom = u.denominator()
        u = roundoff( rounding_factor*u )/rounding_factor

        assert( abs( ln(norm(u_stash))-ln(norm(u)) ) < 0.01 ), f"u is not a unit! Diff: { abs(ln(norm(u_stash))-ln(norm(u))).n()}"

    #u = roundoff(rounding_factor*u)/rounding_factor
    assert abs(ln(u.norm())) < 0.2, f"arakelov_crawl - precision issues: {abs(ln(u.norm())).n()}"

    return( II*u )

def arakelov_jump(II, B):
    p = rand_p_ideal( II.ring(), B )
    II = p*II

    return II

def arakelov_rand_walk(II, B, s, N, normalize=True, smooth_unit=False):
    """
    pass all bounds to the function, no need to recompute the d
    """
    pari( r"\p 58" )
    pari.allocatemem( 1500*2**20 )
    norm_stash = II.norm()
    for i in range(N):
        II =  arakelov_jump( II, B )
    II = arakelov_crawl( II, s, smooth_unit ) #we do crawl only once
    if normalize:
        d = II.ring().degree()
        scale = ( norm(II) / norm_stash )^(1/d)
        II = II  / nearest_P_smooth_number( scale, factor_base )
    return( II )
