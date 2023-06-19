from time import perf_counter
from utils import minkowski_embedding, inv_minkowski_embedding, minkowski_embedding_blind, roundoff
from copy import deepcopy
import warnings
from fpylll import LLL as LLL_FPYLLL
from fpylll import IntegerMatrix, GSO, FPLLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll import BKZ as BKZ_FPYLLL
from itertools import chain #toflatten the lists

from common_params import *

def log_embedding(a):
    """Performs the Log-embedding on number field element a."""
    ac = minkowski_embedding(a)
    return vector([ln(abs(t)) for t in ac])

class nf_vect:
    """
    Same as FreeModuleElement_generic_dense, but in freq domain.
    """

    def __init__( self, v_ ):
        """
        Initializes self from the minkowski embedding of power-of-2 CyclotomicField vector.
        """
        self.v = v_

    def __len__(self):
        return self.v.__len__()

    def __str__(self):
        return self.v.__str__()

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self,i,value):
        self.v[i] = value

    def conjugate(self):
        return nf_vect( [vv.conjugate() for vv in self.v] )

    def to_number_field(self, K):
        """
        Maps self to power-of-2 CyclotomicField vector.
        param K: CyclotomicField
        """
        v_ = vector([K(inv_minkowski_embedding(vv).list()) for vv in self.v])
        return v_

    def to_number_field_round(self, K):
        """
        Maps self to power-of-2 CyclotomicField vector and rounds it off.
        param K: CyclotomicField
        """
        v_ =[inv_minkowski_embedding(vv) for vv in self.v]
        for i in range(len(v_)):
            v_[i] = [ round(real(t)) for t in v_[i] ]
        v_ = vector([K(vv) for vv in v_])
        return v_

    def descend_fft(self):
        w = chain.from_iterable( nf_elem(vv).descend_fft() for vv in self.v )
        return nf_vect( [ww.elem for ww in w] )

    def __eq__(self, w):
        return norm(self.v - w.v).numerical_approx()<10^-8

    def __add__(self, w):
        ls = len(self)
        assert ls==len(w), f"Can't add nf_vect of lens {len(self)} and {len(w)}"
        return nf_vect([self.v[i]+w.v[i] for i in range(ls)])

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, w):
        if w in ZZ and w==0:
            return self
        assert len(self)==len(w), f"Can't add nf_vect of lens {len(self)} and {len(w)}"
        return nf_vect([self.v[i]-w.v[i] for i in range(len(self))])

    def __mul__(self,c):
        if isinstance( c, (int, float, sage.rings.integer.Integer,sage.rings.real_mpfr.RealNumber) ):
            return nf_vect( [c*(self.v[i]) for i in range(len(self))] )
        tmp = [c.elem.pairwise_product(self.v[i]) for i in range(len(self))]
        return nf_vect( tmp )

    def dot_product(self,w):
        assert len(self)==len(w), f"Can\'t add nf_vect of lens {len(self)} and {len(w)}"
        u = [self.v[i].pairwise_product(w.v[i]) for i in range(len(self.v))]
        return nf_elem( sum(u) )

    def hermitian_inner_product(self,w):
        assert len(self)==len(w), f"Can't add nf_vect of lens {len(self)} and {len(w)}"
        u = [self.v[i].pairwise_product(w.v[i].conjugate()) for i in range(len(self.v))]
        return nf_elem( sum(u) )

    def to_long_CC_vector(self):
        """
        Concatinates all the vestors in self.v
        """
        return vector(CC, chain.from_iterable( self.v ) )

    def alg_norm(self):
        v = self.hermitian_inner_product(self)
        return v.alg_norm()

    def canon_norm(self):
        """
        Returns squared euclid norm w.r.t canonical embedding.
        """
        c = self.to_long_CC_vector()
        return norm(c)


class nf_elem:
    """
    Number field element implenentation. Elements are stored in freq domain.
    """
    def __init__(self,elem):
        """
        Initializes self from the minkowski embedding of power-of-2 CyclotomicField element.
        """
        self.elem = elem

    def __len__(self):
        return  len(self.elem)

    def __getitem__(self, index):
        return self.elem[index]

    def __add__(self, other):
        return nf_elem( self.elem+other.elem )

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self,other):
        return nf_elem( self.elem-other.elem )

    def __mul__(self,other):
        if isinstance( other, (int, float, sage.rings.integer.Integer,sage.rings.real_mpfr.RealNumber) ):
            return nf_elem( other*self.elem )
        if isinstance(other,nf_vect):
            return other*self
        assert len(self)==len(other), f"Wrong elements in mul: {len(self),len(other)}"
        return( nf_elem( self.elem.pairwise_product(other.elem) ) )

    def __rmul__(self,other):
        return self*other

    def inv(self):
        """
        Reurns 1/self.
        """
        return nf_elem( vector([1/t for t in self.elem]) )

    def __truediv__(self,other):
        """
        Returns self/other.
        """
        b = other.inv()
        return self*b

    def sqrt(self):
        """
        Returns sqrt of self. Works properly if self in R^+.
        """
        return nf_elem( vector(CC, [ sqrt(x) for x in self.elem]) )

    def __pow__(self, p):
        """
        Returnd p-th power of self.
        param p: integer.
        """
        assert p in ZZ, "Only integer powers are supported!"
        if p==0:
            return nf_elem( vector(CC,[1 for ii in self.elem]) )
        if p<0:
            x=self.inv()
            p=-p
        else:
            x = nf_elem(self.elem)
        y = nf_elem( vector(CC,[1 for ii in self.elem]) )
        while p>1:
            if p%2==0:
                x *= x
                p = p//2
            else:
                y *= x
                x *= x
                p = (p-1)//2
        return x*y

    def descend(self, L):
        """
        Descends self to field L.
        param L: CyclotomicField to descend to.
        """
        v = self.elem
        d = 2*len(v)
        v = inv_minkowski_embedding( v )
        a, b = v[0:d:2], v[1:d:2]
        a = L( vector(QQ,a).list() )
        b = L( vector(QQ,b).list() )

        return [ nf_elem( minkowski_embedding(a) ), nf_elem( minkowski_embedding(b) ) ]

    def descend_fft(self):
        """
        Descends self to subfield L such that [K:L] = 2. Returns (u,w) such that self = u+z*w where K = Q[z].
        """
        v = self.elem
        d = len(self)
        z = CC( exp( I*RR(pi_const)/(2*d) ) )

        u = [None for i in range(d/2)]
        for i in range(d/2):
            u[i] = v[i]+v[d-1-i].conjugate()

        w = [None for i in range(d/2)]
        for i in range(d/2):
            w[i] = -z^(2*d-2*i-1) * v[i] + z^(2*d-2*i-1) * v[d-1-i].conjugate()

        return ( nf_elem( vector(u)/2 ), nf_elem( vector(w)/2 ) )


    def get_close_unit(self, FI, for_sqrt=True):
        """
          This function returns the unit u, s.t. u*sqrt(self) is roughly shortest possible in ideal(self).
          param FI: FieldInfo object
          param for_sqrt: if True, we search for the unit close to sqrt(self)

          Output is in fft domain.
        """
        assert FI.Field.degree() == 2*len(self), f"Wrong FieldInfo! {FI.Field.degree() , len(self)}"
        try:
            h = self.log_unit_projection( for_sqrt=for_sqrt )
        except AssertionError as err:
            print(err)
            raise err
            return nf_elem( minkowski_embedding( FI.Field(1) ) )
        u=round_babai(FI,h)

        return u

    def log_unit_projection(self, for_sqrt=True):    #debug
        """
        Finds the projection of self onto the span of Log-unit lattice.
        param for_sqrt: if True, we project sqrt(self)
        """
        dim = len(self)

        l = self.elem
        if for_sqrt:
            l/=2

        l = vector( [ ln(abs(t)) for t in l] )
        h = l - sum(l)/(dim) * vector([1 for i in range(dim)])   #projection of l orthogonally against one-vector
        #assert abs( sum(h) ) < global_variables.log_distortion, f"Unit reduced badly! {abs( sum(h) )}, {self.elem}"

        return h

    def to_number_field(self,K):
        """
        Return corresponding number field element.
        param K: original field self is defined over.
        """
        return K( inv_minkowski_embedding(self.elem) )

    def to_number_field_round(self, K):
        v = [ round(t) for t in inv_minkowski_embedding(self.elem)]
        return K( v )

    def conjugate(self):
        """
        Conjugates self.
        """
        return nf_elem( self.elem.conjugate() )

    def __str__(self):
        return str(self.elem)

    def roundoff(self,K):
        """
        Does roundoff. Returns answer in time domain.
        """
        return roundoff(self.to_number_field(K))

    def roundoff_fft(self,K):
        v = [ round(vv) for vv in inv_minkowski_embedding(self.elem) ]
        return nf_elem( minkowski_embedding_blind(v,K) )

    def alg_norm(self):
        """
        Conputes algebraic norm of self.
        """
        l = self.elem.list()
        l = prod(l)
        return RR( l*l.conjugate() )

    def canon_norm(self):
        """
        Returns square of euclidean norm of self
        """
        return norm( self.elem )

class GSO_obj: #checked

    def __init__(self,B):
        """
        GSO_obj constructor.
        param B: list of nf_vect - module basis.
        """

        n = len(B)
        m = len(B[0])
        self.n, self.m = n,m
        self.d = len(B[0][0])
        self.G = [ [None for j in range(i+1)] for i in range(n) ]
        self.valid_blocks = 0

        for i in range(n):
            for j in range(i+1):
                self.G[i][j] = B[i].hermitian_inner_product(B[j])
            self.valid_blocks += 1<<i

        self.Mu, self.Rr = [[None for j in range(m)] for i in range(n)], [[None for j in range(m)] for i in range(n)]

    def __len__(self):
        return len(self.G)

    def __str__(self):
        s=""
        for i in range(self.n):
            for j in range(i+1):
                print(i,j)
                s+= str(G[i,j]) + ", "
            if i!= len(self)-1: s+="\n"
        return(s)

    def get_block_validity(self,i):
        """
        Returns True if Mu[i][j] and Rr[i][j] are up-to-date for all j
        param i: position < n
        """
        return (self.valid_blocks>>i)%2==1

    def set_block_validity(self,i,state):
        """
        Remembers if Mu[i][j] and Rr[i][j] are up-to-date for all j.
        param i: position < n
        param state: if True, marks Mu[i][j] and Rr[i][j] as valid for given i, else - marks those as non-valid
        """
        if state==self.get_block_validity(i):
            return
        if state:
            self.valid_blocks+=1<<i
        else:
            self.valid_blocks-=1<<i

    def update_G(self, B, start):
        """
        Updates G starting at start.
        param B: list of nf_vect - module basis.
        param G: position <n.
        """
        for i in range(start, self.n):
            if self.get_block_validity(i):
                continue
            for j in range(i+1):
                self.G[i][j] = B[i].hermitian_inner_product(B[j])
            self.set_block_validity(i,True)

    def compute_GSO(self,start=0, end=None):
        """
        Modifies GSO coefficients Mu and Rr at positions start,...,end-1.
        param start: position to start modifying Mu and Rr
        param end: position to end modifying Mu and Rr

        The algorithm is an adaptation of [https://perso.ens-lyon.fr/damien.stehle/downloads/fpLLL_journal.pdf].
        Notice that <au,bv> is a*b.conjugate()*<u,v>.
        """
        n = self.n
        if end is None:
            end=n
        assert end<=n, "Wrong dimensions!"
        assert start<end, "start>=end!"

        #self.Mu[0][0] = nf_elem( vector(CC,[1 for i in range(self.d)]) )
        for i in range(start,end):
            for j in range(i):
                self.Rr[i][j] = self.G[i][j]
                for k in range(j):
                    self.Rr[i][j] -= self.Mu[j][k].conjugate()*self.Rr[i][k]
                self.Mu[i][j] = self.Rr[i][j] / self.Rr[j][j]
            self.Rr[i][i]= self.G[i][i]
            for j in range(i):
                self.Rr[i][i]-=self.Mu[i][j].conjugate()*self.Rr[i][j]

    def unit_reduce(self, FI,B,U, start=0, end=0, debug=0):
        """
          This function returns set of units that reduce B_start,...,B_{end-1} where B is the basis of lattice.
          param FI: FieldInfo object
          param B: list of nf_vect - module basis.
          param T: list of nf_vect - current transformation matrix.
          param start: position to start at.
          params end: position to end before.

          Updates U, self and B s.t. B\' = diag(1,...,U,...,1)*B entries starting at start and ending at end-1 are unit reduce.
        """
        n = self.n
        K=FI.Field

        for i in range(start,end):
            u = self.Rr[i][i].get_close_unit(FI)
            rii = self.Rr[i][i]*u
            if rii.canon_norm()>self.Rr[i][i].canon_norm():
                if debug&debug_flags.verbose_unit:
                    print(f"{bcolors.WARNING}Unit was about to make it worse! Worsening: {(rii.canon_norm()/self.Rr[i][i].canon_norm()).n(33)} {bcolors.ENDC}")
                continue
            if debug&debug_flags.verbose_unit:
                print(f"{bcolors.OKGREEN}Unit done something! Improvement: {(self.Rr[i][i].canon_norm()/rii.canon_norm()).n(33)} {bcolors.ENDC}")
            B[i] *= u
            U[i] *= u
            uc = u.conjugate()
            for j in range(i):
                self.G[i][j]=u*self.G[i][j]
            for l in range(i+1,n):
                self.G[l][i]=uc*self.G[l][i]
            self.G[i][i]=u*uc*self.G[i][i]
        self.compute_GSO(start=start,end=end)

    def size_reduce(self, FI,B,U, start=0, end=0, debug=0):
        """
          This function returns transform matrix that size reduce B_start,...,B_{end-1} where B is the basis of lattice.
          param FI: FieldInfo object
          param B: basis to be updated
          param U: transform matrix to be updated
          param start: position to start at.
          params end: position to end before.
          Updates U, self and B s.t. U*B entries starting at start and ending at end-1 are size reduced and self is valid from 0 to end-1.
        """
        n = self.n
        K = FI.Field
        for i in range(start,end):
            for j in range(i-1,-1,-1):
                delta_nf = self.Mu[i][j].roundoff_fft(K)
                mu_ = self.Mu[i][j] - delta_nf

                if mu_.canon_norm() <= self.Mu[i][j].canon_norm():
                    if debug&debug_flags.verbose_size_reduction:
                        print(f"{bcolors.OKGREEN}Size reduction OK! Improvement: {self.Mu[i][j].canon_norm()/mu_.canon_norm()} {bcolors.ENDC}")
                    dnf = delta_nf.conjugate()
                    gi = [self.G[i][kappa] for kappa in range(i)]
                    gii = [self.G[kappa][i] for kappa in range(i+1,n)]
                    gij, gjj, gji = self.G[i][j], self.G[j][j], self.G[i][j].conjugate()
                    for kappa in range(i):
                        gi[kappa] -=  delta_nf*self.G[j][kappa] if j>=kappa else delta_nf*self.G[kappa][j].conjugate()
                    for kappa in range(i+1,n):
                        gii[kappa-i-1] -=  dnf*self.G[kappa][j] if j<=kappa else dnf*self.G[j][kappa].conjugate()
                    self.G[i][i] -= ( delta_nf*gji+dnf*gij-delta_nf*dnf*gjj )
                    for kappa in range(i):
                        self.G[i][kappa] = gi[kappa]
                    for kappa in range(i+1,n):
                        self.G[kappa][i] = gii[kappa-i-1]
                    U[i] -= delta_nf*U[j]
                    B[i] -= delta_nf*B[j]
                    self.compute_GSO( start=i, end=i+1 )
                elif debug&debug_flags.verbose_size_reduction:
                    print(f"{bcolors.WARNING}Size reduction was about to make it worse! Worsening: {mu_.canon_norm()/self.Mu[i][j].canon_norm()} {bcolors.ENDC}")

    def update_after_svp_oracle_rank_2(self,U_,i):
        """
        After SVP oracle in lll is done, updates self.
        param U_: transformation matrix (list of nf_vect) returned by SVP oracle
        param i: position at which the SVP oracle has been called
        """
        n=self.n
        u0,u1 = U_
        g0, g1, g2 = self.G[i][i], self.G[i+1][i+1], self.G[i+1][i]

        """
        For the both rows (zone I) we have that:
            <b'[i],b'[ell]> = U_[0][0]*<b[i],b[ell]> + U_[0][1]*<b[i+1],b[ell]>
            <b'[i+1],b'[ell]> = U_[1][0]*<b[i],b[ell]> + U_[1][1]*<b[i+1],b[ell]>
        """
        for ell in range(i):
            tmp = U_[0][0]*self.G[i][ell] + U_[0][1]*self.G[i+1][ell]
            self.G[i+1][ell] = U_[1][0]*self.G[i][ell] + U_[1][1]*self.G[i+1][ell]
            self.G[i][ell] = tmp

        """
        For the both columns (zone III) we have that:
            <b'[kappa],b'[i]>   = U_[0][0].conjugate()*<b[kappa],b[i]> + U_[0][1].conjugate()*<b[kappa],b[i+1]>
            <b'[kappa],b'[i+1]> = U_[1][0].conjugate()*<b[kappa],b[i]> + U_[1][1].conjugate()*<b[kappa],b[i+1]>
        """
        for kappa in range(i+2,n):
            tmp =                U_[0][0].conjugate()*self.G[kappa][i] + U_[0][1].conjugate()*self.G[kappa][i+1]
            self.G[kappa][i+1] = U_[1][0].conjugate()*self.G[kappa][i] + U_[1][1].conjugate()*self.G[kappa][i+1]
            self.G[kappa][i] = tmp

        u0_ = ( u0[0].conjugate(), u0[1].conjugate() )
        u1_ = ( u1[0].conjugate(), u1[1].conjugate() )
        g2_ = g2.conjugate()
        # the rest is dealt with with similar formulas (see overleaf)
        self.G[i][i] =      u0[0]*u0_[0]*g0 + u0[0]*u0_[1]*g2_+u0[1]*u0_[0]*g2+u0[1]*u0_[1]*g1
        self.G[i+1][i+1] =  u1[0]*u1_[0]*g0 + u1[0]*u1_[1]*g2_+u1[1]*u1_[0]*g2+u1[1]*u1_[1]*g1
        self.G[i+1][i] =    u1[0]*u0_[0]*g0 + u1[0]*u0_[1]*g2_+u1[1]*u0_[0]*g2+u1[1]*u0_[1]*g1

        for ii in range(i,n):
            self.set_block_validity(ii,False)

    def check_consistency(self, B, start=0, end=1):
        """
        Checks:
        1) If Mu factor of B is equal to self.Mu
        2) If Rr factor of B is equal to self.Rr
        3) If Mu[i][j] == Rr[i][j] / Rr[j][j]
        param B: up-to-date basis of module (consists of nf_vect)
        param start: position to start at.
        params end: position to end before.
        """
        n = len(self.G)
        m = n
        self.compute_GSO(start=0,end=len(self))
        G = GSO_obj(B)
        G.compute_GSO(start=0, end=end)

        Mu, Rr = G.Mu, G.Rr
        Mu_, Rr_ = self.Mu, self.Rr
        for i in range(end):
            for j in range(i):
                if abs( Mu[i][j].elem - Mu_[i][j].elem ) > global_variables.threshold:
                    print(f"{bcolors.WARNING}Non consistent GSO at Mu_{i,j}:{bcolors.ENDC} {abs( Mu[i][j].elem - Mu_[i][j].elem ).n() }")

        for i in range(start, end):
            for j in range(i+1):
                if Rr[i][j] is None:
                    continue
                else:
                    if abs( Rr[i][j].elem - Rr_[i][j].elem ) > global_variables.threshold:
                        print(f"{bcolors.WARNING}Non consistent GSO at Rr_{i,j}:{bcolors.ENDC} {abs( Rr[i][j].elem - Rr_[i][j].elem ).n() }")

        for i in range(start, end):
            for j in range(i):
                tmp0 = inv_minkowski_embedding( Rr[i][j] / Rr[j][j] )
                tmp = inv_minkowski_embedding( Mu[i][j] )
                if abs(tmp-tmp0)>global_variables.threshold:
                    print(f"{bcolors.WARNING}Warning:{bcolors.ENDC} R/R={abs(ln(tmp))} at {i,j}")

    def check_integrity( self,B,start=0,end=0 ):
        """
        Checks if Gram matrix of self is equal to the Gram matrix of B
        param B: up-to-date basis of module (consists of nf_vect)
        param start: position to start at.
        params end: position to end before.
        """
        H = GSO_obj(B)
        for i in range(start,end):
            for j in range(len(self.G[i])):
                diff = self.G[i][j].elem - H.G[i][j].elem
                if  abs(diff) > global_variables.threshold:
                    print(f"{bcolors.WARNING}Troubles at {i,j},{bcolors.ENDC} delta={diff.n(50)}")

    def check_gs_vect( self, B, B_star_for_check, start=0,end=0 ):
        """
        Checks if B_star_for_check are gram schmidt vectors for B.
        param B: list of nf_vect objects
        param B_star_for_check: list of nf_vect objects
        param start: position to start at.
        params end: position to end before.
        """
        G = GSO_obj(B)
        G.compute_GSO(start=0, end=end)
        n = len(B)
        B_star = [None for i in range(n)]
        for i in range(end):
            B_star[i] = B[i] - sum( G.Mu[i][k]*B_star[k] for k in range(i) )
            tmp = (B_star_for_check[i]-B_star[i]).canon_norm()
            if tmp>global_variables.threshold:
                print(f"{bcolors.WARNING}Troubles at {i}th vector! {bcolors.ENDC}{tmp}")
            tmp = B_star_for_check[i].hermitian_inner_product(B_star_for_check[i])
            tmp1 = abs((tmp-G.Rr[i][i]).elem)
            if tmp1>global_variables.threshold:
                print(f"{bcolors.WARNING}Troubles at Rr_ii {i}! {bcolors.ENDC} {tmp1}")

        for i in range(1,end):
            for j in range(i):
                tmp = B_star_for_check[i].hermitian_inner_product(B_star_for_check[j])
                if norm(tmp.elem)>global_variables.threshold:
                    print(f"{bcolors.WARNING} b*_{i} and b*_{j} are non-orthogonal: {norm(tmp.elem)} {bcolors.ENDC}")

    def check( self,B,debug, B_star_for_check=None, start=0, end=0  ):
        if debug&debug_flags.check_integrity:
            self.check_integrity( B,start=0,end=end )
        if debug&debug_flags.check_consistency:
            self.check_consistency( B, start=0, end=end )
        if debug&debug_flags.check_gs_vect:
            self.check_gs_vect( B, B_star_for_check, start=0,end=end )

    def det_squared( self ):
        t = -1
        for i in range(self.n):
            if not self.get_block_validity(i):
                self.compute_GSO(start=i,end=self.n)
                break
        return prod( self.Rr[i][i] for i in range(self.n) )

class FieldInfo:
    #Class that wraps the information about pow-of-2 cyclotomic field, its units and the Log-unit lattice.
    def __init__(self, h):
        self.Field=CyclotomicField(2 **h)
        d = self.Field.degree()

        G, units_old = compute_log_unit_lattice(self.Field)
        T = G.U

        print("Computing Log-unit...")
        units_new = [ prod(units_old[j]**int(T[i][j]) for j in range(d/2-1)) for i in range(d/2-1) ]

        self.LLL_GSO = G
        self.cyclotomic_units = units_new

def round_babai(L_i,t):
    """
    Given FieldInfo object L_i, finds closest to t vector of Log-lattice of field L_i.Field for t - the vector of floats.
    param L_i: appropriate FieldInfo object
    param t: nf_elem object
    """
    T = vector( [ tt*2**global_variables.scale_factor_log_unit for tt in t ] )
    M = L_i.LLL_GSO

    tmp = M.babai(T)
    v= vector(ZZ, tmp )
    uts = L_i.cyclotomic_units
    return prod(uts[i]**-v[i] for i in range(len(uts)))

def compute_log_unit_lattice(K, debug=False):
    """
    Computes log unit lattice and the generating set of cyclotomic units for field K.
    param K: CyclotomicField
    param debug: reserved if any kind of debug is needed.
    """
    z_ = K.gen()
    d = K.degree()

    assert d>1, "Log unit lattice is not defined for QQ!"
    units = [ nf_elem( minkowski_embedding( K(i*[1]) ) ) for i in invertibles(d)[1:] ]

    d2 = K.degree()/2
    B=matrix([
        units[i].log_unit_projection() * 2**global_variables.scale_factor_log_unit for i in range( d2-1 )
    ])

    Bint = IntegerMatrix(d2-1 ,d2)

    for i in range( d2-1 ):
        for j in range( d2 ):
            Bint[i,j]=int( round(B[i,j])  )

    T = IntegerMatrix.identity(d2-1)
    G = GSO.Mat(Bint,float_type='mpfr',U=T)

    G.update_gso()
    lll_ = LLL_FPYLLL.Reduction(G, delta=0.999, eta=0.501)  #this updates G
    lll_()
    G.update_gso()

    return (G, units)

def invertibles(f):
    """
    Returns all 0<=i<f tat are coprime with f.
    param f: integer
    """
    out=[0  for i in range(euler_phi(f))]

    t=0
    for i in range(f):
        if gcd(i,f)==1 :
            out[t]=i
            t+=1
    return out

def size_unit_reduce_internal(B,FI,start=0,end=-1):
    """
    Performs size and unit reduction on basis B between start and end-1.
    param B: matrix over CyclotomicField
    param FI: appropriate FieldInfo object
    Returns corresponding unimodular transformation.
    Needed in BezTransform to reduce the size of transformation, or if one wants to preprocess the basis.
    """
    n = B.nrows()
    if end<0:
        end = n

    K = FI.Field
    d = K.degree()
    B_ = [ nf_vect( [minkowski_embedding(bb) for bb in b] ) for b in B ]
    G = GSO_obj( B_ )
    G.compute_GSO( start=0,end=end )    #compute Mu's, Rr's and the Gram matrix.

    zero_elem = vector(CC, [ 0 for ii in range(d/2) ])
    U = [ nf_vect([zero_elem]*n) for ii in range(n) ]
    one = minkowski_embedding(K(1))
    for ii in range(n):     #initialize the transformation matrix.
        U[ii].v[ii]=one

    G.unit_reduce( FI,B_,U, start=start, end=end)   #size reduce for start<=i<end
    G.size_reduce( FI,B_,U, start=start, end=end)  #unit reduce for start<=i<end

    return [
        u.to_number_field_round(K) for u in U
    ]   #return transformation in time domain
