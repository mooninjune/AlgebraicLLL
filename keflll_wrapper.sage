import keflll
Prec = 2**200

def butterfly(v_,s):
    return keflll.butterfly(v_,s)

def kef__sqrt_numfield(a):
    return keflll.kef__sqrt_numfield(a)

def kef__inv_sqrt_numfield(a):
    return keflll.kef__inv_sqrt_numfield(a)

def fast_mult(a,b):
    return keflll.fast_mult(a,b)

def roundoff(a):
    return keflll.roundoff(a)

def fast_inv(a):
    return keflll.fast_inv(a)

def fast_sqrt(a):
    return keflll.fast_sqrt(a)

def fast_hermitian_inner_product(u,v):
    return keflll.fast_hermitian_inner_product(u,v)

def alg_norm(a):
    return keflll.alg_norm(a)

def enorm_numfield(a):
    return keflll.enorm_numfield(a)

def mat_mult(A,B):
    return keflll.mat_mult(A,B)

def fast_mat_mult(A,B):
    return keflll.fast_mat_mult(A,B)

def enorm_vector_over_numfield(v):
    return keflll.enorm_vector_over_numfield(v)

def qrfact_herm(B, debug=False):
    return keflll.qrfact_herm(B, debug)

def rfact_herm(B, debug=False, Prec=Prec):
    return keflll.rfact_herm(B, debug, Prec)

def ascend(K,v):
    return keflll.ascend(K,v)

def descend(K,a):   #only for K - cyclotomic of power 2
    return keflll.descend(K,a)

def invertibles(f):
    return keflll.invertibles(f)

def ifft(v):
    return keflll.ifft(v)

def minkowski_embedding(a):
    return keflll.minkowski_embedding(a)

def inv_minkowski_embedding(s):
    return keflll.inv_minkowski_embedding(s)

def log_embedding(a):
    return keflll.log_embedding(a)

def inv_log_embedding(s):
    return keflll.inv_log_embedding(s)

def round_matrix(M):
    return keflll.round_matrix(M)

def round_babai(L_i,t):
    return keflll.round_babai(L_i,t)

def get_close_unit(a, L_i, debug=False):
    return keflll.get_close_unit(a, L_i, debug)

def size_reduce(R, M, L_i):
    return keflll.size_reduce(R, M, L_i)

def GEuclide(L, Lptr,a,b, debug=False):
    return keflll.GEuclide(L, Lptr,a,b, debug)

def GEuclide_pari(L, Lptr,a,b):
    return keflll.GEuclide_pari(L, Lptr,a,b)

def descend_rank2_matrix(K,B):
    return keflll.descend_rank2_matrix(K,B)

def BezTransform(L,Lptr,v,debug=False,use_custom_idealaddtoone=True):
    return keflll.BezTransform(L,Lptr,v,debug,use_custom_idealaddtoone)

def compute_log_unit_lattice(K, debug=False):
    return keflll.compute_log_unit_lattice(K, debug)

def gen_unimod_matrix(K,n, rng=[-3 ,3 ], density=0.75):
    return keflll.gen_unimod_matrix(K,n, rng, density)

def lll(L,Lptr,M, rho=3 , rho_sub=6 , eps=0.5 , alpha=0.1 , return_basis=False, debug=False, svp_oracle_threshold=32 , beta=10 , stats = None, task_id=None):
    return keflll.lll(L,Lptr,M, rho, rho_sub, eps, alpha, return_basis, debug, svp_oracle_threshold, beta, stats, task_id)
