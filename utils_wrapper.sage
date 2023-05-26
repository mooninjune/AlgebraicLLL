import utils

def embed_Q(B, scale_for_bkz=True):
    return utils.embed_Q( B,scale_for_bkz=scale_for_bkz )

def roundoff(a):
  return utils.roundoff(a)

def butterfly(v_,s):
    return utils.butterfly(v_,s)

def ifft(v,real_value=True):
    return utils.ifft(v,real_value=True)

def minkowski_embedding(a):
    return utils.minkowski_embedding(a)

def minkowski_embedding_vector(a):
    return minkowski_embedding_vector(a)

def inv_minkowski_embedding(s):
    return utils.inv_minkowski_embedding(s)

def randrange_not_null(a,b):
    return utils.randrange_not_null(a,b)

def compare_sage_versions(ver0, ver1):
    return utils.compare_sage_versions(ver0, ver1)

def scale_matrix(M):
    return utils.scale_matrix(M)


def bkz_reduce_fractional_matrix(B, block_size, verbose=False, dump=True):
    return utils.bkz_reduce_fractional_matrix(B, block_size, verbose=False, dump=True)

def nfhnf_pari(A, Is, U=None):
    return utils.nfhnf_pari(A, Is, U)
