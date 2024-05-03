import pickle

## Hack to import my own sage scripts
def my_import(module_name, func_name='*'):
    import os
    os.system('sage --preparse ' + module_name + '.sage')
    os.system('mv ' + module_name + '.sage.py ' + module_name + '.py')

    from sage.misc.python import Python
    python = Python()
    python.eval('from ' + module_name + ' import ' + func_name, globals())

def qprof( n, d, q, base=2 ):
    assert n%2 == 0, f"not even"
    r = (n/2)*[ d*log(q,base) ] + (n/2)*[ 0 ]
    return(r)

def ZGSA( n, d, q, lgab, base=2 ):
    assert n%2 == 0, f"not even"
    r = [ 0 for i in range(n) ]

    lgqd = d*log( q, base )
    nprime = 1/2 + lgqd / lgab
    slope = lgqd / (2*nprime)
    nprime = min(n/2,round( 1/2 + lgqd / lgab  ))
    print(f"nprime: {nprime}")
#     slope = (3*nprime-n/2-1)/(nprime*(nprime^2+1))
    for i in range(n/2-nprime+1):
        r[i] = lgqd
    for i in range(max(0,n/2-nprime+1), min(n,n/2+nprime-1)):
        r[i] = RR( lgqd - slope*( i-n/2+nprime ) )
    scale_fact = (n/2*lgqd - sum(r)) / (2*(min(n,nprime)-1))
    for i in range(max(0,n/2-nprime+1), min(n,n/2+nprime-1)):
        r[i] += scale_fact
    return(r)

my_import("profile_tests")

nthreads = 8
tests_per_q = 20
dump_public_key = False

# - - - f=small

f=64
k=4
qs = [ next_prime( ceil(2^tmp) ) for tmp in [20.0+i*5 for i in range(3)] ] * tests_per_q
beta=25

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
process_output( output, tests_per_q,f )

for q in list(set(qs)):
    with open(f'{f}_({q}, {tests_per_q-1})_newprofile.pkl', mode='rb') as file:
        data = pickle.load( file )

    K.<z> = CyclotomicField( f )
    n = len(data)
    d = K.degree()

    C = 2.6
    lgab = C * d/2*(1+log(d,2))

    r = ZGSA( n, d, q, lgab, base=2 )

    f_, g_ = K( [randrange(-1,2) for i in range(d)] ), K( [randrange(-1,2) for i in range(d)] )
    dense_dim = 4
    nrm = dense_dim * log( norm(f_*f_.conjugate() + g_*g_.conjugate()).n(), 2 )/2
    print( f"norm: {nrm}" )
    print( f"PT vol: {sum( sorted(r)[:dense_dim] ).n()}" )

    l = qprof( n, d, q )
    P = list_plot( data, color = "blue",legend_label="Practice",plotjoined=True, figsize=11, axes_labels=['$i$', '$\sqrt{r_{i,i}}$') + list_plot(r, color = "red", legend_label="AZGSA",plotjoined=True) + list_plot(l, color = "brown", legend_label="Initial",plotjoined=True)

    P.save_image( f'{f}_({q}, {tests_per_q-1})_profilecompare.png' )
