from common_params import *

class LLL_params:

        def __init__(self,**kwargs):
            """
            Intitializes strategy and corresponding parameters.
            kwarg rho: amount of Lovàsz condition checks. Relevant if early_abort_niters==True
            kwarg rho_sub: amount of Lovàsz condition checks in the recursive calls of lll_fft.
            kwarg gamma: expected approx factor of the SVP oracle.
            kwarg gamma_sub: expected approx factor of the SVP oracle in recursive lll_fft call.
            kwarg bkz_beta: block size of BKZ algorithm used as the SVP oracle.
            kwarg svp_oracle_threshold: maximal degree of a field we're allowed to call BKZ on. Always reached due to descend.
            kwarg use_coeff_embedding: if True, use coefficien embedding for the oracle, else - use canonical.
            kwarg debug: computes internal info.
            kwarg verbose: verboses internal info.
            kwarg dump_intermediate_basis: dumps current basis after each tour
            kwarg bkz_r00_abort: aborts bkz as soon as r_0_0 decreased.
            kwarg use_custom_idealaddtoone: use our implenentation of pari gp's idealaddtoone.
            kwarg first_block_beta: if >0 the first SVP oracle call is called with specified block size
            kwarg use_pip_solver: use pip after first reduction
            kwarg experiment_name: part of a name of the file intermediate basis will be dumped to
            """
            self.params = dict()
            for key in kwargs.keys():
                self.params[key] = kwargs[key]
            default_params =(
                    ("rho",12), ("rho_sub",12), ("gamma",0.2), ("gamma_sub",0.16), ("bkz_beta",20), ("svp_oracle_threshold",64),
                    ("use_coeff_embedding",False), ("debug",0), ("verbose",True), ("early_abort_niters",False), ("early_abort_r00_decrease",False),
                    ("dump_intermediate_basis",False), ("bkz_r00_abort",False), ("use_custom_idealaddtoone",True), ("first_block_beta",0), ("use_pip_solver",True),
                    ("experiment_name", randrange(2^32))
                )
            for k in default_params:
                if not k[0] in kwargs.keys():
                    self.params[k[0]] = k[1]

        def __str__(self):
            s = ""
            for key in self.params.keys():
                s += f"{str(key)}: {str( self.params[key] )}" + ", "
            s = s[:len(s)-2]
            return s

        def __getitem__(self,key):
            if key in self.params.keys():
                return self.params[key]
            return None

        def __setitem__(self,key,value):
            self.params[key] = value

        def set_precision(Prec_user=global_variables.Prec, fplllPrec_user=global_variables.fplllPrec, bkz_scaling_factor_user=global_variables.bkz_scaling_factor):
            global_variables.Prec=Prec_user
            global_variables.fplllPrec=fplllPrec_user
            global_variables.bkz_scaling_factor=bkz_scaling_factor_user

        def overstretched_NTRU( conductor,q,descend_number=0, beta=20, first_block_beta=0, early_abort_niters=False ):
            """
            Default strategy for reducing NTRU lattices.
            param f: K's conductor where K is CyclotomicField NTRU is defined over.
            param q: NTRU modulus.
            param descend_number: number of times the recursive descend is applied.
            param beta: BKZ beta for SVP oracle.
            """
            d = conductor/2

            # make sure bkz_scaling_factor_user is less than mpfr_usage_threshold_prec with some gap, otherwise the approxCVP on log-unit lattice in BezTransform
            # might crash
            global_variables.log_basis_degradation_factor = 20.0
            if d<=128 and q<=2^18:
                LLL_params.set_precision(Prec_user=144, fplllPrec_user=global_variables.ld_usage_threshold_prec, bkz_scaling_factor_user=50)
            elif d<=256 and q<=2^32:
                LLL_params.set_precision(Prec_user=192, fplllPrec_user=global_variables.mpfr_usage_threshold_prec, bkz_scaling_factor_user=80)
            else:
                tmp = round( len(bin(q))/2 )
                LLL_params.set_precision(Prec_user=192+tmp, fplllPrec_user=256, bkz_scaling_factor_user=160)
            p =  LLL_params( rho=9, rho_sub=9, gamma=0.22, gamma_sub=0.19,
                svp_oracle_threshold=d/2**descend_number, bkz_beta=beta, early_abort_niters=early_abort_niters, first_block_beta=first_block_beta,
                debug = debug_flags.verbose_anomalies | debug_flags.verbose_summary )
            p["use_pip_solver"] = True
            return p

        def LWE( f,q,descend_number=0, beta=35, gamma=0.22, gamma_sub=0.19, early_abort_r00_decrease=False ):
            d = f/2**(1+descend_number)

            if d<256 and q<=2^22:
                LLL_params.set_precision(Prec_user=144, fplllPrec_user=global_variables.ld_usage_threshold_prec, bkz_scaling_factor_user=80)
            elif d<512 and q<=2^32:
                LLL_params.set_precision(Prec_user=192+ceil(log(q,2)), fplllPrec_user=global_variables.mpfr_usage_threshold_prec, bkz_scaling_factor_user=110)
            else:
                LLL_params.set_precision(Prec_user=192+ceil(log(q,2)), fplllPrec_user=256, bkz_scaling_factor_user=144)
            return LLL_params( gamma=gamma, gamma_sub=gamma_sub, early_abort_r00_decrease=early_abort_r00_decrease,
                svp_oracle_threshold=d/2**descend_number, bkz_beta=beta )
