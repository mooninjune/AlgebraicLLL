# Finding Dense Submodules with Algebraic Lattice Reduction
This repository contains the framework that reduces algebraic lattices defined over power-of-2 cyclotomic fields. It contains the implementation of an algebraic LLL algorithm described in the article "**Finding Dense Submodules with Algebraic Lattice Reduction**" - Alexander Karenin and Elena Kirshanova.
## Acknowledgment
Both authors are affiliated with:
- Cryptography Research Center, Technology Innovation Institute, Abu Dhabi, UAE
- I.Kant Baltic Federal University, Kaliningrad, Russia
# Requirements
 - [SageMath 9.8+](https://www.sagemath.org/)
 - Optionally install `multiprocess` by running `pip install multiprocess` in sage session. This would allow Mac users to run the code.
 - Optionally update the FPYLLL in sage so that it supports qd. See: **Manual update of fpylll and fplll inside Sagemath 9.0+** in  [fpylll](https://github.com/fplll/fpylll).
 - Optionally install [g6k](https://github.com/fplll/g6k) and make a docker container `Dockerfile.sagemath` to enable BKZ with sieving.
 - Optionally install [flatter](https://github.com/keeganryan/flatter).
# Description of files
The repository consists of the following files:
 - `l2_.sage` - main file that defines the interface between sage and the algorithm.
 - `common_params.sage` - file that stores default values for various constants used by the algebraic LLL algorithm.
 - `LLL_params.sage` - file that encompass strategies algebraic LLL acts according to.
 - `gen_lat.sage` - file that generates algebraic lattices of particular interest.
 - `GenRec.gp, GentrySzydlo.gp` - files with code that solves the Principal Ideal Problem from [BEF+17](https://alexgelin.fr/index_en.html)
 - `f64bnf.pkl, f128bnf.pkl` - precomputed bnfinit using [PariGP](https://pari.math.u-bordeaux.fr/) for 64-th and 128-th cyclotomic fields.
 - `Field_infos.pkl` precomputed Log-unit lattices for some fields.
 - `keflll.sage` - functions implementing ascending, descending and vector insertion.
 - `util_l2.sage` - implementation of fast arithmetic over number fields, fast linear algebra, size- and unit- reduction and Log-unit computations.
 - `utils.sage` - contains helping functions.
 - `svp_tools.sage` - SVP oracle and PIP interface.
 - `profile_tests.sage, ntru_experiments.sage, modfalcon_experiments.sage` - the experiments.
 - `arakelov.sage` - implementation of Arakelov random walk from [dBDPMW20]
 - `lllbkz_modfalcon.sage, lllbkz_ntru.sage` - fpylll reduction for NTRU and ModFalcon.
 - `pump.py` - bkz with sieving.

[BEF+17] - Jean-François. Biasse, Thomas Espitau, Pierre-Alain Fouque, Alexandre
G´elin, and Paul Kirchner. Computing generator in cyclotomic integer
rings: A subfield algorithm for the principal ideal problem in and appli-
cation to the cryptanalysis of a FHE scheme. In Advances in Cryptology–
EUROCRYPT 2017, pages 60–88, 2017
[dBDPMW20] - Koen de Boer, L´eo Ducas, Alice Pellet-Mary, and Benjamin Wesolowski.
Random self-reducibility of ideal-SVP via Arakelov random walks. In
Advances in Cryptology - CRYPTO 2020, pages 243–273, 2020
# How to use
Run `build.sh` in the downloaded directory. This will build all the necessary files.

The interaction with the program is organized through sage's interface. In order to import all necessary classes, just type:

    from l2_ import *
To generate a test 2x3 matrix `B` over 64-th cyclotomic field and LLL reduce it, type:


    from l2_ import *
	from gen_lat import gen_UBW

	f,q,n,m = 64, 1, 2, 3
	B = gen_UBW( f,q,n,m )

	lllpar = LLL_params( bkz_beta=16 )
	lllobj = L2( B,strategy=lllpar )

	U = lllobj.lll()

	B_red = lllobj.B
The result will be stored in `B_red`.
# Experiments
To run the experiments from Section 5.1, type:

     sage profile_tests.sage
To run the experiments from Section 5.2, type:

     sage ntru_experiment_dump.sage
To run the experiments from Section 5.3, type:

     sage modfalcon_experiment.sage

To run the experiments from Section 5.4, type:

     sage idealtest.sage

To get the runtimes of FPYLLL on ModFalcon and NTRU lattices, run files `lllbkz_modfalcon.sage` and `lllbkz_ntru.sage` respectively.
