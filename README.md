# Josephson Effect ED

Use exact diagonalization to study Josephson effect in case of lead order parameter $\Delta$ substantially larger than the system order parameter (no quasiparticle excitations at the leads, leads reflect fully).

Usage:

There are many ways to install the required QuSpin package. The standard way is to make an Anaconda environment by e.g.

`conda create --name josephsonED`

`conda install -n josephsonED matplotlib`

`conda install -n josephsonED -c weinbe58 omp  quspin`

The environment can be then activated or deactivated by 

`conda active josephsonED`

`conda deactivate`

Then you can run the codes by

`python3 example_name.py`

There are two versions of the code:

-Josephson_ED_varied_number.py consider the system in two independent blocks: the even particle (up spin + down spin particles)  number and the odd particle number. This version gives sensible results. Now it considers sawtooth lattice but one can also build other models

-Josephson_ED_constant_ph.py considers the system in many independent branches: on each the up spin particle number + down spin hole number is constant (conserved quantity within the considered Hamiltonian). This version does not work yet properly.



