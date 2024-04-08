import os
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinful_fermion_basis_1d # bosonic Hilbert space
from quspin.tools.measurements import diag_ensemble
from scipy.sparse.linalg import expm_multiply
from quspin.tools.lanczos import lanczos_full,lanczos_iter,lin_comb_Q_T,expm_lanczos

import numpy as np # general math functions
import matplotlib.pyplot as plt # plotting library
import math
import cmath
import itertools as it
import scipy.linalg as sla

#from time import time # timing package
#ti = time() # start timer

no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)
#
##### define model parameters
# initial seed for random number generator
np.random.seed(0) # seed is 0 to produce plots from QuSpin2 paper
# setting up parameters of simulation
L = 2       # length of chain
Ns = 2*L+1   # number of sites
N = Ns+2      # number of sites with leads

t_AA = -1.0  # AA hopping
t_AB = np.sqrt(2.0)*t_AA # AB hopping
t_SL = 1.0
t_SR = 1.0
#mu = 0.7
mu = 0.0
#DeltaL = 100.0*cmath.exp(0.5*3.14159265*1.0j)
DeltaL = 100.0*cmath.exp(0.5*3.14159265*1.0j)
DeltaR = 100.0
U = -1.0     # interaction strength
VB = 0.0 	 # boundary potential



##### set up Hamiltonian and observables
# define site-coupling lists
E_u_list = [[U+VB-mu,1,1],[U+VB-mu,N-2,N-2]] # boundary potential
E_u_list.extend([[U-mu,i,i] for i in range(2,N-2)])
E_u_list.extend(([0.0,0,0],[0.0,N-1,N-1]))

E_d_list = [[-VB+mu,1,1],[-VB+mu,N-2,N-2]] # boundary potential
E_d_list.extend([[mu,i,i] for i in range(2,N-2)])
E_d_list.extend(([0.0,0,0],[0.0,N-1,N-1]))

Deltadu_list = [[-np.conj(DeltaL),0,0],[-np.conj(DeltaR),N-1,N-1]]
Deltaud_list = [[DeltaL,0,0],[DeltaR,N-1,N-1]]

int_list = [[-U,i,i] for i in range(1,N-1,1)]  # on-site interactions within lattice
int_list.extend(([0.0,0,0],[0.0,N-1,N-1]))

# setting up hopping lists
hop_u_left = [[t_AA,i,(i+2)] for i in range(1,N-2,2)] # AA hopping left
hop_u_right = [[-t_AA,i,(i+2)] for i in range(1,N-2,2)] # AA hopping right
hop_u_left.extend([[t_AB,i,(i+1)] for i in range(1,N-2,1)]) # AB hopping left
hop_u_right.extend([[-t_AB,i,(i+1)] for i in range(1,N-2,1)]) # AB hopping right

hop_u_left.extend([[np.conj(t_SL),0,1],[np.conj(t_SR),N-2,N-1]]) # lead hopping 1
hop_u_right.extend([[-t_SL,0,1],[-t_SR,N-2,N-1]]) # lead hopping 2

hop_d_left = [[-t_AA,i,(i+2)] for i in range(1,N-2,2)] # AA hopping left
hop_d_right = [[t_AA,i,(i+2)] for i in range(1,N-2,2)] # AA hopping right
hop_d_left.extend([[-t_AB,i,(i+1)] for i in range(1,N-2,1)]) # AB hopping left
hop_d_right.extend([[t_AB,i,(i+1)] for i in range(1,N-2,1)]) # AB hopping right

hop_d_left.extend([[-np.conj(t_SL),0,1],[-np.conj(t_SR),N-2,N-1]]) # lead hopping 1
hop_d_right.extend([[t_SL,0,1],[t_SR,N-2,N-1]]) # lead hopping 2

#hop_list_hc = [[J.conjugate(),i,j] for J,i,j in hop_list] # add h.c. terms

# setting up local Fock basis
# the particle+hole number is conserved and goes from 0 to 2*N, with 2^(2N) states in total

# non-interacting Hamiltonian
static = [
        ["+-|",E_u_list],      # on_site energies up
        ["|+-",E_d_list],      # on_site energies down
        ["+-|",hop_u_left],    # hopping up
        ["-+|",hop_u_right],    # hopping up h.c.
        ["|+-",hop_d_left],    # hopping down
        ["|-+",hop_d_right],    # hopping down h.c.
        ["-|+",Deltadu_list], # lead Delta up
        ["+|-",Deltaud_list], # lead Delta down
        ["n|n",int_list] # interaction
        ]

n_mus = 100
n_phis = 100
mus = np.linspace(-4.2,2.0,n_mus)
phis = np.linspace(0,2*np.pi,n_phis)

nvsmu = np.zeros(n_mus)
Ivsmu = np.zeros(n_mus)
Nf_minvsmu = np.zeros(n_mus)

Ivsphi = np.zeros(n_phis)
for i in range(n_mus):
    mu = mus[i]

    # define site-coupling lists
    E_u_list = [[U+VB-mu,1,1],[U+VB-mu,N-2,N-2]] # boundary potential
    E_u_list.extend([[U-mu,i,i] for i in range(2,N-2)])
    E_u_list.extend(([0.0,0,0],[0.0,N-1,N-1]))

    E_d_list = [[VB-mu,1,1],[VB-mu,N-2,N-2]] # boundary potential
    E_d_list.extend([[U-mu,i,i] for i in range(2,N-2)])
    E_d_list.extend(([0.0,0,0],[0.0,N-1,N-1]))

    static = [
            ["+-|",E_u_list],      # on_site energies up
            ["|+-",E_d_list],      # on_site energies down
            ["+-|",hop_u_left],    # hopping up
            ["-+|",hop_u_right],    # hopping up h.c.
            ["|+-",hop_d_left],    # hopping down
            ["|-+",hop_d_right],    # hopping down h.c.
            ["-|+",Deltadu_list], # lead Delta up
            ["+|-",Deltaud_list], # lead Delta down
            ["n|n",int_list] # interaction
            ]
    E_GS_min = 1000
    I_min = 0
    n_min = 0
    Nf_min = 0

    # cases with Nf = 0 or Nf = 2N are not interesting because then nothing happens (empty, full)
    for Nf in range(1,2*N,1):
        nums = [(j,Nf-j) for j in range(max(Nf-N,0),min(N,Nf)+1,1)]
        basis = spinful_fermion_basis_1d(L=N,Nf=nums)
        print(basis.Ns)
        H = hamiltonian(static,[],basis=basis,dtype=np.complex128,check_pcon = False)

        #I_SL = hamiltonian([["+-|",[[1.0j*t_SL,1,0]]], ["+-|",[[-1.0j*t_SL,0,1]]],["|+-",[[-1.0j*t_SL,1,0]]], ["|+-",[[1.0j*t_SL,0,1]]]],[],basis=basis,dtype=np.complex128,**no_checks)
        I_SL = hamiltonian([["+-|",[[1.0j*t_SL,1,0]]], ["+-|",[[-1.0j*t_SL,0,1]]]],[],basis=basis,dtype=np.complex128,**no_checks)

        n_op = hamiltonian([["+-|",[[1.0,i+1,i+1] for i in range(Ns)]],["|+-",[[-1.0,i+1,i+1] for i in range(Ns)]]],[],basis=basis,dtype=np.complex128,**no_checks)

        E_GS,psi_GS = H.eigsh(k=1,which="SA")
        print(Nf,E_GS)
        if(E_GS < E_GS_min):
            E_GS_min = E_GS
            I_min = I_SL.expt_value(psi_GS).real
            n_min = n_op.expt_value(psi_GS).real+N
            Nf_min = Nf
        
    Ivsmu[i] = I_min
    nvsmu[i] = n_min
    Nf_minvsmu[i] = Nf_min


#for i in range(Ns):
#    Delta_A1 = hamiltonian([["-|-",[[U,i+1,i+1]]]],[],basis=basis_even,dtype=np.complex128,**no_checks)
#    expt_Delta_A1 = Delta_A1.expt_value(psi_GS_even)
#    print([abs(expt_Delta_A1),cmath.phase(expt_Delta_A1)])

    #Ivsphi[i] = expt_I_SL

plt.figure()
plt.plot(mus,Ivsmu)
plt.xlabel("Chemical potential")
plt.ylabel("Current")

plt.figure()
plt.plot(mus,nvsmu)
plt.xlabel("Chemical potential")
plt.ylabel("Particle number")

plt.figure()
plt.plot(mus,Nf_min)
plt.xlabel("Chemical potential")
plt.ylabel("N_f min")
plt.show()
#plt.figure()
#plt.plot(mus,Ivsphi)
#plt.xlabel("Phase twist")
#plt.ylabel("Current")
#plt.show()





# diagonalise Hamiltonian
#E_even,V_even = H_even.eigh()
#E_odd,V_odd = H_odd.eigh()
#print("Eigenenergies", E)
## initial state
#psi = V[:,-1]
#
## setting up observables
#n_list = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.complex128,**no_checks) for i in range(N)]
##I = commutator(H, n_list[4])
#it_AA = 1.0j*t_AA
#it_AB = 1.0j*t_AB
#
#
##I_A2A3 = hamiltonian([["+-",[[1.0j*t_AA,N-1,N-3]]], ["+-",[[-1.0j*t_AA,N-3,N-1]]]],[],basis=basis,dtype=np.complex128,**no_checks)
##I_B2A3 = hamiltonian([["+-",[[1.0j*t_AB,N-1,N-2]]], ["+-",[[-1.0j*t_AB,N-2,N-1]]]],[],basis=basis,dtype=np.complex128,**no_checks)
##I_A2A3 = hamiltonian([["+-",[[t_AA,4,2]]], ["+-",[[-t_AA,2,4]]]],[],basis=basis,dtype=np.complex128,**no_checks)
##I_B2A3 = hamiltonian([["+-",[[t_AB,4,3]]], ["+-",[[-t_AB,3,4]]]],[],basis=basis,dtype=np.complex128,**no_checks)
#
#
## setting up parameters for evolution
##start,stop,num = 0,100,2501 # 0.1 equally spaced points
#start,stop,num = 0,400,5001 # 0.1 equally spaced points
#times = np.linspace(start,stop,num)
#n_VBs = 50
#VBs = np.linspace(0.0, 2.0, n_VBs)
#n_Vstate_A3 = np.zeros(n_VBs)
#curs_A2A3_Vstate = np.zeros(n_VBs)
#curs_B2A3_Vstate = np.zeros(n_VBs)
#curs_A2A3_Vstate_diag = np.zeros(n_VBs)
#curs_B2A3_Vstate_diag = np.zeros(n_VBs)
#n_A3_Vstate_diag = np.zeros(n_VBs)
#curs_total_Vstate = np.zeros(n_VBs)
#
#for i in range(n_VBs):
#    VB = VBs[i]
#    print(VB)
#    E_list = [[VB,0],[VB,N-1]] # boundary potential
#    # set up static and dynamic lists
#    static = [
#            ["+-",hop_list],    # hopping
#            ["-+",hop_list_hc], # hopping h.c.
#            ["++--",int_list],  # two-particle interaction
#            ["n",E_list]
#            ]
#    dynamic = [] # no dynamic operators
#    # build real-space Hamiltonian
#    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128,**no_checks)
#    Es,Vs = H.eigh()
#    #print("Eigenvalue",Es)
#
#    ##### do time evolution
#    # calculating the evolved states
#    psi_t = H.evolve(psi,t0=times[0],times=times)
#
#    """
#    expt_I_A2A3 = np.zeros(num)
#    expt_I_B2A3 = np.zeros(num)
#    for j,psi in enumerate(psi_t):
#            expt_I_A2A3[j] = I_A2A3.expt_value(psi,time=times[j]).imag
#            expt_I_B2A3[j] = I_B2A3.expt_value(psi,time=times[j]).imag
#    """
#
#    # calculating the local densities as a function of time
#    expt_n_t = n_list[N-1].expt_value(psi_t).real
#    expt_I_A2A3 = I_A2A3.expt_value(psi_t).real
#    expt_I_B2A3 = I_B2A3.expt_value(psi_t).real
#
#    # Diagonal ensemble averages
#    Diag_ens_I_A2A3 = diag_ensemble(N,psi,Es,Vs,Obs=I_A2A3,delta_t_Obs=True)
#    Diag_ens_I_B2A3 = diag_ensemble(N,psi,Es,Vs,Obs=I_B2A3,delta_t_Obs=True)
#    Diag_ens_n_A3 = diag_ensemble(N,psi,Es,Vs,Obs=n_list[N-1],delta_t_Obs=True)
#
#
#    # mean particle number at A3
#    n_Vstate_A3[i] = np.mean(expt_n_t[math.floor(num/2):1:-1])
#    n_A3_Vstate_diag[i] = Diag_ens_n_A3['Obs_pure']
#    # RMS particle current
#    curs_A2A3_Vstate_diag[i] = Diag_ens_I_A2A3['Obs_pure']
#    curs_B2A3_Vstate_diag[i] = Diag_ens_I_B2A3['Obs_pure']
#
#    print(curs_A2A3_Vstate_diag[i])
#    print(curs_B2A3_Vstate_diag[i])
#    #curs_A2A3_Vstate[i] = np.sqrt(np.mean(expt_I_A2A3[math.floor(num/2):1:-1]**2))
#    #curs_B2A3_Vstate[i] = np.sqrt(np.mean(expt_I_B2A3[math.floor(num/2):1:-1]**2))
#    #curs_total_Vstate[i] = np.sqrt(np.mean((expt_I_A2A3[math.floor(num/2):1:-1]+expt_I_B2A3[math.floor(num/2):1:-1])**2))
#
#    curs_A2A3_Vstate[i] = np.abs(expt_I_A2A3[1])
#    curs_B2A3_Vstate[i] = np.abs(expt_I_B2A3[1])
#    curs_total_Vstate[i] =np.abs(expt_I_A2A3[1]+expt_I_B2A3[1])
#
#    #curs_A2A3_Vstate[Ui] = np.mean(np.abs(expt_I_A2A3))
#    #curs_B2A3_Vstate[Ui] = np.mean(np.abs(expt_I_B2A3))
#
#print("simulation took {0:.4f} sec".format(time()-ti))
#
## plotting static figures
#print("Plotting")
#
#plt.plot(VBs, n_Vstate_A3)
#plt.xlim(VBs[0],VBs[-1])
#plt.ylim(ymin=0)
#plt.xlabel("Boundary energy",fontsize=20)
#plt.ylabel("$n_\\mathrm{A3}$",fontsize=20)
#plt.savefig("particle_number_"+str(Nb)+"_"+str(VBs[0])+"-"+str(VBs[-1])+".png")
#
#plt.figure()
#
#plt.plot(VBs, curs_A2A3_Vstate,label="A2->A3")
#plt.plot(VBs, curs_B2A3_Vstate,label="B2->A3")
#plt.plot(VBs, curs_total_Vstate,label="total")
#plt.xlim(VBs[0],VBs[-1])
#plt.ylim(ymin=0)
#plt.xlabel("Boundary energy",fontsize=20)
#plt.ylabel("Particle current",fontsize=20)
#plt.legend()
#plt.grid()
#plt.savefig("current_"+str(Nb)+"_"+str(VBs[0])+"-"+str(VBs[-1])+".png")
#
#plt.figure()
#
#plt.plot(VBs, curs_A2A3_Vstate_diag,label="A2->A3")
#plt.plot(VBs, curs_B2A3_Vstate_diag,label="B2->A3")
#plt.xlim(VBs[0],VBs[-1])
#plt.ylim(ymin=0)
#plt.xlabel("Boundary potential",fontsize=20)
#plt.ylabel("Particle current",fontsize=20)
#plt.legend()
#plt.grid()
#
#plt.figure()
#
#plt.plot(VBs, n_A3_Vstate_diag)
#plt.xlim(VBs[0],VBs[-1])
#plt.ylim(ymin=0)
#plt.xlabel("Boundary potential",fontsize=20)
#plt.ylabel("Particle numbet at A_3",fontsize=20)
#
#plt.show()
#plt.close()
