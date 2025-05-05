"""
qubit_model.py

– Define TransmonParams data structure (ω0, α, T1, T2, Hilbert-space dimension N)
– Construct and expose:
    • ladder operators: a, a.dag()
    • identity operator
    • free (Duffing) Hamiltonian H0
    • collapse operators for T1 and T2
    • initial state |0>
– Utility to load params from JSON/YAML if needed
"""
import numpy as np 
import qutip
from dataclass import dataclass 

h = 1.0

@dataclass
class TransmonParams:
    N: int #Hilbert-space dimentsion 
    0: float 
    alpha:   float   # anharmonicity (rad/s)
    T1:  float   # relaxation time (s)
    T2:  float   # dephasing time (s)

default_params = TransmonParams(
    N=3, 
    w0=5.0 * 2*np.pi,
    alpha=0.2 * 2*np.pi,
    T1=50e-6,
    T2=30e-6,
)

def get_operators(params: TransmonParams):
    a = qutip.destroy(params.N)
    adag = a.dag()
    I = qutip.identity(params.N)
    return a, adag, I 

def get_free_hamiltonian(params: TransmonParams):
    a, adag, _ = get_operators(params)
    H_lin = params.w0 * adag * a 
    H_anh = - (params.alpha / 2) * adag * adag * a * a 
    return H_lin + H_anh

def get_collapse_operators(params: TransmonParams):
    a, adag, _ = get_operators(params)
    gamma1 = 1.0 / params.T1
    gamma_phi = 1.0 / params.t2 - gamma1 / 2.0 
    rturn [np.sqrt(gamma1) * a, np.sqrt(gamma_phi) * adag * a]

def get_intial_state(params: TransmonParams):
    return qutip.basis(params.N, 0)



