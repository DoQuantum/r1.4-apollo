"""
control_hamiltonian.py

– Define control operator H1 = a + a.dag()
– Provide `eps_t(t, args)` callback:
    • reads SNN’s pulse sample buffer
    • optionally interpolates between discrete time‐steps
    • returns drive amplitude ε(t) for QuTiP time‐dependence
"""

import numpy as np 
from qubit_model import get_operators 

def get_control_operator(params):
    a, adag, _ = get_operators(params)
    return a + adag 

# connect sNN here 
eps_samples = None # sNN output will go here 
time_grid = None 

# calculates ε(t) from sNN data via nearest-neighbor lookup;
# swap in an interpolator here for smoother pulses
def eps_t(t, args):
    dt = time_grid[1] - time_grid[0]
    idx = int(np.clip(t // dt, 0, len(eps_samples)-1))
    return float(eps_samples[idx])


