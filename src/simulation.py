"""
simulation.py

– Import H0, H1, collapse operators, initial state
– Assemble time‐dependent Hamiltonian list: [H0, [H1, eps_t]]
– Wrap calls to:
    • qutip.mesolve for open‐system dynamics
    • qutip.sesolve or qutip.propagator for closed‐system/unitary evolution
– Return simulation outputs:
    • state trajectories or propagators
    • interface to compute fidelity downstream
"""
from qubit_model import get_free_hamiltonian, get_collapse_operators, get_initial_state, default_params
from control_hamiltonian import get_control_operator, eps_t
import qutip
import numpy as np

def build_hamiltonian(params):
    H0 = get_free_hamiltonian(params)
    H1 = get_control_operator(params)

    return [H0, [H1, eps_t]]

def run_simulation(params, t_list):
    # initial state 
    psi0 = get_initial_state(params)
    # collapse operators 
    c_ops = get_collapse_operators(params)
    # build H 
    H = build_hamiltonian(params)
    # solve open-system dynamics 
    result = qutip.mesolve(H, psi0, t_list, c_ops)
    return result

# extract unitary propogator if needed 
def get_propagator(params, t_list): 
    H = build_hamiltonian(params)
    return qutip.propagator(H, t_list, get_collapse_operators(params))


if __name__ == "__main__":
    params = default_params
    t_final, M = 50e-9, 100
    t_list = np.linspace(0, t_final, M)

    # test Gaussian pulse 
    from control_hamiltonian import eps_samples, time_grid 

    time_grid = t_list 
    eps_samples = np.exp(-((t_list - t_final/2)**2)/(2*(t_final/10)**2))

    result = run_simulation(params, t_list)
    print("Last state:", result.states[-1])
