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
