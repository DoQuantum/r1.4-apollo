"""
control_hamiltonian.py

– Define control operator H1 = a + a.dag()
– Provide `eps_t(t, args)` callback:
    • reads SNN’s pulse sample buffer
    • optionally interpolates between discrete time‐steps
    • returns drive amplitude ε(t) for QuTiP time‐dependence
"""
