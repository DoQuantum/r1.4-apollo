
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
