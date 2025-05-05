"""
fidelity_metrics.py

– Wrap common QuTiP fidelity and process‐fidelity calculations.
– Expose simple functions to compute:
    • state_fidelity: overlap between two state vectors or density matrices
    • process_fidelity: how close two gates/unitaries are
    • average_gate_fidelity: average performance over all inputs
"""

import qutip

def state_fidelity(psi_target, psi_actual):
    """
    Compute the fidelity between two states.

    Args:
        psi_target (Qobj): target state |ψₜ⟩
        psi_actual (Qobj): actual/produced state |ψₐ⟩

    Returns:
        float: F = |⟨ψₜ|ψₐ⟩|²
    """
    return qutip.metrics.fidelity(psi_target, psi_actual)


def process_fidelity(U_target, U_actual):
    """
    Compute process fidelity between two unitaries.

    Args:
        U_target (Qobj): target unitary
        U_actual (Qobj): simulated unitary

    Returns:
        float: Fₚ = Tr(Uₜ† Uₐ) / dim
    """
    return qutip.metrics.process_fidelity(U_target, U_actual)


def average_gate_fidelity(U_target, U_actual):
    """
    Compute the average gate fidelity between two processes.

    Args:
        U_target (Qobj): target gate
        U_actual (Qobj): actual gate applied

    Returns:
        float: average fidelity over all pure inputs
    """
    return qutip.metrics.average_gate_fidelity(U_target, U_actual)

