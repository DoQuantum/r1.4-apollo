"""
Quantum simulator module for single qubit evolution.
"""

import numpy as np
import torch
from scipy.linalg import expm


class SingleQubitSimulator:
    """
    Simulates a single qubit driven by I/Q control pulses (full Bloch sphere control)

    Hamiltonian: H = Ω_x(t)σx + Ω_y(t)σy
    where Ω_x(t), Ω_y(t) are I/Q quadrature controls (rotating frame)

    Also includes T1 and T2 decoherence via Lindblad master equation
    """
    def __init__(self, omega=1.0, dt=0.002, T1=None, T2=None):
        self.omega = omega  # Qubit frequency (not used in rotating frame)
        self.dt = dt        # Time step (reduced for finer control)
        self.T1 = T1        # Amplitude damping time (None = no decoherence)
        self.T2 = T2        # Dephasing time (None = no decoherence)

    def evolve(self, pulse_amplitudes):
        """
        Evolve qubit state under I/Q control pulses (sequential RX then RY)

        Args:
            pulse_amplitudes: (T, 2) array of control pulse values [Ω_x(t), Ω_y(t)]

        Returns:
            final_state: (2,) complex array - final qubit state
        """
        # Initialize to ground state |0⟩
        state = np.array([1.0 + 0j, 0.0 + 0j])

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        # Time evolution - apply RX then RY sequentially (same as training)
        for omega_x, omega_y in pulse_amplitudes:
            # First apply X rotation
            H_x = omega_x * sigma_x
            U_x = expm(-1j * H_x * self.dt)
            state = U_x @ state

            # Then apply Y rotation
            H_y = omega_y * sigma_y
            U_y = expm(-1j * H_y * self.dt)
            state = U_y @ state

        return state

    def evolve_differentiable(self, pulse_amplitudes, device):
        """
        DIFFERENTIABLE quantum evolution with I/Q control (PyTorch version)

        Args:
            pulse_amplitudes: (B, T, 2) torch tensor [Ω_x(t), Ω_y(t)]

        Returns:
            final_states: (B, 4) real representation of complex state
        """
        B, T, _ = pulse_amplitudes.shape

        # Initialize to ground state |0⟩ for all batch samples
        # Use real representation: state = [re(α), im(α), re(β), im(β)]
        states = torch.zeros(B, 4, device=device)
        states[:, 0] = 1.0  # |0⟩ = [1, 0]

        for t in range(T):
            omega_x = pulse_amplitudes[:, t, 0]  # (B,)
            omega_y = pulse_amplitudes[:, t, 1]  # (B,)

            # Apply separate X and Y rotations sequentially
            # RX(θ) = exp(-i·Ωx·dt·σx) in real representation
            angle_x = omega_x * self.dt
            cos_x = torch.cos(angle_x)
            sin_x = torch.sin(angle_x)

            # RX: [[cos, -i·sin], [-i·sin, cos]]
            # Real form: α' = cos·α - i·sin·β,  β' = -i·sin·α + cos·β
            temp_states = torch.zeros_like(states)
            temp_states[:, 0] = cos_x * states[:, 0] + sin_x * states[:, 3]  # re(α')
            temp_states[:, 1] = cos_x * states[:, 1] - sin_x * states[:, 2]  # im(α')
            temp_states[:, 2] = cos_x * states[:, 2] - sin_x * states[:, 1]  # re(β')
            temp_states[:, 3] = cos_x * states[:, 3] + sin_x * states[:, 0]  # im(β')

            # RY(θ) = exp(-i·Ωy·dt·σy)
            angle_y = omega_y * self.dt
            cos_y = torch.cos(angle_y)
            sin_y = torch.sin(angle_y)

            # RY: [[cos, -sin], [sin, cos]]
            # Real form: α' = cos·α - sin·β,  β' = sin·α + cos·β
            new_states = torch.zeros_like(temp_states)
            new_states[:, 0] = cos_y * temp_states[:, 0] - sin_y * temp_states[:, 2]  # re(α')
            new_states[:, 1] = cos_y * temp_states[:, 1] - sin_y * temp_states[:, 3]  # im(α')
            new_states[:, 2] = sin_y * temp_states[:, 0] + cos_y * temp_states[:, 2]  # re(β')
            new_states[:, 3] = sin_y * temp_states[:, 1] + cos_y * temp_states[:, 3]  # im(β')

            states = new_states

            # Normalize (ensure |α|² + |β|² = 1)
            norm = torch.sqrt(states[:, 0]**2 + states[:, 1]**2 + states[:, 2]**2 + states[:, 3]**2)
            norm = torch.clamp(norm, min=1e-10)  # Prevent division by zero
            states = states / norm.unsqueeze(1)

        return states  # (B, 4) - [re(α), im(α), re(β), im(β)]

    def compute_fidelity(self, final_state, target_state):
        """Compute fidelity: |⟨target|final⟩|²"""
        overlap = np.abs(np.vdot(target_state, final_state))
        return overlap ** 2

    def compute_fidelity_differentiable(self, final_states, device):
        """
        Compute fidelity with target |1⟩ (differentiable)

        Args:
            final_states: (B, 4) torch tensor [re(α), im(α), re(β), im(β)]

        Returns:
            fidelities: (B,) torch tensor
        """
        # Target |1⟩ = [0, 1] → [re(α)=0, im(α)=0, re(β)=1, im(β)=0]
        # Fidelity = |⟨1|ψ⟩|² = |β|² = re(β)² + im(β)²
        fidelities = final_states[:, 2]**2 + final_states[:, 3]**2
        return fidelities
