"""
Evaluation and visualization utilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from .simulator import SingleQubitSimulator


def evaluate_and_visualize(model):
    """Test the trained SNN controller"""
    device = next(model.parameters()).device
    model.eval()

    # Target: |1⟩ state
    target_input = torch.tensor([[0.0, 1.0]], device=device)

    with torch.no_grad():
        pulse_sequence = model(target_input)  # (1, T, 2) - I/Q control

    pulse_np = pulse_sequence[0].cpu().numpy()  # (T, 2) - [Ω_x(t), Ω_y(t)]

    # Simulate quantum evolution (use same dt as training)
    num_steps = 200
    dt = 0.005
    simulator = SingleQubitSimulator(omega=1.0, dt=dt)
    target_state = np.array([0.0 + 0j, 1.0 + 0j])
    final_state = simulator.evolve(pulse_np)
    fidelity = simulator.compute_fidelity(final_state, target_state)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Target state: |1⟩")
    print(f"Final state: [{final_state[0]:.3f}, {final_state[1]:.3f}]")
    print(f"Fidelity: {fidelity:.4f}")
    print(f"Gate error: {1 - fidelity:.4e}")
    print(f"{'='*60}\n")

    # Plot pulse shape
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    time_steps = np.arange(len(pulse_np))

    # I/Q Pulse shapes
    axes[0].plot(time_steps, pulse_np[:, 0], 'b-', linewidth=1.5, label='Ω_x(t) [I]')
    axes[0].plot(time_steps, pulse_np[:, 1], 'r-', linewidth=1.5, label='Ω_y(t) [Q]')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Pulse amplitude')
    axes[0].set_title(f'SNN-Generated I/Q Control Pulses (Fidelity: {fidelity:.4f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Pulse magnitude
    pulse_magnitude = np.sqrt(pulse_np[:, 0]**2 + pulse_np[:, 1]**2)
    axes[1].plot(time_steps, pulse_magnitude, 'g-', linewidth=2)
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('|Ω(t)|')
    axes[1].set_title('Pulse Magnitude')
    axes[1].grid(alpha=0.3)

    # State evolution (probability of |1⟩ over time)
    state_evolution = []
    state = np.array([1.0 + 0j, 0.0 + 0j])  # Start in |0⟩
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    for omega_x, omega_y in pulse_np:
        # Sequential RX then RY (same as training)
        H_x = omega_x * sigma_x
        U_x = expm(-1j * H_x * simulator.dt)
        state = U_x @ state

        H_y = omega_y * sigma_y
        U_y = expm(-1j * H_y * simulator.dt)
        state = U_y @ state

        prob_1 = np.abs(state[1]) ** 2  # Probability of |1⟩
        state_evolution.append(prob_1)

    axes[2].plot(time_steps, state_evolution, 'purple', linewidth=2)
    axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Target')
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('P(|1⟩)')
    axes[2].set_title('Qubit State Evolution')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('quantum_pulse_control/quantum_snn_pulse.png', dpi=150)
    print("✓ Plot saved to quantum_pulse_control/quantum_snn_pulse.png")
