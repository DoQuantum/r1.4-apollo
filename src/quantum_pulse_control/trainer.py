"""
Training logic for SNN quantum pulse controller.
"""

import torch
import numpy as np
from .model import FeedforwardSNN_PulseGenerator
from .simulator import SingleQubitSimulator


def train_snn_controller(num_epochs=300, batch_size=8, lr=1e-4):
    """
    Train SNN to generate I/Q pulses that drive qubit to target state |1⟩

    Task: π-pulse (flip |0⟩ → |1⟩) with full Bloch sphere control
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize model
    num_steps = 200  # Balanced: better than 100, not as extreme as 500
    model = FeedforwardSNN_PulseGenerator(num_steps=num_steps, beta=0.9).to(device)

    # Quantum simulator with balanced timestep
    dt = 0.005  # Balanced for num_steps=200
    simulator = SingleQubitSimulator(omega=1.0, dt=dt, T1=None, T2=None)  # No decoherence initially

    # Target state: |1⟩ (excited state)
    target_state_quantum = np.array([0.0 + 0j, 1.0 + 0j])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTask: Learn pi-pulse to flip |0> -> |1>")
    print(f"SNN timesteps: {num_steps}")
    print(f"Quantum dt: {dt}")
    print(f"Training for {num_epochs} epochs with batch size {batch_size}\n")

    fidelity_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Create batch: all samples want to reach |1⟩
        # Input encoding: target state as [P(|0⟩), P(|1⟩)] = [0, 1]
        target_input = torch.tensor([[0.0, 1.0]], device=device).repeat(batch_size, 1)

        # Generate I/Q pulse sequence
        pulse_sequence = model(target_input)  # (B, T, 2) - [Ω_x(t), Ω_y(t)] in [-5, 5]

        # DIFFERENTIABLE quantum simulation (for backprop)
        final_states = simulator.evolve_differentiable(pulse_sequence, device)  # (B, 4)
        fidelities = simulator.compute_fidelity_differentiable(final_states, device)  # (B,)

        # Also compute non-differentiable fidelity for monitoring
        with torch.no_grad():
            avg_fidelity_monitor = fidelities.mean().item()

        # PRIMARY LOSS: Maximize fidelity (minimize 1 - fidelity)
        loss_fidelity = (1.0 - fidelities).mean()

        # SECONDARY LOSS: Encourage smooth pulses (first derivative penalty)
        pulse_diff = pulse_sequence[:, 1:, :] - pulse_sequence[:, :-1, :]
        loss_smoothness = (pulse_diff ** 2).mean()

        # TERTIARY LOSS: Second derivative penalty (bandwidth constraint)
        pulse_diff2 = pulse_diff[:, 1:, :] - pulse_diff[:, :-1, :]
        loss_bandwidth = (pulse_diff2 ** 2).mean()

        # QUATERNARY LOSS: Energy efficiency (penalize high power)
        loss_energy = (pulse_sequence ** 2).mean()

        # GRAPE-style combined loss with stronger smoothness constraints
        loss = 10.0 * loss_fidelity + 0.5 * loss_smoothness + 0.2 * loss_bandwidth + 0.01 * loss_energy

        loss.backward()

        # Gradient clipping to prevent weight explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        fidelity_history.append(avg_fidelity_monitor)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Fidelity: {avg_fidelity_monitor:.4f} | Loss: {loss.item():.4f}")

    print("\n✓ Training complete!")
    return model, fidelity_history
