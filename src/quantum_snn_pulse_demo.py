"""
Quantum Gate Pulse Control with Spiking Neural Networks - MVP Demo

Demonstrates:
1. Feedforward SNN (no recurrence, init_hidden=False)
2. Proper (spk, mem) tuple handling for all LIF layers
3. Output layer membrane potentials read as analog control signals
4. Single-qubit Rabi oscillation task (simplest quantum gate)

The SNN learns to generate pulse shapes that drive a qubit to a target state.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    import snntorch as snn
    from snntorch import surrogate
    SNN_AVAILABLE = True
except ImportError:
    print("ERROR: snntorch not installed. Run: pip install snntorch")
    exit(1)

# ============================================================
# 1. QUANTUM SIMULATOR (Single Qubit)
# ============================================================

class SingleQubitSimulator:
    """
    Simulates a single qubit driven by control pulses (Rabi oscillations)

    Hamiltonian: H = (ω/2)σz + Ω(t)σx
    where Ω(t) is the control pulse (Rabi frequency)
    """
    def __init__(self, omega=1.0, dt=0.01):
        self.omega = omega  # Qubit frequency
        self.dt = dt        # Time step

    def evolve(self, pulse_amplitudes):
        """
        Evolve qubit state under control pulses

        Args:
            pulse_amplitudes: (T,) array of control pulse values

        Returns:
            final_state: (2,) complex array - final qubit state
        """
        # Initialize to ground state |0⟩
        state = np.array([1.0 + 0j, 0.0 + 0j])

        # Pauli matrices
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

        # Time evolution (rotating frame - only X rotations)
        for omega_t in pulse_amplitudes:
            # In rotating frame at qubit frequency, only X rotation remains
            # Rotation angle θ = Ω(t)·dt
            angle = omega_t * self.dt
            cos_a = np.cos(angle / 2)
            sin_a = np.sin(angle / 2)

            # Apply RX(θ) rotation
            new_state = np.array([
                cos_a * state[0] - 1j * sin_a * state[1],
                -1j * sin_a * state[0] + cos_a * state[1]
            ], dtype=complex)

            state = new_state
            # Already unitary, no need to renormalize

        return state

    def evolve_differentiable(self, pulse_amplitudes, device):
        """
        DIFFERENTIABLE quantum evolution (PyTorch version for backprop)

        Args:
            pulse_amplitudes: (B, T) torch tensor

        Returns:
            final_states: (B, 2) complex torch tensor
        """
        B, T = pulse_amplitudes.shape

        # Initialize to ground state |0⟩ for all batch samples
        # Use real representation: state = [re(α), im(α), re(β), im(β)]
        states = torch.zeros(B, 4, device=device)
        states[:, 0] = 1.0  # |0⟩ = [1, 0]

        # Pauli matrices (real representation)
        # σx causes transitions, σz is diagonal

        for t in range(T):
            omega_t = pulse_amplitudes[:, t]  # (B,)

            # For each sample, evolve by dt
            # Hamiltonian: H = (ω/2)σz + Ω(t)σx
            # In rotating frame (resonant drive), σz term vanishes
            # This is valid when driving at qubit frequency

            # X rotation by angle θ = Ω(t)·dt
            angle = omega_t * self.dt
            cos_a = torch.cos(angle / 2)
            sin_a = torch.sin(angle / 2)

            # Rotation matrix for X-gate: RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
            # In real form: [re_alpha, im_alpha, re_beta, im_beta]
            new_states = torch.zeros_like(states)
            new_states[:, 0] = cos_a * states[:, 0] - sin_a * states[:, 3]  # re(α') = cos·re(α) - sin·im(β)
            new_states[:, 1] = cos_a * states[:, 1] + sin_a * states[:, 2]  # im(α') = cos·im(α) + sin·re(β)
            new_states[:, 2] = cos_a * states[:, 2] - sin_a * states[:, 1]  # re(β') = cos·re(β) - sin·im(α)
            new_states[:, 3] = cos_a * states[:, 3] + sin_a * states[:, 0]  # im(β') = cos·im(β) + sin·re(α)

            states = new_states

            # Normalize (ensure |α|² + |β|² = 1)
            norm = torch.sqrt(states[:, 0]**2 + states[:, 1]**2 + states[:, 2]**2 + states[:, 3]**2)
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


# ============================================================
# 2. FEEDFORWARD SNN PULSE GENERATOR
# ============================================================

class FeedforwardSNN_PulseGenerator(nn.Module):
    """
    Feedforward SNN that generates quantum control pulses

    Architecture:
    - Input: Target state encoding (rate-coded spikes)
    - Hidden layers: LIF neurons (purely spiking, init_hidden=False)
    - Output layer: LIF neurons (membrane potentials = analog pulse signal)

    Key: We read membrane potentials from output layer as continuous control signal
    """
    def __init__(self, num_steps=50, beta=0.9):
        super().__init__()

        self.num_steps = num_steps  # Number of timesteps
        self.beta = beta            # LIF leak factor

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Feedforward architecture (no recurrence)
        self.fc1 = nn.Linear(2, 64, bias=True)   # Input: 2 (target state: prob |0⟩, prob |1⟩)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self.fc2 = nn.Linear(64, 128, bias=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self.fc3 = nn.Linear(128, 64, bias=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        # Output layer: 1 neuron (generates pulse amplitude)
        self.fc_out = nn.Linear(64, 1, bias=True)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization with smaller output layer"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Output layer gets smaller initialization to prevent saturation
        nn.init.xavier_normal_(self.fc_out.weight, gain=0.1)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, target_state):
        """
        Generate pulse sequence for target state

        Args:
            target_state: (B, 2) - target state probabilities [P(|0⟩), P(|1⟩)]

        Returns:
            pulse_sequence: (B, T) - membrane potentials of output layer
        """
        B = target_state.shape[0]

        # Rate-encode target state to spike trains
        spike_input = self._rate_encode(target_state)  # (T, B, 2)

        # Reset hidden states (init_hidden=False means we manually reset)
        from snntorch import utils
        utils.reset(self)

        # Storage for output membrane potentials (analog pulse signal)
        pulse_sequence = []

        # Process over time
        for t in range(self.num_steps):
            x_t = spike_input[t]  # (B, 2)

            # Layer 1
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1)  # MUST handle tuple

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2)  # MUST handle tuple

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3)  # MUST handle tuple

            # Output layer
            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out)  # MUST handle tuple

            # Read membrane potential as analog control signal
            pulse_sequence.append(mem_out)  # (B, 1)

        # Stack over time: (T, B, 1) -> (B, T)
        pulse_sequence = torch.stack(pulse_sequence, dim=0).squeeze(-1).transpose(0, 1)

        # Apply tanh to constrain pulse amplitude to reasonable range [-5, 5]
        # Increased from 2.0 to allow sufficient rotation amplitude
        pulse_sequence = 5.0 * torch.tanh(pulse_sequence)

        return pulse_sequence

    def _rate_encode(self, target_state, max_rate=1.0):
        """
        Convert target state to constant spike trains (deterministic)

        Args:
            target_state: (B, 2) - [P(|0⟩), P(|1⟩)]

        Returns:
            spike_trains: (T, B, 2)
        """
        B = target_state.shape[0]

        # Use constant input instead of stochastic Poisson spikes
        # This ensures consistent pulse generation during eval
        constant_input = target_state * max_rate  # (B, 2)

        spike_trains = []
        for t in range(self.num_steps):
            spike_trains.append(constant_input)

        return torch.stack(spike_trains, dim=0)  # (T, B, 2)


# ============================================================
# 3. TRAINING LOOP
# ============================================================

def train_snn_controller(num_epochs=200, batch_size=8, lr=1e-4):
    """
    Train SNN to generate pulses that drive qubit to target state |1⟩

    Task: π-pulse (flip |0⟩ → |1⟩)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize model
    num_steps = 100  # Increased timesteps for better temporal resolution
    model = FeedforwardSNN_PulseGenerator(num_steps=num_steps, beta=0.9).to(device)

    # Quantum simulator
    # Use small dt for accurate time evolution (first-order approximation needs dt << 1)
    dt = 0.01  # Small timestep for numerical accuracy
    simulator = SingleQubitSimulator(omega=1.0, dt=dt)

    # Target state: |1⟩ (excited state)
    target_state_quantum = np.array([0.0 + 0j, 1.0 + 0j])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTask: Learn π-pulse to flip |0⟩ → |1⟩")
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

        # Generate pulse sequence (now returns tanh-constrained values)
        pulse_sequence = model(target_input)  # (B, T) - already in [-2, 2]

        # DIFFERENTIABLE quantum simulation (for backprop)
        final_states = simulator.evolve_differentiable(pulse_sequence, device)  # (B, 4)
        fidelities = simulator.compute_fidelity_differentiable(final_states, device)  # (B,)

        # Also compute non-differentiable fidelity for monitoring
        with torch.no_grad():
            avg_fidelity_monitor = fidelities.mean().item()

        # PRIMARY LOSS: Maximize fidelity (minimize 1 - fidelity)
        loss_fidelity = (1.0 - fidelities).mean()

        # SECONDARY LOSS: Encourage smooth pulses (regularization)
        pulse_diff = pulse_sequence[:, 1:] - pulse_sequence[:, :-1]
        loss_smoothness = (pulse_diff ** 2).mean()

        # TERTIARY LOSS: Encourage energy efficiency
        loss_energy = (pulse_sequence ** 2).mean()

        # Combined loss (balanced to prevent saturation)
        loss = 10.0 * loss_fidelity + 0.1 * loss_smoothness + 0.01 * loss_energy

        loss.backward()

        # Gradient clipping to prevent weight explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        fidelity_history.append(avg_fidelity_monitor)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Fidelity: {avg_fidelity_monitor:.4f} | Loss: {loss.item():.4f}")

    print("\n✓ Training complete!")
    return model, fidelity_history


# ============================================================
# 4. EVALUATION & VISUALIZATION
# ============================================================

def evaluate_and_visualize(model):
    """Test the trained SNN controller"""
    device = next(model.parameters()).device
    model.eval()

    # Target: |1⟩ state
    target_input = torch.tensor([[0.0, 1.0]], device=device)

    with torch.no_grad():
        pulse_sequence = model(target_input)  # (1, T) - already normalized

    pulse_np = pulse_sequence[0].cpu().numpy()  # Already in [-5, 5] from model

    # Simulate quantum evolution (use same dt as training)
    num_steps = 100
    dt = 0.01
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
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    time_steps = np.arange(len(pulse_np))

    # Pulse shape
    axes[0].plot(time_steps, pulse_np, 'b-', linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Pulse amplitude Ω(t)')
    axes[0].set_title(f'SNN-Generated Control Pulse (Fidelity: {fidelity:.4f})')
    axes[0].grid(alpha=0.3)

    # State evolution (probability of |1⟩ over time)
    state_evolution = []
    state = np.array([1.0 + 0j, 0.0 + 0j])  # Start in |0⟩

    for omega_t in pulse_np:
        # Rotating frame evolution (same as training)
        angle = omega_t * simulator.dt
        cos_a = np.cos(angle / 2)
        sin_a = np.sin(angle / 2)

        new_state = np.array([
            cos_a * state[0] - 1j * sin_a * state[1],
            -1j * sin_a * state[0] + cos_a * state[1]
        ], dtype=complex)
        state = new_state

        prob_1 = np.abs(state[1]) ** 2  # Probability of |1⟩
        state_evolution.append(prob_1)

    axes[1].plot(time_steps, state_evolution, 'r-', linewidth=2)
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Target')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('P(|1⟩)')
    axes[1].set_title('Qubit State Evolution')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('quantum_snn_pulse.png', dpi=150)
    print("✓ Plot saved to quantum_snn_pulse.png")


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTUM GATE PULSE CONTROL WITH SNN - MVP DEMO")
    print("="*60)
    print("\nTask: Train feedforward SNN to generate π-pulse")
    print("Goal: Flip qubit from |0⟩ to |1⟩ with high fidelity")
    print("\nKey features:")
    print("  - Feedforward SNN (init_hidden=False)")
    print("  - Proper (spk, mem) tuple handling")
    print("  - Output membrane potentials = analog control signal")
    print("="*60 + "\n")

    # Train
    model, fidelity_history = train_snn_controller(
        num_epochs=200,
        batch_size=8,
        lr=1e-4
    )

    # Evaluate
    evaluate_and_visualize(model)

    print("\n✅ Demo complete!")
    print("\nNote: This is a proof-of-concept. Real quantum control faces:")
    print("  - Timescale mismatch (ns pulses vs ms SNN dynamics)")
    print("  - Need for GHz sampling rates")
    print("  - Hardware constraints for cryogenic operation")
    print("\nBut it shows SNNs CAN generate temporal patterns for control tasks.")
