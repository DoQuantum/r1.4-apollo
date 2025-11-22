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
from scipy.linalg import expm

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

        # Output layer: 2 neurons (generates I/Q control: Ω_x and Ω_y)
        self.fc_out = nn.Linear(64, 2, bias=True)
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
        Generate I/Q pulse sequence for target state

        Args:
            target_state: (B, 2) - target state probabilities [P(|0⟩), P(|1⟩)]

        Returns:
            pulse_sequence: (B, T, 2) - I/Q control signals [Ω_x(t), Ω_y(t)]
        """
        B = target_state.shape[0]
        device = target_state.device

        # Rate-encode target state to spike trains
        spike_input = self._rate_encode(target_state)  # (T, B, 2)

        # Initialize membrane potentials explicitly (required for snntorch 0.9+)
        mem1 = torch.zeros(B, 64, device=device)
        mem2 = torch.zeros(B, 128, device=device)
        mem3 = torch.zeros(B, 64, device=device)
        mem_out = torch.zeros(B, 2, device=device)

        # Storage for output membrane potentials (analog pulse signal)
        pulse_sequence = []

        # Process over time
        for t in range(self.num_steps):
            x_t = spike_input[t]  # (B, 2)

            # Layer 1 - explicitly pass membrane potential
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Output layer
            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            # Read membrane potential as analog control signal
            pulse_sequence.append(mem_out)  # (B, 2)

        # Stack over time: (T, B, 2) -> (B, T, 2)
        pulse_sequence = torch.stack(pulse_sequence, dim=0).transpose(0, 1)

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
    plt.savefig('quantum_snn_pulse.png', dpi=150)
    print("✓ Plot saved to quantum_snn_pulse.png")


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTUM GATE PULSE CONTROL WITH SNN - MVP DEMO")
    print("="*60)
    print("\nTask: Train feedforward SNN to generate pi-pulse")
    print("Goal: Flip qubit from |0> to |1> with high fidelity")
    print("\nKey features:")
    print("  - Feedforward SNN (init_hidden=False)")
    print("  - Proper (spk, mem) tuple handling")
    print("  - Output membrane potentials = analog control signal")
    print("="*60 + "\n")

    # Train
    model, fidelity_history = train_snn_controller(
        num_epochs=300,
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
