"""
SNN-based pulse generator model.
"""

import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate
    SNN_AVAILABLE = True
except ImportError:
    print("ERROR: snntorch not installed. Run: pip install snntorch")
    exit(1)


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
