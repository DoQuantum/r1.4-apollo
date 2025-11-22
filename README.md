
# Quantum Pulse Control with Spiking Neural Networks

An implementation for learning spiking-neural-network (SNN)–based control pulses for superconducting qubits.

---

## 🚀 Project Overview

Superconducting qubits require precisely shaped microwave pulses for high-fidelity gates, and need to react to realtime changes.

Our pipeline:

1. **Build the transmon qubit model** (Hamiltonian, dissipation)
2. **Generate pulse samples** from an SNN
3. **Drive the simulation** with those samples
4. **Compute gate fidelity** and use the loss for training
5. **Backpropagate** through time (BPTT) to update the SNN

---

## 📂 Repository Structure

```
demo                          # Demo for 0 to 1 state
├── quantum_snn_pulse_demo.py
└── writeup.txt
src
├── utils.py                  # Utilities for logging and contexts
├── run_quantum_pulse.py      # Demo for |0> -> |1> pulse
└── quantum_pulse_control/    # SNN model module
    ├── simulator.py           # Quantum simulator (differentiable & non-differentiable)
    ├── model.py               # Feedforward SNN pulse generator
    ├── trainer.py             # Training loop with GRAPE-style loss
    └── evaluator.py           # Evaluation and visualization
```

## Usage

### As a module:
```python
from quantum_pulse_control import train_snn_controller, evaluate_and_visualize

model, history = train_snn_controller(num_epochs=300, batch_size=8, lr=1e-4)
evaluate_and_visualize(model)
```

## Components

- **SingleQubitSimulator**: Simulates single qubit evolution with I/Q control pulses
- **FeedforwardSNN_PulseGenerator**: 4-layer spiking neural network that outputs analog control signals
- **train_snn_controller**: GRAPE-inspired training with fidelity, smoothness, bandwidth, and energy losses
- **evaluate_and_visualize**: Tests trained model and generates plots

---

## 🛠️ Setup & Installation

0. Install [uv](https://docs.astral.sh/uv/)
1. Clone
2. Setup virtual environment: `uv sync`
3. Run the respective main file

---


