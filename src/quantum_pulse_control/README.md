# Quantum Pulse Control with Spiking Neural Networks

A modular implementation for training SNNs to generate quantum control pulses.

## Structure

```
quantum_pulse_control/
├── __init__.py          # Package exports
├── simulator.py         # Quantum simulator (differentiable & non-differentiable)
├── model.py            # Feedforward SNN pulse generator
├── trainer.py          # Training loop with GRAPE-style loss
├── evaluator.py        # Evaluation and visualization
├── main.py             # Entry point
└── README.md           # This file
```

## Usage

### As a script:
```bash
cd quantum_pulse_control
python main.py
```

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

## Requirements

```bash
pip install torch numpy scipy matplotlib snntorch
```
