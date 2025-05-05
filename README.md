
# Quantum-SNN-Control

A hybrid framework for learning spiking-neural-network (SNN)–based control pulses for superconducting qubits using QuTiP simulations.

---

## 🚀 Project Overview

Superconducting qubits require precisely shaped microwave pulses for high-fidelity gates. We combine:

* **QuTiP**: a Python toolbox for modeling and simulating open quantum systems
* **snnTorch**: spiking neural networks in PyTorch, trained via surrogate gradients

Our pipeline:

1. **Build the transmon model** (Hamiltonian, dissipation) in QuTiP
2. **Generate pulse samples** from a recurrent SNN
3. **Drive the QuTiP simulation** with those samples
4. **Compute gate fidelity** and use **loss = 1 – fidelity** for training
5. **Backpropagate** through time to update the SNN

---

## 📂 Repository Structure

```
quantum-snn-control/
├── README.md
├── requirements.txt       # pin QuTiP, snnTorch, NumPy versions
├── qubit_model.py         # transmon parameters, H₀, collapse ops, initial state
├── control_hamiltonian.py # H₁ = a + a† and eps_t callback for drive waveforms
├── simulation.py          # physics engine: build H, run mesolve/propagator, smoke test
├── fidelity_metrics.py    # wrappers for state, process, and average gate fidelities
├── train_snn.py           # (TODO) SNN definition, forward pass → eps_samples → loss → BPTT
├── utils.py               # config loading, noise injection, logging helpers
├── batch_runner.py        # (TODO) parameter sweeps, Monte Carlo robustness tests
└── visualization.py       # (TODO) plotting populations, Bloch spheres, fidelity curves
```

---

## 📋 File Responsibilities

### `qubit_model.py`

*Defines the transmon’s static physics.*

* **TransmonParams**: device constants (omega\_0, alpha, T1, T2, N)
* **get\_operators()** → ladder & identity operators
* **get\_free\_hamiltonian()** → Duffing Hamiltonian H₀
* **get\_collapse\_operators()** → relaxation (T₁) and dephasing (T₂) collapse ops
* **get\_initial\_state()** → ground state |0⟩

### `control_hamiltonian.py`

*Sets up the time-dependent drive interface.*

* **get\_control\_operator()** → H₁ = a + a†
* **eps\_samples**, **time\_grid**: global buffers overwritten by the SNN
* **eps\_t(t, args)**: callback for QuTiP returning drive amplitude ε(t)

### `simulation.py`

*Physics engine glue.*

* **build\_hamiltonian()** → `[H₀, [H₁, eps_t]]`
* **run\_simulation()** → open-system solver (`mesolve`)
* **get\_propagator()** → unitary propagator builder
* **Smoke-test stub** under `if __name__ == "__main__"` to verify setup

### `fidelity_metrics.py`

*Compute fidelity-based losses.*

* **state\_fidelity(ψ\_target, ψ\_actual)**
* **process\_fidelity(U\_target, U\_actual)**
* **average\_gate\_fidelity(U\_target, U\_actual)**

### `train_snn.py`

*Training loop (TODO).*

* Fetch or define **U\_target** (e.g. X or π gate)
* Forward pass: **eps\_samples = model(input\_encoding)**
* Overwrite `control_hamiltonian.eps_samples` and `time_grid`
* Call **run\_simulation()** → final state or propagator
* Compute **loss = 1 – fidelity\_metrics**
* Backpropagate via surrogate gradients and **optimizer.step()**

### `utils.py`

*General utilities.*

* Configuration I/O (JSON/YAML)
* Random noise injection into parameters
* Logging and checkpointing helpers

### `batch_runner.py`

*Robustness & parameter sweeps (TODO).*

* Loop over noise/drift scenarios
* Aggregate mean, std, worst-case fidelities

### `visualization.py`

*Plotting & reporting (TODO).*

* Population vs. time, Bloch-sphere trajectories, Wigner maps
* Fidelity vs. epoch or noise curves

---

## 🔗 Pipeline & File Mapping

| Step                              | File                     | Key Function(s)                                                     |
| --------------------------------- | ------------------------ | ------------------------------------------------------------------- |
| **1. Target gate definition**     | `train_snn.py`           | define/import `U_target`                                            |
| **2. SNN → eps\_samples**         | `train_snn.py`           | `model(input)`, overwrite buffers                                   |
| **3. H₁ & callback**              | `control_hamiltonian.py` | `get_control_operator()`, `eps_t(t,args)`                           |
| **4. Simulation**                 | `simulation.py`          | `build_hamiltonian()`, `run_simulation()`, `get_propagator()`       |
| **5. Fidelity evaluation**        | `fidelity_metrics.py`    | `state_fidelity()`, `process_fidelity()`, `average_gate_fidelity()` |
| **6. Loss = 1 − Fidelity**        | `train_snn.py`           | `loss = 1 - fidelity`                                               |
| **7. Training loop**              | `train_snn.py`           | surrogate‐gradient BPTT, `loss.backward()`, `optimizer.step()`      |
| **8. Config & logging**           | `utils.py`               | `load_config()`, `save_results()`, logging                          |
| **9. Robustness sweeps**          | `batch_runner.py`        | `run_parameter_sweep()`, Monte Carlo loops                          |
| **10. Visualization & reporting** | `visualization.py`       | `plot_population_dynamics()`, `plot_fidelity_vs_epoch()`            |

---

## 🛠️ Setup & Installation

### 1. Clone and enter

```bash
git clone https://github.com/DoQuantum/SNNs-to-modulate-quantum-gate-pulses.git
cd SNNs-to-modulate-quantum-gate-pulses
```

### 2. Ensure **uv** is installed

If you don’t already have `uv`, install it via pip or your system package manager:

```bash
pip install uv
# or follow official instructions: https://astral.sh/uv/install.sh
```

### 3. Create and activate the virtual environment with uv

```bash
# Initialize the project venv (uses Python 3.9+ by default)
uv venv

# Activate it on macOS/Linux:
source .venv/bin/activate

# Activate it on Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# Activate it on Windows Command Prompt:
# .\.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
# Sync the requirements file into the uv environment
uv pip sync requirements.txt
```

### 5. Verify installation

```bash
uv run --python -- python -c "import qutip, snntorch, numpy; print('OK:', qutip.__version__, snntorch.__version__)"
```

### 6. Run the smoke test

```bash
# Use uv to run the simulation script in the correct environment
uv run python simulation.py
```

This should print the final state under the Gaussian-pulse stub. You’re now ready to start coding the SNN!

---


