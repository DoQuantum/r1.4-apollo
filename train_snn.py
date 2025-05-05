"""
train_snn.py

– Define and instantiate spiking neural network (snnTorch)
– For each training iteration:
    1. Forward pass → generate pulse samples array
    2. Call simulation.run(...) to get fidelity
    3. Compute loss = 1 – fidelity
    4. Backpropagate via surrogate gradients
– Save checkpoints and final trained weights
"""
