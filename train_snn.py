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
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
import snntorch.surrogate as surrogate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(100, 10)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps=10):
        spk_out = None
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, _ = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, _ = self.lif2(cur2)

            if spk_out is None:
                spk_out = spk2
            else:
                spk_out += spk2
        return spk_out / num_steps #Averaged output