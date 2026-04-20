"""
model.py
--------
Defines the PrunableLinear layer and the SelfPruningNet network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate
    to every weight. During training, gates are pushed toward 0
    (via L1 sparsity loss), effectively pruning weak connections.

    Forward pass:
        gates         = sigmoid(gate_scores)          # ∈ (0, 1)
        pruned_weight = weight * gates                # element-wise
        output        = x @ pruned_weight.T + bias
    
    Gradients flow through both `weight` and `gate_scores` automatically
    via PyTorch autograd (sigmoid + element-wise mul are differentiable).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard learnable parameters
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight, initialized at 0
        # sigmoid(0) = 0.5 → gates start half-open, optimizer decides fate
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Detached gate values for analysis (no grad)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below the threshold."""
        return (self.get_gates() < threshold).float().mean().item()


class SelfPruningNet(nn.Module):
    """
    4-layer MLP for CIFAR-10 using PrunableLinear layers.

    Architecture:
        Flatten → FC(1024) → BN → ReLU
                → FC(512)  → BN → ReLU
                → FC(256)  → BN → ReLU
                → FC(10)   [logits]
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

    def prunable_layers(self):
        """Yield all PrunableLinear modules."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def all_gates(self) -> torch.Tensor:
        """Flat tensor of every gate value across all layers."""
        return torch.cat([l.get_gates().view(-1) for l in self.prunable_layers()])

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Global fraction of weights considered pruned."""
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()