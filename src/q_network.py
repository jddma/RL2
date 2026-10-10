import torch
from torch import nn

class QNetwork(nn.Module):
    """MLP con siete entradas normalizadas, dos capas ocultas y tres valores Q de salida."""
    def __init__(self, input_dim=7, output_dim=3, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,output_dim))
    def forward(self, x):
        return self.network(x)
