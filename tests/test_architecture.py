"""Verifica el contrato de arquitectura y la configuración de entrenamiento."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from config import load_config
from q_network import QNetwork

def test_network_shapes_and_gradients():
    cfg = load_config()
    net = QNetwork(cfg["observation_dim"], cfg["n_actions"], cfg["network"]["hidden_layers"][0])
    x = torch.randn(64, 7, dtype=torch.float32)
    y = net(x)
    assert y.shape == (64, 3) and y.dtype == torch.float32
    assert sum(p.numel() for p in net.parameters()) == 17923
    y.square().mean().backward()
    assert all(p.grad is not None for p in net.parameters())

def test_config_contract():
    cfg = load_config()
    assert cfg["training"]["batch_size"] == 64
    assert cfg["training"]["target_update_frequency"] == 1000
