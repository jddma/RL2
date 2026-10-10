"""Carga y valida la configuración oficial del experimento."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dqn_main.yaml"

def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg["observation_dim"] != 7 or cfg["n_actions"] != 3:
        raise ValueError("La arquitectura debe corresponder al vector de 7 observaciones y 3 acciones")
    if cfg["network"]["hidden_layers"] != [128, 128]:
        raise ValueError("La arquitectura actual utiliza dos capas ocultas de 128 neuronas")
    if cfg["training"]["learning_starts"] < cfg["training"]["batch_size"]:
        raise ValueError("learning_starts debe ser mayor o igual que batch_size")
    return cfg
