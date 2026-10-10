"""Entrenamiento reproducible con configuración centralizada y registro de episodios."""
import argparse
import csv
import json
import platform
import random
import sys
from collections import deque
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import gymnasium
import numpy as np
import torch
from config import load_config, DEFAULT_CONFIG
from env_utils import make_env, ENV_ID, ENV_KWARGS
from q_network import QNetwork
from replay_buffer import ReplayBuffer
from dqn import train_dqn

BEST_CHECKPOINT_WINDOW = 20

def main():
    p = argparse.ArgumentParser(description="Entrenamiento DQN en Hill Climb Racing")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default="results")
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    t = cfg["training"]
    episodes = args.episodes if args.episodes is not None else t["episodes"]
    seed = args.seed if args.seed is not None else t["seed"]
    learning_starts = args.learning_starts if args.learning_starts is not None else t["learning_starts"]
    batch_size = args.batch_size if args.batch_size is not None else t["batch_size"]
    if episodes < 1 or batch_size < 1 or learning_starts < batch_size:
        raise ValueError("Verifique episodios, batch_size y learning_starts")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env()
    hidden = cfg["network"]["hidden_layers"][0]
    online = QNetwork(cfg["observation_dim"], cfg["n_actions"], hidden)
    target = QNetwork(cfg["observation_dim"], cfg["n_actions"], hidden)
    optimizer = torch.optim.Adam(online.parameters(), lr=t["learning_rate"])

    # Mejor checkpoint: media movil de recompensa sobre las ultimas BEST_CHECKPOINT_WINDOW episodios
    reward_window = deque(maxlen=BEST_CHECKPOINT_WINDOW)
    best_tracker = {"best_moving_average_reward": float("-inf"), "episode": None}

    def on_episode_end(episode_index, episode_reward, model):
        reward_window.append(episode_reward)
        if len(reward_window) < BEST_CHECKPOINT_WINDOW:
            return
        moving_average = sum(reward_window) / BEST_CHECKPOINT_WINDOW
        if moving_average > best_tracker["best_moving_average_reward"]:
            best_tracker["best_moving_average_reward"] = moving_average
            best_tracker["episode"] = episode_index + 1
            torch.save({"model": model.state_dict(), "seed": seed,
                        "episode": episode_index + 1,
                        "moving_average_reward": moving_average,
                        "window_size": BEST_CHECKPOINT_WINDOW,
                        "network": cfg["network"], "observation_dim": cfg["observation_dim"],
                        "n_actions": cfg["n_actions"]}, out / "checkpoint_best.pt")

    result = train_dqn(
        env, online, target, ReplayBuffer(t["replay_capacity"]), optimizer,
        num_episodes=episodes, gamma=t["gamma"], batch_size=batch_size,
        learning_starts=learning_starts, train_frequency=t["train_frequency"],
        target_update_frequency=t["target_update_frequency"],
        eps_start=t["epsilon_start"], eps_end=t["epsilon_end"],
        eps_decay_steps=t["epsilon_decay_steps"], seed=seed,
        episode_callback=on_episode_end,
    )
    torch.save({"model": online.state_dict(), "seed": seed, "episodes": episodes,
                "network": cfg["network"], "observation_dim": cfg["observation_dim"],
                "n_actions": cfg["n_actions"]}, out / "checkpoint_final.pt")
    with (out / "logs.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["episode", "reward", "episode_length", "epsilon", "avg_loss", "terminated", "truncated", "max_distance"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["history"])
    official = dict(cfg)
    official["training"] = dict(t, episodes=episodes, seed=seed,
                                learning_starts=learning_starts, batch_size=batch_size)
    official["env_kwargs"] = ENV_KWARGS
    official["env_id"] = ENV_ID
    official["command"] = " ".join(sys.argv)
    try:
        hill_racing_env_version = pkg_version("hill-climb-racing-env")
    except PackageNotFoundError:
        hill_racing_env_version = "unknown"
    official["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "hill_racing_env": hill_racing_env_version,
    }
    official["best_checkpoint"] = {
        "criterion": f"media movil de recompensa sobre {BEST_CHECKPOINT_WINDOW} episodios",
        "episode": best_tracker["episode"],
        "moving_average_reward": (
            None if best_tracker["episode"] is None
            else best_tracker["best_moving_average_reward"]
        ),
        "file": "checkpoint_best.pt" if best_tracker["episode"] is not None else None,
    }
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(official, f, indent=2)
    env.close()
    print("Guardados logs.csv, config.json y checkpoint_final.pt en", out)
    if best_tracker["episode"] is not None:
        print("Mejor checkpoint (checkpoint_best.pt) en episodio", best_tracker["episode"],
              "con media movil de recompensa", best_tracker["best_moving_average_reward"])

if __name__ == "__main__":
    main()
