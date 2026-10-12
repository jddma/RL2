import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from env_utils import make_env
from q_network import QNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", choices=["dqn", "random"], default="dqn"
    )
    parser.add_argument(
        "--checkpoint", default="results/checkpoint_best.pt"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default="results/evaluation_v2")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes debe ser positivo")

    model = None
    if args.policy == "dqn":
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        hidden = checkpoint.get(
            "network", {"hidden_layers": [128, 128]}
        )["hidden_layers"]

        if len(hidden) != 2 or hidden[0] != hidden[1]:
            raise ValueError("QNetwork requiere dos capas de igual tamaño")

        model = QNetwork(
            checkpoint.get("observation_dim", 7),
            checkpoint.get("n_actions", 3),
            hidden[0],
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env()

    # make_env devuelve el wrapper del proyecto.
    raw_env = env.env.unwrapped
    max_score = inspect.getmodule(type(raw_env)).MAX_SCORE
    rows = []

    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            state, info = env.reset(seed=seed)

            # Generador independiente del azar utilizado por el terreno.
            rng = np.random.default_rng(seed)
            counts = np.zeros(env.action_space.n, dtype=int)

            total_reward = 0.0
            steps = 0
            max_score_observed = 0
            terminated = truncated = False

            while not (terminated or truncated):
                if args.policy == "random":
                    action = int(rng.integers(env.action_space.n))
                else:
                    with torch.inference_mode():
                        tensor = torch.as_tensor(
                            state, dtype=torch.float32
                        ).unsqueeze(0)
                        action = int(model(tensor).argmax(dim=1).item())

                counts[action] += 1
                state, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1
                max_score_observed = max(
                    max_score_observed, int(info["score"])
                )

            death = bool(raw_env.agent.dead)
            success = bool(
                terminated
                and not death
                and int(info["score"]) >= max_score
            )

            row = {
                "episode": episode + 1,
                "seed": seed,
                "policy": args.policy,
                "reward": total_reward,
                "max_score": max_score_observed,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": success,
                "death": death,
                "idle_count": int(counts[0]),
                "gas_count": int(counts[1]),
                "reverse_count": int(counts[2]),
            }
            rows.append(row)

            print(
                f"Episodio {episode + 1:02d} | seed={seed} | "
                f"reward={total_reward:.2f} | score={max_score_observed} | "
                f"pasos={steps} | muerte={death} | éxito={success}",
                flush=True,
            )
    finally:
        env.close()

    with (out / "evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rewards = np.array([r["reward"] for r in rows])
    scores = np.array([r["max_score"] for r in rows])
    lengths = np.array([r["steps"] for r in rows])

    metrics = {
        "policy": args.policy,
        "checkpoint": args.checkpoint if model is not None else None,
        "episodes": len(rows),
        "seed_start": args.seed,
        "seed_end": args.seed + args.episodes - 1,
        "success_threshold": int(max_score),
        "reward_mean": float(rewards.mean()),
        "reward_median": float(np.median(rewards)),
        "reward_std": float(rewards.std(ddof=0)),
        "score_mean": float(scores.mean()),
        "score_max": int(scores.max()),
        "steps_mean": float(lengths.mean()),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "death_rate": float(np.mean([r["death"] for r in rows])),
        "truncation_rate": float(np.mean([r["truncated"] for r in rows])),
    }

    for action in ["idle", "gas", "reverse"]:
        metrics[f"{action}_fraction"] = float(
            sum(r[f"{action}_count"] for r in rows) / lengths.sum()
        )

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\nMétricas:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()