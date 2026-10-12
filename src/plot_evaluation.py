import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

EVALUATIONS = [
    ("DQN best", "evaluation_best_v2"),
    ("DQN final", "evaluation_final_v2"),
    ("Aleatorio", "evaluation_random_v2"),
]
COLORS = ["#2878B5", "#E07A28", "#399553"]


def load_rows(folder):
    path = ROOT / "results" / folder / "evaluation.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"El archivo está vacío: {path}")

    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Hay semillas repetidas: {path}")

    return sorted(rows, key=lambda row: int(row["seed"]))


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    datasets = [load_rows(folder) for _, folder in EVALUATIONS]
    reference_seeds = [int(row["seed"]) for row in datasets[0]]

    for rows in datasets[1:]:
        if [int(row["seed"]) for row in rows] != reference_seeds:
            raise ValueError("Las evaluaciones deben usar las mismas semillas.")

    labels = [label for label, _ in EVALUATIONS]
    n = len(reference_seeds)
    panels = [
        ("reward", "Recompensa acumulada"),
        ("max_score", "Puntuación máxima del episodio"),
        ("steps", "Duración (pasos)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (field, title) in zip(axes, panels):
        for index, (rows, color) in enumerate(zip(datasets, COLORS)):
            values = np.array([float(row[field]) for row in rows])
            # Desplazamiento visual fijo: no modifica los datos.
            offsets = np.linspace(-0.14, 0.14, len(values))
            ax.scatter(
                index + offsets, values,
                color=color, alpha=0.65, s=30,
                label="Episodio" if index == 0 else None,
            )
            ax.scatter(
                index, values.mean(), marker="D",
                color="black", s=55, zorder=4,
                label="Media" if index == 0 else None,
            )
            ax.plot(
                [index - 0.20, index + 0.20],
                [np.median(values)] * 2,
                color="black", linewidth=2,
                label="Mediana" if index == 0 else None,
            )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Recompensa")
    axes[1].set_ylabel("Puntuación")
    axes[2].set_ylabel("Pasos")
    axes[0].legend(fontsize=9)
    fig.suptitle(
        f"Evaluación independiente: {n} episodios por política\n"
        f"Semillas {reference_seeds[0]}–{reference_seeds[-1]} · DQN con ε=0"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    comparison_path = ASSETS / "evaluation_comparison.png"
    fig.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(labels))
    action_fields = ["idle_count", "gas_count", "reverse_count"]
    action_labels = ["Sin acelerar", "Acelerar", "Reversa"]
    action_colors = ["#A9AFB5", "#2878B5", "#E07A28"]

    for field, label, color in zip(
        action_fields, action_labels, action_colors
    ):
        percentages = np.array([
            100 * sum(int(row[field]) for row in rows)
            / sum(int(row["steps"]) for row in rows)
            for rows in datasets
        ])
        ax.bar(
            labels, percentages, bottom=bottom,
            label=label, color=color,
        )
        for index, percentage in enumerate(percentages):
            if percentage >= 5:
                ax.text(
                    index, bottom[index] + percentage / 2,
                    f"{percentage:.1f}%",
                    ha="center", va="center",
                )
        bottom += percentages

    ax.set_ylim(0, 100)
    ax.set_ylabel("Porcentaje de los pasos evaluados")
    ax.set_title("Distribución de acciones por política")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    fig.tight_layout()
    actions_path = ASSETS / "evaluation_actions.png"
    fig.savefig(actions_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura guardada: {comparison_path}")
    print(f"Figura guardada: {actions_path}")
    print(f"Episodios por política: {n}")


if __name__ == "__main__":
    main()