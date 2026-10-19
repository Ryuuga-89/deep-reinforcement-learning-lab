"""実験 1 の図を report/assets に書き出す。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TASK_DIR = Path(__file__).resolve().parent
RUNS = TASK_DIR / "runs" / "default" / "ppo"
ASSETS = TASK_DIR / "report" / "assets"

CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("baseline_continuous", "Continuous", "#0072B2"),
    ("table_main_lateral_6", "Table", "#E69F00"),
    ("binned_gymnasium_style", "Binned", "#009E73"),
)


def _load_eval_success(seed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(seed_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars("Eval/success_rate")
    steps = np.array([event.step for event in events], dtype=np.int64)
    values = np.array([event.value for event in events], dtype=np.float64)
    return steps, values


def _curves(run_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns steps (T,) and values (n_seeds, T) on steps shared by every seed."""
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for seed_dir in sorted((RUNS / run_name).glob("seed_*")):
        series.append(_load_eval_success(seed_dir))
    if not series:
        raise FileNotFoundError(f"no seed runs under {RUNS / run_name}")
    common = set(series[0][0].tolist())
    for steps, _values in series[1:]:
        common &= set(steps.tolist())
    # 終了時の last-weight 評価（500000）は log 間隔の点ではないので除く。
    common = {step for step in common if step == 4 or step % 32768 == 0}
    steps = np.array(sorted(common), dtype=np.int64)
    aligned = np.stack(
        [values[np.isin(seed_steps, steps)] for seed_steps, values in series]
    )
    return steps, aligned


def _final_success(run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    for path in sorted((RUNS / run_name).glob("seed_*/eval_final.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores[int(payload["seed"])] = float(payload["metrics"]["success_rate"])
    if len(scores) != 10:
        raise ValueError(f"{run_name}: expected 10 seeds, got {len(scores)}")
    return scores


def _plot_learning_curves(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for run_name, label, color in CONDITIONS:
        steps, values = _curves(run_name)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        ax.plot(steps, mean, color=color, label=label, linewidth=2.0)
        ax.fill_between(
            steps,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xlim(0, 500_000)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval success rate")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_paired_final(path: Path) -> None:
    scores = {run_name: _final_success(run_name) for run_name, _label, _color in CONDITIONS}
    seeds = sorted(scores["baseline_continuous"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs = np.arange(len(CONDITIONS))
    for seed in seeds:
        ys = [scores[run_name][seed] for run_name, _label, _color in CONDITIONS]
        ax.plot(xs, ys, color="#666666", alpha=0.45, linewidth=1.0, zorder=1)
    for index, (run_name, _label, color) in enumerate(CONDITIONS):
        ys = [scores[run_name][seed] for seed in seeds]
        ax.scatter(
            np.full(len(seeds), index),
            ys,
            color=color,
            s=36,
            zorder=2,
            label=CONDITIONS[index][1],
        )
    ax.set_xticks(xs, [label for _run, label, _color in CONDITIONS])
    ax.set_ylim(0.65, 1.02)
    ax.set_ylabel("Final success rate")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    curve_path = ASSETS / "exp1_eval_success_curve.png"
    paired_path = ASSETS / "exp1_final_success_paired.png"
    _plot_learning_curves(curve_path)
    _plot_paired_final(paired_path)
    print(curve_path)
    print(paired_path)


if __name__ == "__main__":
    main()
