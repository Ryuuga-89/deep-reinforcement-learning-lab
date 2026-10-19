"""実験 2: seed 42 の best 方策を同じ環境シードで並べ、GIF を report/assets に書く。"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from checkpoint import load_ac_checkpoint
from config_loader import load_lunar_experiment
from policy import DeterministicEpisodeResult, run_deterministic_episode
from record import _GridTimeline, _render_cell

TASK_DIR = Path(__file__).resolve().parent
RUNS = TASK_DIR / "runs" / "default" / "ppo"
ASSETS = TASK_DIR / "report" / "assets"
CELL_WIDTH = 280
FRAME_STRIDE = 3
GIF_FPS = 12
ENV_SEEDS = (142, 143)
CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("table_gymnasium_4", "table_gymnasium_4_train.yaml", "4-point"),
    ("table_main_lateral_6", "table_main_lateral_6_train.yaml", "6-point"),
    ("table_main5_lat3", "table_main5_lat3_train.yaml", "15-point"),
)


def _scale_rollout(rollout: DeterministicEpisodeResult, width: int) -> DeterministicEpisodeResult:
    frames: list[np.ndarray] = []
    for frame in rollout.frames:
        image = Image.fromarray(np.ascontiguousarray(frame[..., :3], dtype=np.uint8))
        height = max(1, round(image.height * width / image.width))
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        frames.append(np.asarray(resized))
    return DeterministicEpisodeResult(
        total_reward=rollout.total_reward,
        episode_steps=rollout.episode_steps,
        outcome=rollout.outcome,
        solved=rollout.solved,
        main_fuel_penalty=rollout.main_fuel_penalty,
        side_fuel_penalty=rollout.side_fuel_penalty,
        landing_abs_tilt=rollout.landing_abs_tilt,
        landing_abs_angular_speed=rollout.landing_abs_angular_speed,
        frames=tuple(frames),
        fuel_at_frames=rollout.fuel_at_frames,
        fail_frame_index=rollout.fail_frame_index,
        ended_by_termination=rollout.ended_by_termination,
    )


def _header(width: int, labels: tuple[str, ...]) -> np.ndarray:
    height = 36
    image = Image.new("RGB", (width * len(labels), height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    for index, label in enumerate(labels):
        draw.text(
            (index * width + width // 2, height // 2),
            label,
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
    return np.asarray(image)


def _compose(
    rows: tuple[tuple[DeterministicEpisodeResult, ...], ...],
    time_index: int,
) -> np.ndarray:
    font = ImageFont.load_default(size=16)
    margin = 6
    cells: list[np.ndarray] = []
    for row in rows:
        timeline = _GridTimeline.from_rollouts(row)
        rendered = tuple(
            _render_cell(rollout, time_index, timeline, font=font, margin=margin)
            for rollout in row
        )
        cells.append(np.concatenate(rendered, axis=1))
    grid = np.concatenate(cells, axis=0)
    header = _header(rows[0][0].frames[0].shape[1], tuple(label for _a, _b, label in CONDITIONS))
    return np.concatenate([header, grid], axis=0)


def main() -> None:
    labels_rollouts: list[list[DeterministicEpisodeResult]] = []
    for run_name, config_name, label in CONDITIONS:
        config_path = TASK_DIR / "configs" / "default" / config_name
        experiment = load_lunar_experiment(config_path).experiment
        checkpoint = RUNS / run_name / "seed_0042" / "ac_best.pt"
        model = load_ac_checkpoint(checkpoint)
        model.eval()
        row: list[DeterministicEpisodeResult] = []
        for env_seed in ENV_SEEDS:
            rollout = run_deterministic_episode(
                model,
                experiment=experiment,
                seed=env_seed,
                render=True,
            )
            scaled = _scale_rollout(rollout, CELL_WIDTH)
            row.append(scaled)
            print(
                f"{label} env_seed={env_seed} steps={rollout.episode_steps} "
                f"return={rollout.total_reward:.1f} outcome={rollout.outcome.value}"
            )
        labels_rollouts.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # rows are env seeds, columns are conditions
    by_seed: tuple[tuple[DeterministicEpisodeResult, ...], ...] = tuple(
        tuple(condition[seed_index] for condition in labels_rollouts)
        for seed_index in range(len(ENV_SEEDS))
    )
    frame_count = max(len(rollout.frames) for row in by_seed for rollout in row)
    frames = [
        _compose(by_seed, time_index)
        for time_index in range(0, frame_count, FRAME_STRIDE)
    ]
    ASSETS.mkdir(parents=True, exist_ok=True)
    output = ASSETS / "exp2_rollout_seed42.gif"
    imageio.mimsave(output, frames, format="GIF", fps=GIF_FPS, loop=0)
    print(f"wrote {output} ({output.stat().st_size} bytes, {len(frames)} frames)")


if __name__ == "__main__":
    main()
