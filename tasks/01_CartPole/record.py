"""学習チェックポイント時の greedy グリッド動画（imageio でフレーム逐次書き出し）。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from deep_reinforcement_learning_lab.config import ExperimentConfig, RunConfig
from env import subtask_render_fps
from policy import GreedyEpisodeResult, run_greedy_episode

PLAYBACK_SPEED = 0.5
FAILURE_BRIGHTNESS = 0.25
ALL_FAILED_HOLD_FRAMES = 15
GRID_LINE_WIDTH = 2
GRID_LINE_COLOR = np.array((32, 32, 32), dtype=np.uint8)
STEP_OVERLAY_COLOR = (0, 0, 0)
STEP_OVERLAY_FONT_DIVISOR = 3
STEP_OVERLAY_MIN_FONT_SIZE = 36
STEP_OVERLAY_MARGIN_DIVISOR = 24
_STEP_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)


@dataclass(frozen=True, slots=True)
class _GridTimeline:
    """グリッド動画の長さと暗転ルール。"""

    all_episodes_fell: bool
    last_failure_frame: int
    frame_count: int

    @classmethod
    def from_rollouts(cls, rollouts: tuple[GreedyEpisodeResult, ...]) -> _GridTimeline:
        longest = max(len(rollout.frames) for rollout in rollouts)
        all_fell = all(rollout.ended_by_termination for rollout in rollouts)
        if not all_fell:
            return cls(all_episodes_fell=False, last_failure_frame=-1, frame_count=longest)
        failure_frames = [
            rollout.fail_frame_index
            for rollout in rollouts
            if rollout.fail_frame_index is not None
        ]
        if not failure_frames:
            raise ValueError("all episodes fell but no failure frame index recorded")
        last_failure = max(failure_frames)
        frame_count = max(longest, last_failure + 1 + ALL_FAILED_HOLD_FRAMES)
        return cls(
            all_episodes_fell=True,
            last_failure_frame=last_failure,
            frame_count=frame_count,
        )

    def should_darken(self, rollout: GreedyEpisodeResult, time_index: int) -> bool:
        if rollout.fail_frame_index is not None and time_index >= rollout.fail_frame_index:
            return True
        if self.all_episodes_fell and time_index >= self.last_failure_frame:
            return True
        return False


def should_record_video(env_step: int, run: RunConfig) -> bool:
    if not run.video_enabled:
        return False
    return env_step % run.video_interval_steps == 0


def video_output_path(log_dir: Path, env_step: int) -> Path:
    return log_dir / "videos" / f"train_step{env_step:07d}.mp4"


def _darken(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * FAILURE_BRIGHTNESS, 0, 255).astype(np.uint8)


def _cell_short_side(cell_height: int, cell_width: int) -> int:
    return min(cell_height, cell_width)


def _overlay_margin(cell_height: int, cell_width: int) -> int:
    return max(8, _cell_short_side(cell_height, cell_width) // STEP_OVERLAY_MARGIN_DIVISOR)


def _overlay_font_size(cell_height: int, cell_width: int) -> int:
    return max(
        STEP_OVERLAY_MIN_FONT_SIZE,
        _cell_short_side(cell_height, cell_width) // STEP_OVERLAY_FONT_DIVISOR,
    )


@lru_cache(maxsize=8)
def _overlay_font(cell_height: int, cell_width: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = _overlay_font_size(cell_height, cell_width)
    for path in _STEP_FONT_CANDIDATES:
        if not Path(path).is_file():
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _overlay_cell_step(
    cell: np.ndarray,
    step: int,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    margin: int,
) -> np.ndarray:
    cell_h, cell_w = int(cell.shape[0]), int(cell.shape[1])
    image = Image.fromarray(np.ascontiguousarray(cell[..., :3], dtype=np.uint8))
    ImageDraw.Draw(image).text(
        (cell_w - margin, margin),
        str(step),
        fill=STEP_OVERLAY_COLOR,
        font=font,
        anchor="rt",
    )
    return np.asarray(image)


def _render_cell(
    rollout: GreedyEpisodeResult,
    time_index: int,
    timeline: _GridTimeline,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    margin: int,
) -> np.ndarray:
    frame_index = min(time_index, len(rollout.frames) - 1)
    cell = rollout.frames[frame_index]
    if timeline.should_darken(rollout, time_index):
        cell = _darken(cell)
    return _overlay_cell_step(
        cell,
        rollout.display_step_at(time_index),
        font=font,
        margin=margin,
    )


def _draw_grid_separators(
    grid: np.ndarray,
    grid_side: int,
    cell_h: int,
    cell_w: int,
) -> None:
    half = GRID_LINE_WIDTH // 2
    for index in range(1, grid_side):
        y_center = index * cell_h
        y0 = max(y_center - half, 0)
        y1 = min(y_center - half + GRID_LINE_WIDTH, grid.shape[0])
        grid[y0:y1, :, :] = GRID_LINE_COLOR

        x_center = index * cell_w
        x0 = max(x_center - half, 0)
        x1 = min(x_center - half + GRID_LINE_WIDTH, grid.shape[1])
        grid[:, x0:x1, :] = GRID_LINE_COLOR


def _compose_grid_frame(
    rollouts: tuple[GreedyEpisodeResult, ...],
    time_index: int,
    grid_side: int,
    timeline: _GridTimeline,
) -> np.ndarray:
    first_frame = rollouts[0].frames[0]
    cell_h, cell_w = first_frame.shape[:2]
    font = _overlay_font(cell_h, cell_w)
    margin = _overlay_margin(cell_h, cell_w)
    grid = np.zeros((grid_side * cell_h, grid_side * cell_w, 3), dtype=np.uint8)
    for index, rollout in enumerate(rollouts):
        row, col = divmod(index, grid_side)
        cell = _render_cell(
            rollout,
            time_index,
            timeline,
            font=font,
            margin=margin,
        )
        y0, y1 = row * cell_h, (row + 1) * cell_h
        x0, x1 = col * cell_w, (col + 1) * cell_w
        grid[y0:y1, x0:x1] = cell
    _draw_grid_separators(grid, grid_side, cell_h, cell_w)
    return grid


def _iter_grid_video_frames(
    rollouts: tuple[GreedyEpisodeResult, ...],
    grid_side: int,
) -> Iterator[np.ndarray]:
    if not rollouts[0].frames:
        raise ValueError("grid video requires rendered rollouts")
    timeline = _GridTimeline.from_rollouts(rollouts)
    for time_index in range(timeline.frame_count):
        yield _compose_grid_frame(rollouts, time_index, grid_side, timeline)


def _write_grid_video(
    frames: Iterator[np.ndarray],
    output_path: Path,
    fps: int,
) -> None:
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


@torch.no_grad()
def record_greedy_grid_video(
    q_model: nn.Module,
    experiment: ExperimentConfig,
    log_dir: Path,
    env_step: int,
) -> Path:
    """``video_episodes`` 本の greedy をグリッド合成し MP4 を書き出す（マスごとにステップ表示・失敗時暗転）。"""
    was_training = q_model.training
    q_model.eval()

    (log_dir / "videos").mkdir(parents=True, exist_ok=True)
    output_path = video_output_path(log_dir, env_step)

    run = experiment.run
    grid_side = run.video_grid_side
    rollouts = tuple(
        run_greedy_episode(
            q_model,
            subtask=experiment.subtask,
            seed=run.video_rollout_seed(episode),
            render=True,
        )
        for episode in range(run.video_episodes)
    )
    video_fps = max(1, int(subtask_render_fps(experiment.subtask) * PLAYBACK_SPEED))
    _write_grid_video(
        _iter_grid_video_frames(rollouts, grid_side),
        output_path,
        video_fps,
    )

    q_model.train(was_training)
    return output_path
