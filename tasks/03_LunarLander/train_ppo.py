from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.algorithms.PPO import (
    ContinuousPPO,
    PPO,
    VectorContinuousFixedRolloutBuffer,
    VectorDiscreteFixedRolloutBuffer,
)
from deep_reinforcement_learning_lab.algorithms.registry import TrainingContext
from deep_reinforcement_learning_lab.config import (
    ExperimentConfig,
    PPOHyperparameters,
    RunConfig,
    ensure_run_dir_is_new,
)
from ac_model_registry import PpoKind, ppo_kind_for_model, resolve_rollout_spec
from config_loader import load_model_config_from_path
from model_config import ModelConfig
from run_paths import resolve_run_dir
from checkpoint import BestCheckpointTracker, checkpoint_path, save_ac_checkpoint
from eval_export import build_eval_export, save_eval_final
from console import print_progress_line, print_run_footer, print_run_header
from env import LUNAR_ENV_ACTION_DIM, make_sync_vector_env
from factorized_ppo import FactorizedPPO, VectorFactorizedRolloutBuffer
from logger import TrainingLogger, attach_training_loss_logging
from models import build_model_for_experiment
from policy import EvalMetrics, evaluate_policy
from record import record_deterministic_grid_video, should_record_video


class _VectorPpoAgent(Protocol):
    def select_actions(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def step_vector(self, *args: Any, **kwargs: Any) -> None: ...

    def flush_if_pending(self, states: torch.Tensor) -> None: ...


@dataclass(frozen=True)
class _TrainingSetup:
    ac_model: nn.Module
    agent: _VectorPpoAgent
    state_dim: int
    rollout_action_dim: int
    action_low: torch.Tensor | None
    action_high: torch.Tensor | None


def should_log_metrics(env_step: int, run: RunConfig, n_envs: int) -> bool:
    return env_step == n_envs or env_step % run.log_interval_steps == 0


def log_eval_checkpoint_progress(
    env_step: int,
    train_reward: float,
    *,
    experiment: ExperimentConfig,
    ac_model: nn.Module,
    logger: TrainingLogger,
    tracker: BestCheckpointTracker,
    run_dir: Path,
    setup: _TrainingSetup,
    ppo_kind: PpoKind,
    model: ModelConfig,
    record_video: bool,
    report_progress: bool = True,
) -> EvalMetrics:
    metrics = evaluate_policy(ac_model, experiment)
    logger.log_eval_metrics(env_step, metrics)
    saved_best = tracker.update_if_better(
        metrics,
        env_step,
        experiment=experiment,
        log_dir=run_dir,
        ac_model=ac_model,
        state_dim=setup.state_dim,
        rollout_action_dim=setup.rollout_action_dim,
        ppo_kind=ppo_kind,
        model=model,
        action_low=setup.action_low,
        action_high=setup.action_high,
    )

    note_parts: list[str] = []
    if saved_best:
        note_parts.append("best ckpt")
    if record_video:
        video_path = record_deterministic_grid_video(
            ac_model,
            experiment,
            run_dir,
            env_step,
        )
        note_parts.append(f"video -> {video_path.name}")

    if report_progress:
        print_progress_line(
            env_step,
            train_reward,
            metrics,
            note="  ".join(note_parts) or None,
        )
    return metrics


def _require_ppo_hparams(experiment: ExperimentConfig) -> PPOHyperparameters:
    hparams = experiment.hyperparameters
    if not isinstance(hparams, PPOHyperparameters):
        raise TypeError("train_ppo requires PPOHyperparameters")
    return hparams


def _build_training_setup(
    experiment: ExperimentConfig,
    model: ModelConfig,
    env: Any,
) -> _TrainingSetup:
    hparams = _require_ppo_hparams(experiment)

    state_dim = int(env.single_observation_space.shape[0])
    env_action_dim = int(env.single_action_space.shape[0])
    if env_action_dim != LUNAR_ENV_ACTION_DIM:
        raise ValueError(f"unexpected env action_dim: {env_action_dim}")

    rollout_spec = resolve_rollout_spec(model, env_action_dim=env_action_dim)
    ppo_kind = rollout_spec.ppo_kind
    rollout_action_dim = rollout_spec.stored_action_dim

    action_low: torch.Tensor | None = None
    action_high: torch.Tensor | None = None
    if ppo_kind == "continuous":
        action_low = torch.tensor(env.single_action_space.low, dtype=torch.float32)
        action_high = torch.tensor(env.single_action_space.high, dtype=torch.float32)
    ac_model = build_model_for_experiment(
        experiment,
        model,
        state_dim,
        env_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    optimizer = torch.optim.Adam(ac_model.parameters(), lr=hparams.learning_rate)

    if ppo_kind == "discrete":
        rollout_buffer = VectorDiscreteFixedRolloutBuffer(
            hparams.rollout_steps,
            hparams.n_envs,
            state_dim,
        )
        agent: _VectorPpoAgent = PPO(
            ac_model,
            optimizer,
            rollout_buffer,
            gamma=hparams.gamma,
            gae_lambda=hparams.gae_lambda,
            clip_coef=hparams.clip_coef,
            entropy_coef=hparams.entropy_coef,
            value_coef=hparams.value_coef,
            rollout_steps=hparams.rollout_steps,
            n_epochs=hparams.n_epochs,
            minibatch_size=hparams.minibatch_size,
            clip_value_coef=hparams.clip_value_coef,
            max_grad_norm=hparams.max_grad_norm,
        )
    elif ppo_kind == "factorized":
        rollout_buffer = VectorFactorizedRolloutBuffer(
            hparams.rollout_steps,
            hparams.n_envs,
            state_dim,
            rollout_action_dim,
        )
        agent = FactorizedPPO(
            ac_model,
            optimizer,
            rollout_buffer,
            gamma=hparams.gamma,
            gae_lambda=hparams.gae_lambda,
            clip_coef=hparams.clip_coef,
            entropy_coef=hparams.entropy_coef,
            value_coef=hparams.value_coef,
            rollout_steps=hparams.rollout_steps,
            n_epochs=hparams.n_epochs,
            minibatch_size=hparams.minibatch_size,
            clip_value_coef=hparams.clip_value_coef,
            max_grad_norm=hparams.max_grad_norm,
        )
    else:
        rollout_buffer = VectorContinuousFixedRolloutBuffer(
            hparams.rollout_steps,
            hparams.n_envs,
            state_dim,
            rollout_action_dim,
        )
        agent = ContinuousPPO(
            ac_model,
            optimizer,
            rollout_buffer,
            gamma=hparams.gamma,
            gae_lambda=hparams.gae_lambda,
            clip_coef=hparams.clip_coef,
            entropy_coef=hparams.entropy_coef,
            value_coef=hparams.value_coef,
            rollout_steps=hparams.rollout_steps,
            n_epochs=hparams.n_epochs,
            minibatch_size=hparams.minibatch_size,
            clip_value_coef=hparams.clip_value_coef,
            max_grad_norm=hparams.max_grad_norm,
        )

    return _TrainingSetup(
        ac_model=ac_model,
        agent=agent,
        state_dim=state_dim,
        rollout_action_dim=rollout_action_dim,
        action_low=action_low,
        action_high=action_high,
    )


def _run_ppo_training(
    context: TrainingContext,
    model: ModelConfig,
) -> float:
    experiment = context.experiment
    run = experiment.run
    hparams = _require_ppo_hparams(experiment)
    n_envs = hparams.n_envs

    torch.manual_seed(run.seed)
    env = make_sync_vector_env(experiment, n_envs, seed=run.seed)
    setup = _build_training_setup(experiment, model, env)
    ppo_kind = ppo_kind_for_model(model)

    run_dir = resolve_run_dir(context.task_dir, experiment)
    ensure_run_dir_is_new(run_dir)
    tracker = BestCheckpointTracker()

    print_run_header(experiment, context.config_path, run_dir, context.task_dir, model)

    observation, _ = env.reset(seed=run.seed)
    episode_rewards = np.zeros(n_envs, dtype=np.float64)
    episode_lengths = np.zeros(n_envs, dtype=np.int64)
    last_completed_train_reward = 0.0
    env_step = 0

    with TrainingLogger(log_dir=run_dir) as logger:
        attach_training_loss_logging(setup.agent, logger)
        logger.log_experiment_start(experiment, context.config_path)

        while env_step < run.max_env_steps:
            states = torch.as_tensor(observation, dtype=torch.float32)
            actions, log_probs, values_old = setup.agent.select_actions(states)
            env_actions = setup.ac_model.rollout_actions_to_env(actions)
            next_observation, rewards, terminated, truncated, _ = env.step(
                env_actions.detach().cpu().numpy().astype(np.float32)
            )
            bootstrap_done = terminated.astype(np.float32)

            setup.agent.step_vector(
                states,
                actions,
                log_probs,
                torch.as_tensor(rewards, dtype=torch.float32),
                torch.as_tensor(bootstrap_done, dtype=torch.float32),
                values_old,
                torch.as_tensor(next_observation, dtype=torch.float32),
            )
            observation = next_observation
            env_step += n_envs

            for env_index in range(n_envs):
                episode_rewards[env_index] += rewards[env_index]
                episode_lengths[env_index] += 1
                if terminated[env_index] or truncated[env_index]:
                    last_completed_train_reward = float(episode_rewards[env_index])
                    logger.log_train_episode(
                        env_step,
                        last_completed_train_reward,
                        int(episode_lengths[env_index]),
                    )
                    episode_rewards[env_index] = 0.0
                    episode_lengths[env_index] = 0

            if should_log_metrics(env_step, run, n_envs):
                log_eval_checkpoint_progress(
                    env_step,
                    last_completed_train_reward,
                    experiment=experiment,
                    ac_model=setup.ac_model,
                    logger=logger,
                    tracker=tracker,
                    run_dir=run_dir,
                    setup=setup,
                    ppo_kind=ppo_kind,
                    model=model,
                    record_video=should_record_video(env_step, run),
                )

        setup.agent.flush_if_pending(torch.as_tensor(observation, dtype=torch.float32))
        env.close()

        final_metrics = log_eval_checkpoint_progress(
            env_step,
            last_completed_train_reward,
            experiment=experiment,
            ac_model=setup.ac_model,
            logger=logger,
            tracker=tracker,
            run_dir=run_dir,
            setup=setup,
            ppo_kind=ppo_kind,
            model=model,
            record_video=False,
            report_progress=False,
        )
        logger.log_experiment_end(experiment, final_metrics)

    export = build_eval_export(
        run_dir,
        experiment,
        tracker,
        setup.ac_model,
        env_step,
    )
    save_eval_final(
        run_dir,
        export.metrics,
        experiment=experiment,
        config_path=context.config_path,
        env_step=export.env_step,
        model_id=model.id,
        ppo_kind=ppo_kind,
        model_options=dict(model.options),
        eval_source=export.eval_source,
        training_end_env_step=env_step,
    )

    ckpt: Path | None = None
    if run.save_checkpoint:
        ckpt = save_ac_checkpoint(
            checkpoint_path(run_dir),
            setup.ac_model,
            experiment,
            setup.state_dim,
            setup.rollout_action_dim,
            ppo_kind,
            model,
            action_low=setup.action_low,
            action_high=setup.action_high,
            env_step=env_step,
            eval_mean_episode_return=final_metrics.mean_episode_return,
            eval_success_rate=final_metrics.success_rate,
        )

    print_run_footer(experiment, env_step, export.metrics, ckpt, tracker)
    return export.metrics.mean_episode_return


def train_ppo(context: TrainingContext) -> float:
    experiment = context.experiment
    model = load_model_config_from_path(context.config_path)
    if experiment.algorithm != "ppo":
        raise ValueError(f"expected algorithm ppo, got {experiment.algorithm}")
    ppo_kind_for_model(model)
    return _run_ppo_training(context, model)
