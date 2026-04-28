# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL checkpoint on task-level metrics."""

"""Launch Isaac Sim Simulator first."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RL agent checkpoint with task metrics.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import test.tasks  # noqa: F401


def _safe_mean(total: float, count: int) -> float:
    return total / count if count > 0 else 0.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate a trained RSL-RL agent."""

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Eval", "").replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=base_env.device)

    reward_names = list(base_env.reward_manager.active_terms)
    reward_indices = {name: idx for idx, name in enumerate(reward_names)}
    reward_weights = {name: base_env.reward_manager.get_term_cfg(name).weight for name in reward_names}

    episode_return = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_steps = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
    episode_stumble = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_thigh_contact = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_shank_contact = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_lin_vel_error = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_ang_vel_error = torch.zeros(base_env.num_envs, device=base_env.device)

    finished_episodes = 0
    total_return = 0.0
    total_episode_length_s = 0.0
    total_stumble = 0.0
    total_thigh_contact = 0.0
    total_shank_contact = 0.0
    total_lin_vel_error = 0.0
    total_ang_vel_error = 0.0
    total_time_outs = 0
    total_base_contacts = 0
    log_metric_totals: defaultdict[str, float] = defaultdict(float)

    def recover_unweighted_term(term_name: str) -> torch.Tensor:
        if term_name not in reward_indices:
            return torch.zeros(base_env.num_envs, device=base_env.device)
        weight = reward_weights[term_name]
        if weight == 0.0:
            return torch.zeros(base_env.num_envs, device=base_env.device)
        values = base_env.reward_manager._step_reward[:, reward_indices[term_name]] / weight
        return torch.clamp(values, min=0.0)

    obs = vec_env.get_observations()

    while simulation_app.is_running() and finished_episodes < args_cli.num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = vec_env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

        command = base_env.command_manager.get_command("base_velocity")
        robot = base_env.scene["robot"]

        episode_return += rewards
        episode_steps += 1
        episode_stumble += recover_unweighted_term("stumble_penalty")
        episode_thigh_contact += recover_unweighted_term("undesired_contacts")
        episode_shank_contact += recover_unweighted_term("undesired_shank_contacts")
        episode_lin_vel_error += torch.norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
        episode_ang_vel_error += torch.abs(command[:, 2] - robot.data.root_ang_vel_b[:, 2])

        done_env_ids = (base_env.reset_terminated | base_env.reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
        num_done = int(done_env_ids.numel())
        if num_done == 0:
            continue

        log_info = base_env.extras.get("log", {})
        for key, value in log_info.items():
            if key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/"):
                log_metric_totals[key] += float(value) * num_done

        base_contact_term = base_env.termination_manager.get_term("base_contact")
        remaining = args_cli.num_episodes - finished_episodes
        take = min(num_done, remaining)
        selected_ids = done_env_ids[:take]

        step_count = episode_steps[selected_ids].clamp(min=1).float()
        total_return += episode_return[selected_ids].sum().item()
        total_episode_length_s += (step_count * base_env.step_dt).sum().item()
        total_stumble += episode_stumble[selected_ids].sum().item()
        total_thigh_contact += episode_thigh_contact[selected_ids].sum().item()
        total_shank_contact += episode_shank_contact[selected_ids].sum().item()
        total_lin_vel_error += (episode_lin_vel_error[selected_ids] / step_count).sum().item()
        total_ang_vel_error += (episode_ang_vel_error[selected_ids] / step_count).sum().item()
        total_time_outs += int(base_env.reset_time_outs[selected_ids].sum().item())
        total_base_contacts += int(base_contact_term[selected_ids].sum().item())
        finished_episodes += take

        episode_return[done_env_ids] = 0.0
        episode_steps[done_env_ids] = 0
        episode_stumble[done_env_ids] = 0.0
        episode_thigh_contact[done_env_ids] = 0.0
        episode_shank_contact[done_env_ids] = 0.0
        episode_lin_vel_error[done_env_ids] = 0.0
        episode_ang_vel_error[done_env_ids] = 0.0

    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "episodes": finished_episodes,
        "mean_episode_return": _safe_mean(total_return, finished_episodes),
        "mean_episode_length_s": _safe_mean(total_episode_length_s, finished_episodes),
        "timeout_rate": _safe_mean(total_time_outs, finished_episodes),
        "base_contact_rate": _safe_mean(total_base_contacts, finished_episodes),
        "mean_stumble_events_per_episode": _safe_mean(total_stumble, finished_episodes),
        "mean_thigh_contact_events_per_episode": _safe_mean(total_thigh_contact, finished_episodes),
        "mean_shank_contact_events_per_episode": _safe_mean(total_shank_contact, finished_episodes),
        "mean_lin_vel_tracking_error": _safe_mean(total_lin_vel_error, finished_episodes),
        "mean_ang_vel_tracking_error": _safe_mean(total_ang_vel_error, finished_episodes),
        "logged_reward_terms_per_second": {},
        "logged_termination_rates": {},
    }

    for key in sorted(log_metric_totals):
        mean_value = _safe_mean(log_metric_totals[key], finished_episodes)
        if key.startswith("Episode_Reward/"):
            summary["logged_reward_terms_per_second"][key.removeprefix("Episode_Reward/")] = mean_value
        elif key.startswith("Episode_Termination/"):
            summary["logged_termination_rates"][key.removeprefix("Episode_Termination/")] = mean_value

    print("[INFO] Evaluation summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    eval_dir = os.path.join(log_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    summary_path = os.path.join(eval_dir, f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    print(f"[INFO] Saved evaluation summary to: {summary_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()