# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL checkpoint on task-level metrics."""
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

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
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import test.tasks  # noqa: F401


def _safe_mean(total: float, count: int) -> float:
    return total / count if count > 0 else 0.0


def _safe_rate(total: int, count: int) -> float:
    return total / count if count > 0 else 0.0


def _compute_obstacle_scan_metrics(terrain_sensor, look_ahead_distance: float, obstacle_threshold: float):
    """Estimate obstacle geometry ahead of the robot from the height scanner."""

    ray_x = terrain_sensor.ray_starts[0, :, 0]
    ray_hits_z = terrain_sensor.data.ray_hits_w[..., 2]

    rear_mask = ray_x <= 0.0
    forward_mask = (ray_x > 0.0) & (ray_x <= look_ahead_distance)

    reference_ground_height = ray_hits_z[:, rear_mask].mean(dim=1)
    forward_hits_z = ray_hits_z[:, forward_mask]
    obstacle_top_height = forward_hits_z.max(dim=1).values
    obstacle_height = torch.clamp(obstacle_top_height - reference_ground_height, min=0.0)
    obstacle_present = obstacle_height > obstacle_threshold

    forward_ray_x = ray_x[forward_mask].to(device=ray_hits_z.device)
    elevated_hits = (forward_hits_z - reference_ground_height.unsqueeze(1)) > obstacle_threshold
    masked_ray_x = torch.where(
        elevated_hits,
        forward_ray_x.unsqueeze(0),
        torch.full((ray_hits_z.shape[0], forward_ray_x.numel()), float("inf"), device=ray_hits_z.device),
    )
    nearest_obstacle_distance = masked_ray_x.min(dim=1).values
    nearest_obstacle_distance = torch.where(
        torch.isfinite(nearest_obstacle_distance),
        nearest_obstacle_distance,
        torch.zeros_like(nearest_obstacle_distance),
    )

    return {
        "reference_ground_height": reference_ground_height,
        "obstacle_top_height": obstacle_top_height,
        "obstacle_height": obstacle_height,
        "obstacle_present": obstacle_present,
        "nearest_obstacle_distance": nearest_obstacle_distance,
    }


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
        checkpoint_arg = args_cli.checkpoint
        if os.path.isabs(checkpoint_arg) or os.path.dirname(checkpoint_arg) or "://" in checkpoint_arg:
            resume_path = retrieve_file_path(checkpoint_arg)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, checkpoint_arg)
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
    termination_names = list(getattr(base_env.termination_manager, "active_terms", ()))

    obstacle_metrics_enabled = all(
        key in reward_indices for key in ("foot_clearance", "undesired_shank_contacts")
    ) and all(sensor_name in base_env.scene.sensors for sensor_name in ("height_scanner", "contact_forces"))

    obstacle_metrics_summary: dict[str, object] = {"enabled": obstacle_metrics_enabled}
    if obstacle_metrics_enabled:
        terrain_sensor = base_env.scene.sensors["height_scanner"]
        contact_sensor = base_env.scene.sensors["contact_forces"]
        robot = base_env.scene["robot"]

        foot_clearance_params = base_env.reward_manager.get_term_cfg("foot_clearance").params
        foot_sensor_cfg = foot_clearance_params["sensor_cfg"]
        foot_asset_cfg = foot_clearance_params["asset_cfg"]
        obstacle_height_threshold = float(foot_clearance_params.get("obstacle_threshold", 0.02))
        look_ahead_distance = float(foot_clearance_params.get("look_ahead_distance", 0.6))
        forward_command_threshold = 0.1
        obstacle_clear_margin = 0.12
        obstacle_clear_steps_required = 3
        top_step_height_fraction = 0.5
        top_step_min_height = 0.02

        attempt_active = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.bool)
        attempt_has_shank_hit = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.bool)
        attempt_strategy = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        attempt_clear_steps = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        attempt_start_xy = torch.zeros(base_env.num_envs, 2, device=base_env.device)
        attempt_forward_dir = torch.zeros(base_env.num_envs, 2, device=base_env.device)
        attempt_required_progress = torch.zeros(base_env.num_envs, device=base_env.device)

        # 这些计数先按 episode 暂存，只有 episode 真正被纳入评估结果时才汇总到总计，
        # 这样可以避免最后一批并行环境超出目标 episode 数时把多余数据算进去。
        episode_obstacle_attempts = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        episode_successful_crossings = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        episode_direct_clear_successes = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        episode_shank_hit_first_attempts = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        episode_shank_hit_first_successes = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        episode_top_step_first_attempts = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        episode_top_step_first_successes = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        world_forward_axis = torch.tensor([1.0, 0.0, 0.0], device=base_env.device)

        total_obstacle_attempts = 0
        total_obstacle_successes = 0
        total_obstacle_failures = 0
        total_direct_clear_successes = 0
        total_shank_hit_first_attempts = 0
        total_shank_hit_first_successes = 0
        total_top_step_first_attempts = 0
        total_top_step_first_successes = 0
        total_episode_obstacle_attempts = 0
        total_episode_successful_crossings = 0

        def reset_attempt_state(env_ids: torch.Tensor):
            if env_ids.numel() == 0:
                return
            attempt_active[env_ids] = False
            attempt_has_shank_hit[env_ids] = False
            attempt_strategy[env_ids] = 0
            attempt_clear_steps[env_ids] = 0
            attempt_start_xy[env_ids] = 0.0
            attempt_forward_dir[env_ids] = 0.0
            attempt_required_progress[env_ids] = 0.0
    else:
        robot = base_env.scene["robot"]
        obstacle_metrics_summary["reason"] = "Missing foot_clearance/shank terms or height/contact sensors."

    episode_return = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_steps = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
    episode_stumble = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_thigh_contact = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_shank_contact = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_lin_vel_error = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_ang_vel_error = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_reward_term_sums = torch.zeros(base_env.num_envs, len(reward_names), device=base_env.device)

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
    reward_term_episode_means: defaultdict[str, float] = defaultdict(float)
    termination_term_totals: defaultdict[str, float] = defaultdict(float)

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

        episode_return += rewards
        episode_steps += 1
        episode_reward_term_sums += base_env.reward_manager._step_reward
        episode_stumble += recover_unweighted_term("stumble_penalty")
        episode_thigh_contact += recover_unweighted_term("undesired_contacts")
        episode_shank_contact += recover_unweighted_term("undesired_shank_contacts")
        episode_lin_vel_error += torch.norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
        episode_ang_vel_error += torch.abs(command[:, 2] - robot.data.root_ang_vel_b[:, 2])

        if obstacle_metrics_enabled:
            obstacle_scan = _compute_obstacle_scan_metrics(
                terrain_sensor,
                look_ahead_distance=look_ahead_distance,
                obstacle_threshold=obstacle_height_threshold,
            )
            obstacle_present = obstacle_scan["obstacle_present"]
            obstacle_height = obstacle_scan["obstacle_height"]
            obstacle_top_height = obstacle_scan["obstacle_top_height"]
            reference_ground_height = obstacle_scan["reference_ground_height"]
            nearest_obstacle_distance = obstacle_scan["nearest_obstacle_distance"]

            moving_forward = command[:, 0] > forward_command_threshold
            new_attempts = (~attempt_active) & obstacle_present & moving_forward
            if torch.any(new_attempts):
                episode_obstacle_attempts[new_attempts] += 1
                attempt_active[new_attempts] = True
                attempt_has_shank_hit[new_attempts] = False
                attempt_strategy[new_attempts] = 0
                attempt_clear_steps[new_attempts] = 0
                attempt_start_xy[new_attempts] = robot.data.root_pos_w[new_attempts, :2]
                forward_world = quat_apply(
                    robot.data.root_quat_w[new_attempts], world_forward_axis.expand(int(new_attempts.sum().item()), -1)
                )[:, :2]
                forward_world = forward_world / torch.norm(forward_world, dim=1, keepdim=True).clamp(min=1.0e-6)
                attempt_forward_dir[new_attempts] = forward_world
                attempt_required_progress[new_attempts] = nearest_obstacle_distance[new_attempts] + obstacle_clear_margin

            shank_contact_now = recover_unweighted_term("undesired_shank_contacts") > 0.0
            attempt_has_shank_hit |= attempt_active & shank_contact_now

            foot_first_contact = contact_sensor.compute_first_contact(base_env.step_dt)[:, foot_sensor_cfg.body_ids]
            foot_height = robot.data.body_pos_w[:, foot_asset_cfg.body_ids, 2]
            top_step_height_threshold = reference_ground_height.unsqueeze(1) + torch.maximum(
                obstacle_height.unsqueeze(1) * top_step_height_fraction,
                torch.full((base_env.num_envs, 1), top_step_min_height, device=base_env.device),
            )
            foot_on_top_now = (
                obstacle_present.unsqueeze(1)
                & foot_first_contact
                & (foot_height >= top_step_height_threshold)
                & (foot_height <= obstacle_top_height.unsqueeze(1) + 0.08)
            )
            top_step_now = torch.any(foot_on_top_now, dim=1)

            new_shank_first = attempt_active & (attempt_strategy == 0) & shank_contact_now
            episode_shank_hit_first_attempts[new_shank_first] += 1
            attempt_strategy[new_shank_first] = 1
            new_top_step_first = attempt_active & (attempt_strategy == 0) & (~shank_contact_now) & top_step_now
            episode_top_step_first_attempts[new_top_step_first] += 1
            attempt_strategy[new_top_step_first] = 2

            progress_along_attempt = torch.sum(
                (robot.data.root_pos_w[:, :2] - attempt_start_xy) * attempt_forward_dir,
                dim=1,
            )
            clear_now = attempt_active & (~obstacle_present) & (progress_along_attempt >= attempt_required_progress)
            attempt_clear_steps[attempt_active & (~clear_now)] = 0
            attempt_clear_steps[clear_now] += 1
            successful_crossings = attempt_active & (attempt_clear_steps >= obstacle_clear_steps_required)

            if torch.any(successful_crossings):
                episode_successful_crossings[successful_crossings] += 1
                direct_clear_successes = successful_crossings & (attempt_strategy == 0)
                shank_hit_first_successes = successful_crossings & (attempt_strategy == 1)
                top_step_first_successes = successful_crossings & (attempt_strategy == 2)

                episode_direct_clear_successes[direct_clear_successes] += 1
                episode_shank_hit_first_successes[shank_hit_first_successes] += 1
                episode_top_step_first_successes[top_step_first_successes] += 1
                reset_attempt_state(successful_crossings.nonzero(as_tuple=False).squeeze(-1))

        done_env_ids = (base_env.reset_terminated | base_env.reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
        num_done = int(done_env_ids.numel())
        if num_done == 0:
            continue

        base_contact_term = base_env.termination_manager.get_term("base_contact")
        remaining = args_cli.num_episodes - finished_episodes
        take = min(num_done, remaining)
        selected_ids = done_env_ids[:take]

        step_count = episode_steps[selected_ids].clamp(min=1).float()
        reward_term_mean_per_episode = episode_reward_term_sums[selected_ids] / step_count.unsqueeze(1)

        total_return += episode_return[selected_ids].sum().item()
        total_episode_length_s += (step_count * base_env.step_dt).sum().item()
        total_stumble += episode_stumble[selected_ids].sum().item()
        total_thigh_contact += episode_thigh_contact[selected_ids].sum().item()
        total_shank_contact += episode_shank_contact[selected_ids].sum().item()
        total_lin_vel_error += (episode_lin_vel_error[selected_ids] / step_count).sum().item()
        total_ang_vel_error += (episode_ang_vel_error[selected_ids] / step_count).sum().item()
        total_time_outs += int(base_env.reset_time_outs[selected_ids].sum().item())
        total_base_contacts += int(base_contact_term[selected_ids].sum().item())
        for reward_idx, reward_name in enumerate(reward_names):
            reward_term_episode_means[reward_name] += reward_term_mean_per_episode[:, reward_idx].sum().item()
        for termination_name in termination_names:
            if termination_name == "time_out":
                termination_values = base_env.reset_time_outs[selected_ids].float()
            else:
                termination_values = base_env.termination_manager.get_term(termination_name)[selected_ids].float()
            termination_term_totals[termination_name] += termination_values.sum().item()
        if obstacle_metrics_enabled:
            total_obstacle_attempts += int(episode_obstacle_attempts[selected_ids].sum().item())
            total_obstacle_successes += int(episode_successful_crossings[selected_ids].sum().item())
            total_direct_clear_successes += int(episode_direct_clear_successes[selected_ids].sum().item())
            total_shank_hit_first_attempts += int(episode_shank_hit_first_attempts[selected_ids].sum().item())
            total_shank_hit_first_successes += int(episode_shank_hit_first_successes[selected_ids].sum().item())
            total_top_step_first_attempts += int(episode_top_step_first_attempts[selected_ids].sum().item())
            total_top_step_first_successes += int(episode_top_step_first_successes[selected_ids].sum().item())
            total_episode_obstacle_attempts += int(episode_obstacle_attempts[selected_ids].sum().item())
            total_episode_successful_crossings += int(episode_successful_crossings[selected_ids].sum().item())

            # 走到 episode 结束仍未完成清障的 attempt，统一记为失败，
            # 且只统计本次真正纳入评估结果的 selected_ids，避免尾批次 spillover。
            failed_attempts = selected_ids[attempt_active[selected_ids]]
            total_obstacle_failures += int(failed_attempts.numel())
            reset_attempt_state(failed_attempts)
        previous_finished_episodes = finished_episodes
        finished_episodes += take

        for completed_episodes in range(previous_finished_episodes + 1, finished_episodes + 1):
            print(
                f"[INFO] Eval progress: {completed_episodes}/{args_cli.num_episodes} episodes completed",
                flush=True,
            )

        episode_return[done_env_ids] = 0.0
        episode_steps[done_env_ids] = 0
        episode_stumble[done_env_ids] = 0.0
        episode_thigh_contact[done_env_ids] = 0.0
        episode_shank_contact[done_env_ids] = 0.0
        episode_lin_vel_error[done_env_ids] = 0.0
        episode_ang_vel_error[done_env_ids] = 0.0
        episode_reward_term_sums[done_env_ids] = 0.0
        if obstacle_metrics_enabled:
            episode_obstacle_attempts[done_env_ids] = 0
            episode_successful_crossings[done_env_ids] = 0
            episode_direct_clear_successes[done_env_ids] = 0
            episode_shank_hit_first_attempts[done_env_ids] = 0
            episode_shank_hit_first_successes[done_env_ids] = 0
            episode_top_step_first_attempts[done_env_ids] = 0
            episode_top_step_first_successes[done_env_ids] = 0
            reset_attempt_state(done_env_ids)

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
        "obstacle_crossing_metrics": obstacle_metrics_summary,
    }

    if obstacle_metrics_enabled:
        obstacle_metrics_summary.update(
            {
                "obstacle_attempts": total_obstacle_attempts,
                "successful_crossings": total_obstacle_successes,
                "failed_crossings": total_obstacle_failures,
                "overall_crossing_success_rate": _safe_rate(total_obstacle_successes, total_obstacle_attempts),
                "direct_clear_successes": total_direct_clear_successes,
                "direct_clear_success_rate": _safe_rate(total_direct_clear_successes, total_obstacle_attempts),
                "direct_clear_share_of_successes": _safe_rate(total_direct_clear_successes, total_obstacle_successes),
                "shank_hit_first_attempts": total_shank_hit_first_attempts,
                "shank_hit_then_cross_successes": total_shank_hit_first_successes,
                "shank_hit_then_cross_rate": _safe_rate(
                    total_shank_hit_first_successes, total_shank_hit_first_attempts
                ),
                "shank_hit_then_cross_share_of_attempts": _safe_rate(
                    total_shank_hit_first_successes, total_obstacle_attempts
                ),
                "shank_hit_then_cross_share_of_successes": _safe_rate(
                    total_shank_hit_first_successes, total_obstacle_successes
                ),
                "top_step_first_attempts": total_top_step_first_attempts,
                "top_step_first_then_cross_successes": total_top_step_first_successes,
                "top_step_first_then_cross_rate": _safe_rate(
                    total_top_step_first_successes, total_top_step_first_attempts
                ),
                "top_step_first_then_cross_share_of_attempts": _safe_rate(
                    total_top_step_first_successes, total_obstacle_attempts
                ),
                "top_step_first_then_cross_share_of_successes": _safe_rate(
                    total_top_step_first_successes, total_obstacle_successes
                ),
                "top_step_first_attempt_share": _safe_rate(
                    total_top_step_first_attempts, total_obstacle_attempts
                ),
                "mean_obstacle_attempts_per_episode": _safe_mean(total_episode_obstacle_attempts, finished_episodes),
                "mean_successful_crossings_per_episode": _safe_mean(
                    total_episode_successful_crossings, finished_episodes
                ),
                "metric_config": {
                    "obstacle_height_threshold": obstacle_height_threshold,
                    "look_ahead_distance": look_ahead_distance,
                    "forward_command_threshold": forward_command_threshold,
                    "obstacle_clear_margin": obstacle_clear_margin,
                    "obstacle_clear_steps_required": obstacle_clear_steps_required,
                    "top_step_height_fraction": top_step_height_fraction,
                    "top_step_min_height": top_step_min_height,
                    "top_step_max_height_above_obstacle": 0.08,
                    "success_strategy_is_mutually_exclusive": True,
                    "strategy_definition": {
                        "direct_clear": "检测到障碍后成功越过，且第一次决定性事件既不是 shank 擦撞，也不是脚先踩上障碍顶面。",
                        "shank_hit_then_cross": "第一次决定性事件是 shank 擦撞，之后仍然成功越过。",
                        "top_step_first_then_cross": "第一次决定性事件是脚先踩上障碍顶面，之后仍然成功越过。",
                    },
                    "rate_definition": {
                        "overall_crossing_success_rate": "successful_crossings / obstacle_attempts",
                        "direct_clear_success_rate": "direct_clear_successes / obstacle_attempts",
                        "shank_hit_then_cross_rate": "shank_hit_then_cross_successes / shank_hit_first_attempts",
                        "top_step_first_then_cross_rate": "top_step_first_then_cross_successes / top_step_first_attempts",
                    },
                },
            }
        )

    for reward_name in sorted(reward_names):
        summary["logged_reward_terms_per_second"][reward_name] = _safe_mean(
            reward_term_episode_means[reward_name], finished_episodes
        )
    for termination_name in sorted(termination_names):
        summary["logged_termination_rates"][termination_name] = _safe_mean(
            termination_term_totals[termination_name], finished_episodes
        )

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