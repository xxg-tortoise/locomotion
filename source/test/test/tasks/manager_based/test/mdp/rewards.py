from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        空中时间越长奖励越大，鼓励迈大步，原地站立时不给奖励
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
        鼓励脚落地时静止，避免滑步
        原理： 脚着地 * 脚在动 = 滑步，配置里用负权重
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str, 
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,  # 奖励曲线的宽度。越小越严格，只有非常接近 target_height 才有明显奖励；越大越宽松，偏高偏低一点也还能拿分。
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1, # 速度门槛。当平面速度指令的模长低于这个值时，这项奖励直接清零，防止静止时也被鼓励抬脚。
) -> torch.Tensor:
    """
        在摆动相奖励足端达到目标离地高度，避免只有空中时间而没有实际抬脚
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    in_air = (contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0).float()
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance_error = torch.square(foot_height - target_height)
    reward = torch.sum(torch.exp(-clearance_error / std**2) * in_air, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
        先将速度旋转到机身坐标系，再计算其与速度指令的指数核奖励
        std越小，代表对跟随精度要求高，但训练初期可能会导致机器人因为拿不到分而“摆烂”
        std越大，即使误差较大，也能拿到不错的奖励。这能帮助初期的机器人快速找到大致方向，但后期可能导致动作“软绵绵”，追踪精度不高
    """

    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
        在世界坐标系下追踪偏航角速度，适用于地形倾斜时的鲁棒追踪
    """

    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)



def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """站立不动时，让机器人保持标准姿势，避免乱晃"""

    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)









# 会的，如果你只在门槛地形上训练。 这就是所谓的 catastrophic forgetting（灾难性遗忘）。
# 你当前的 THRESHOLD_CFG，它学到的策略会过度拟合"每隔一段就抬腿"这个模式。放到平地上可能出现不必要的抬腿，或者在斜坡/台阶上表现很差。
# 你加的 foot clearance reward 会鼓励"总是抬高脚"，在平地上这就是浪费能量和不稳定的来源。
# 你的 foot clearance reward 也需要做条件化——只在真正需要跨障碍时给奖励，或者用一个较小的权重，让它成为"可选优势"而不是"强制行为"。实践中常见的做法是：奖励权重给小一些（如 0.05~0.1），让策略自己决定什么时候抬高脚。


# 当前配置的问题
# 你的奖励结构存在几个与"干净跨越"目标矛盾的地方：
# 1. lin_vel_z_l2 权重过高（-2.0）
# 这个惩罚强烈抑制身体的竖直方向运动。跨门槛时身体必须有一定的抬升，当前权重会让策略倾向于"低姿态滑过"而非"抬腿跨过"。
# 2. 缺少**足端离地高度（foot clearance）**奖励
# 当前的 feet_air_time（权重 0.125）只鼓励足部有空中时间，但不关心脚抬多高。机器人可以"拖着脚快速掠过"也能拿到这个奖励。
# 3. SHANK 接触未惩罚
# 在 velocity_env_cfg.py:263-266 中，undesired_contacts 只惩罚了 .*THIGH，小腿（SHANK）蹭过门槛不会被扣分。
# 4. 无 stumble 惩罚
# 没有检测"脚在swing phase时撞到障碍物"的惩罚，这正是"反射式尝试"和"在门槛上滑过"的来源。

# 建议增加的奖励/惩罚
# 是的，你需要增加特定奖励。 仅靠通用 locomotion 奖励很难涌现出"干净一步跨过"的行为。以下是具体建议：
# (1) Foot Clearance Reward（足端抬高奖励）
# 在 swing phase 中奖励足端达到足够的离地高度：
# (2) Stumble Penalty（绊脚惩罚）
# 在脚向前运动时检测突然的接触力（碰到门槛）：
# (3) 扩展 undesired_contacts 到 SHANK
# (4) 调低 lin_vel_z_l2
# 建议从 -2.0 降到 -0.5 ~ -1.0，给身体留出上下运动的空间。
# (5) 可选：Gait Phase Reward（步态相位奖励）
# 如果你要求对角步态、一步跨过，可以参考 Walk These Ways 的方法，用周期性时钟信号约束步态节奏，确保是协调的抬腿而非混乱的尝试。

# 增加 episode 长度	当前 20s 对于跨障碍场景足够，但确保速度指令让机器人有机会遇到多个门槛
# 增加前方距离感知	你的 height scanner 是 1.6×1.0 的网格，可以考虑加长前方扫描范围（如 2.0×1.0），让策略能提前"看到"门槛