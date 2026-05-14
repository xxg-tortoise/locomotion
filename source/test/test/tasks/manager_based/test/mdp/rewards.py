from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _compute_forward_obstacle_stats(
    terrain_sensor: RayCaster, look_ahead_distance: float, obstacle_threshold: float
) -> dict[str, torch.Tensor]:
    """从前向 height scan 中提取当前障碍的大致几何信息。"""

    ray_x = terrain_sensor.ray_starts[0, :, 0]
    ray_hits_z = terrain_sensor.data.ray_hits_w[..., 2]

    rear_mask = ray_x <= 0.0
    forward_mask = (ray_x > 0.0) & (ray_x <= look_ahead_distance)

    reference_ground_height = ray_hits_z[:, rear_mask].mean(dim=1, keepdim=True)
    forward_hits_z = ray_hits_z[:, forward_mask]
    obstacle_top_height = forward_hits_z.max(dim=1).values.unsqueeze(1)
    obstacle_height = torch.clamp(obstacle_top_height - reference_ground_height, min=0.0)
    obstacle_present = obstacle_height.squeeze(1) > obstacle_threshold

    forward_ray_x = ray_x[forward_mask].to(device=ray_hits_z.device)
    elevated_hits = (forward_hits_z - reference_ground_height) > obstacle_threshold
    inf_mask = torch.full((ray_hits_z.shape[0], forward_ray_x.numel()), float("inf"), device=ray_hits_z.device)
    neg_inf_mask = torch.full((ray_hits_z.shape[0], forward_ray_x.numel()), float("-inf"), device=ray_hits_z.device)

    obstacle_front_distance = torch.where(elevated_hits, forward_ray_x.unsqueeze(0), inf_mask).min(dim=1).values
    obstacle_back_distance = torch.where(elevated_hits, forward_ray_x.unsqueeze(0), neg_inf_mask).max(dim=1).values
    obstacle_front_distance = torch.where(
        torch.isfinite(obstacle_front_distance),
        obstacle_front_distance,
        torch.zeros_like(obstacle_front_distance),
    )
    obstacle_back_distance = torch.where(
        torch.isfinite(obstacle_back_distance),
        obstacle_back_distance,
        torch.zeros_like(obstacle_back_distance),
    )

    return {
        "reference_ground_height": reference_ground_height,
        "obstacle_top_height": obstacle_top_height,
        "obstacle_height": obstacle_height,
        "obstacle_present": obstacle_present,
        "obstacle_front_distance": obstacle_front_distance,
        "obstacle_back_distance": obstacle_back_distance,
    }


def _compute_obstacle_proximity_weight(
    obstacle_front_distance: torch.Tensor,
    obstacle_present: torch.Tensor,
    activation_far_distance: float,
    activation_near_distance: float,
) -> torch.Tensor:
    """根据障碍前缘距离生成一个平滑接近权重，避免奖励过早激活。"""

    if activation_near_distance >= activation_far_distance:
        raise ValueError(
            "activation_near_distance 必须小于 activation_far_distance，"
            f"但收到 {activation_near_distance} >= {activation_far_distance}。"
        )

    ramp = (activation_far_distance - obstacle_front_distance) / (activation_far_distance - activation_near_distance)
    ramp = torch.clamp(ramp, min=0.0, max=1.0)
    return ramp * obstacle_present.float()


def _body_positions_in_base_yaw_frame(asset, body_ids) -> torch.Tensor:
    """将指定刚体的位置转换到机身 yaw 对齐坐标系。"""

    body_pos_w = asset.data.body_pos_w[:, body_ids, :]
    body_pos_rel_w = body_pos_w - asset.data.root_pos_w.unsqueeze(1)
    body_count = body_pos_rel_w.shape[1]

    base_yaw_quat = yaw_quat(asset.data.root_quat_w)
    expanded_base_yaw_quat = base_yaw_quat.unsqueeze(1).expand(-1, body_count, -1).reshape(-1, 4)
    body_pos_rel_b = quat_apply_inverse(expanded_base_yaw_quat, body_pos_rel_w.reshape(-1, 3))
    return body_pos_rel_b.view(-1, body_count, 3)


def _compute_swing_landing_and_top_step_mask(
    obstacle_stats: dict[str, torch.Tensor],
    foot_pos_b: torch.Tensor,
    foot_height: torch.Tensor,
    first_contact: torch.Tensor,
    last_air_time: torch.Tensor,
    moving: torch.Tensor,
    air_time_threshold: float,
    top_step_height_fraction: float,
    top_step_min_height: float,
    top_step_max_height_above_obstacle: float,
    x_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算摆动落脚事件以及“脚先踩到障碍顶面”的掩码。"""

    swing_landing = first_contact & (last_air_time > air_time_threshold)
    top_step_height_threshold = obstacle_stats["reference_ground_height"] + torch.maximum(
        obstacle_stats["obstacle_height"] * top_step_height_fraction,
        torch.full_like(obstacle_stats["obstacle_height"], top_step_min_height),
    )
    foot_on_obstacle_span = (
        foot_pos_b[:, :, 0] >= (obstacle_stats["obstacle_front_distance"].unsqueeze(1) - x_margin)
    ) & (
        foot_pos_b[:, :, 0] <= (obstacle_stats["obstacle_back_distance"].unsqueeze(1) + x_margin)
    )
    foot_on_top = (
        swing_landing
        & moving
        & obstacle_stats["obstacle_present"].unsqueeze(1)
        & foot_on_obstacle_span
        & (foot_height >= top_step_height_threshold)
        & (foot_height <= obstacle_stats["obstacle_top_height"] + top_step_max_height_above_obstacle)
    )
    return swing_landing, foot_on_top


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
    terrain_sensor_cfg: SceneEntityCfg,
    clearance_margin: float, # 比障碍再高多少才算安全
    std: float,  # 奖励曲线的宽度。越小越严格，只有非常接近动态目标高度时才有明显奖励；越大越宽松。
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,  # 速度门槛。当平面速度指令的模长低于这个值时，这项奖励直接清零，防止静止时也被鼓励抬脚。
    obstacle_threshold: float = 0.02, # 小凸起要不要触发抬脚奖励
    look_ahead_distance: float = 0.6,  # 只看前方多远的障碍
    reward_activation_far_distance: float = 0.32,  # 障碍前缘还在这个距离之外时，不提前激活抬脚奖励
    reward_activation_near_distance: float = 0.18,  # 障碍靠近到这个距离内时，抬脚奖励完全激活
) -> torch.Tensor:
    """
        在摆动相奖励足端越过前方扫描到的最高障碍，并额外留出一点安全裕量
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    in_air = (contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0).float()
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    obstacle_stats = _compute_forward_obstacle_stats(
        terrain_sensor,
        look_ahead_distance=look_ahead_distance,
        obstacle_threshold=obstacle_threshold,
    )
    obstacle_top_height = obstacle_stats["obstacle_top_height"]
    target_height = obstacle_top_height + clearance_margin
    proximity_weight = _compute_obstacle_proximity_weight(
        obstacle_stats["obstacle_front_distance"],
        obstacle_stats["obstacle_present"],
        activation_far_distance=reward_activation_far_distance,
        activation_near_distance=reward_activation_near_distance,
    )

    clearance_error = torch.square(foot_height - target_height)
    reward = torch.sum(torch.exp(-clearance_error / std**2) * in_air, dim=1)
    moving = torch.norm(command[:, :2], dim=1) > command_threshold
    reward *= moving.float() * proximity_weight
    return reward


def direct_clear_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    air_time_threshold: float = 0.05,
    obstacle_threshold: float = 0.02,
    look_ahead_distance: float = 0.6,
    reward_activation_far_distance: float = 0.32,
    reward_activation_near_distance: float = 0.18,
    landing_target_margin: float = 0.03,
    min_landing_margin: float = 0.01,
    landing_target_std: float = 0.05,
    top_step_height_fraction: float = 0.5,
    top_step_min_height: float = 0.02,
    top_step_max_height_above_obstacle: float = 0.08,
    x_margin: float = 0.02,
) -> torch.Tensor:
    """
        奖励摆动相后的第一次落脚直接落到障碍后方，而不是先踩在障碍顶面。
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    obstacle_stats = _compute_forward_obstacle_stats(
        terrain_sensor,
        look_ahead_distance=look_ahead_distance,
        obstacle_threshold=obstacle_threshold,
    )
    foot_pos_b = _body_positions_in_base_yaw_frame(asset, asset_cfg.body_ids)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    moving = torch.norm(command[:, :2], dim=1, keepdim=True) > command_threshold
    proximity_weight = _compute_obstacle_proximity_weight(
        obstacle_stats["obstacle_front_distance"],
        obstacle_stats["obstacle_present"],
        activation_far_distance=reward_activation_far_distance,
        activation_near_distance=reward_activation_near_distance,
    )
    swing_landing, foot_on_top = _compute_swing_landing_and_top_step_mask(
        obstacle_stats=obstacle_stats,
        foot_pos_b=foot_pos_b,
        foot_height=foot_height,
        first_contact=first_contact,
        last_air_time=last_air_time,
        moving=moving,
        air_time_threshold=air_time_threshold,
        top_step_height_fraction=top_step_height_fraction,
        top_step_min_height=top_step_min_height,
        top_step_max_height_above_obstacle=top_step_max_height_above_obstacle,
        x_margin=x_margin,
    )

    landing_target_x = obstacle_stats["obstacle_back_distance"].unsqueeze(1) + landing_target_margin
    landing_error = torch.square(foot_pos_b[:, :, 0] - landing_target_x)
    landing_quality = torch.exp(-landing_error / landing_target_std**2)
    landing_beyond_obstacle = foot_pos_b[:, :, 0] >= (
        obstacle_stats["obstacle_back_distance"].unsqueeze(1) + min_landing_margin
    )
    reward = (
        landing_quality
        * swing_landing.float()
        * (~foot_on_top).float()
        * landing_beyond_obstacle.float()
        * moving.float()
        * proximity_weight.unsqueeze(1)
    )
    return torch.max(reward, dim=1).values


def premature_foot_reach_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    obstacle_threshold: float = 0.02,
    look_ahead_distance: float = 0.6,
    reward_activation_far_distance: float = 0.32,
    reward_activation_near_distance: float = 0.18,
    base_allowed_forward_x: float = 0.14,
    max_allowed_forward_x: float = 0.30,
    forward_slack_gain: float = 1.0,
) -> torch.Tensor:
    """
        惩罚障碍还较远时摆动脚过早大幅前伸，避免腿总想提前去够门槛。
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    obstacle_stats = _compute_forward_obstacle_stats(
        terrain_sensor,
        look_ahead_distance=look_ahead_distance,
        obstacle_threshold=obstacle_threshold,
    )
    proximity_weight = _compute_obstacle_proximity_weight(
        obstacle_stats["obstacle_front_distance"],
        obstacle_stats["obstacle_present"],
        activation_far_distance=reward_activation_far_distance,
        activation_near_distance=reward_activation_near_distance,
    )
    early_reach_weight = obstacle_stats["obstacle_present"].float() * (1.0 - proximity_weight)
    foot_pos_b = _body_positions_in_base_yaw_frame(asset, asset_cfg.body_ids)
    in_air = (contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0).float()
    moving = (torch.norm(command[:, :2], dim=1, keepdim=True) > command_threshold).float()

    allowed_forward_reach = base_allowed_forward_x + forward_slack_gain * torch.clamp(
        reward_activation_far_distance - obstacle_stats["obstacle_front_distance"],
        min=0.0,
    )
    allowed_forward_reach = torch.clamp(allowed_forward_reach, max=max_allowed_forward_x)
    excess_reach = torch.clamp(foot_pos_b[:, :, 0] - allowed_forward_reach.unsqueeze(1), min=0.0)

    penalty = (
        excess_reach.square()
        * in_air
        * moving
        * early_reach_weight.unsqueeze(1)
    )
    return torch.max(penalty, dim=1).values


def top_step_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    air_time_threshold: float = 0.05,
    obstacle_threshold: float = 0.02,
    look_ahead_distance: float = 0.6,
    top_step_height_fraction: float = 0.5,
    top_step_min_height: float = 0.02,
    top_step_max_height_above_obstacle: float = 0.08,
    x_margin: float = 0.02,
) -> torch.Tensor:
    """
        惩罚摆动相后的第一次落脚踩在障碍顶面，压制“先踩一下再过去”的策略。
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    obstacle_stats = _compute_forward_obstacle_stats(
        terrain_sensor,
        look_ahead_distance=look_ahead_distance,
        obstacle_threshold=obstacle_threshold,
    )
    foot_pos_b = _body_positions_in_base_yaw_frame(asset, asset_cfg.body_ids)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    moving = torch.norm(command[:, :2], dim=1, keepdim=True) > command_threshold
    _, foot_on_top = _compute_swing_landing_and_top_step_mask(
        obstacle_stats=obstacle_stats,
        foot_pos_b=foot_pos_b,
        foot_height=foot_height,
        first_contact=first_contact,
        last_air_time=last_air_time,
        moving=moving,
        air_time_threshold=air_time_threshold,
        top_step_height_fraction=top_step_height_fraction,
        top_step_min_height=top_step_min_height,
        top_step_max_height_above_obstacle=top_step_max_height_above_obstacle,
        x_margin=x_margin,
    )
    return torch.any(foot_on_top, dim=1).float()


def top_surface_stance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    obstacle_threshold: float = 0.02,
    look_ahead_distance: float = 0.6,
    top_step_height_fraction: float = 0.5,
    top_step_min_height: float = 0.02,
    top_step_max_height_above_obstacle: float = 0.08,
    x_margin: float = 0.02,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """
        惩罚脚在障碍顶面上的驻留接触，降低“先踩顶再过”的整体收益。
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    obstacle_stats = _compute_forward_obstacle_stats(
        terrain_sensor,
        look_ahead_distance=look_ahead_distance,
        obstacle_threshold=obstacle_threshold,
    )
    foot_pos_b = _body_positions_in_base_yaw_frame(asset, asset_cfg.body_ids)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > contact_force_threshold
    moving = torch.norm(command[:, :2], dim=1, keepdim=True) > command_threshold

    top_step_height_threshold = obstacle_stats["reference_ground_height"] + torch.maximum(
        obstacle_stats["obstacle_height"] * top_step_height_fraction,
        torch.full_like(obstacle_stats["obstacle_height"], top_step_min_height),
    )
    foot_on_obstacle_span = (
        foot_pos_b[:, :, 0] >= (obstacle_stats["obstacle_front_distance"].unsqueeze(1) - x_margin)
    ) & (
        foot_pos_b[:, :, 0] <= (obstacle_stats["obstacle_back_distance"].unsqueeze(1) + x_margin)
    )
    foot_on_top_stance = (
        in_contact
        & moving
        & obstacle_stats["obstacle_present"].unsqueeze(1)
        & foot_on_obstacle_span
        & (foot_height >= top_step_height_threshold)
        & (foot_height <= obstacle_stats["obstacle_top_height"] + top_step_max_height_above_obstacle)
    )
    return torch.sum(foot_on_top_stance.float(), dim=1)


def stumble_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    horizontal_force_threshold: float = 1.0,
    horizontal_to_vertical_ratio: float = 4.0,
    air_time_threshold: float = 0.05,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """
        惩罚摆动相结束时脚先撞到近似竖直障碍的情况，避免前摆脚尖/脚背磕到门槛
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    horizontal_force = torch.norm(contact_forces[..., :2], dim=-1).max(dim=1)[0]
    vertical_force = torch.abs(contact_forces[..., 2]).max(dim=1)[0]

    # 摆动相结束后的第一次接触 & 这只脚前面确实离地过一小段时间
    swing_impact = first_contact & (last_air_time > air_time_threshold) 
    # 接触时水平冲击明显大于竖直接触力，像是在撞门槛/横梁，而不是正常踩地
    obstacle_hit = (horizontal_force > horizontal_force_threshold) & (
        horizontal_force > horizontal_to_vertical_ratio * vertical_force
    )
    # 当前速度指令不是接近静止
    moving = torch.norm(command[:, :2], dim=1, keepdim=True) > command_threshold
    penalty = swing_impact & obstacle_hit & moving
    return torch.sum(penalty.float(), dim=1)



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

