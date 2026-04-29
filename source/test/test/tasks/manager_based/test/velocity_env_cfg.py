# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .terrains.threshold import TIANZI_CFG



##
# Scene definition
##


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type = "generator",
        terrain_generator = TIANZI_CFG,
        # max_init_terrain_level = 5,
        collision_group = -1,
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode = "multiply", 
            restitution_combine_mode = "multiply", 
            static_friction = 1.0,
            dynamic_friction = 1.0,
        ),
        visual_material = sim_utils.MdlFileCfg(
            mdl_path = f"/home/robot/Projects/nvidia/isaacsim_assets/Assets/Isaac/5.1/Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw = True,
            texture_scale = (0.25, 0.25)
        ),
        debug_vis = False
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # robot
    robot: ArticulationCfg = MISSING

    # sensor
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    





##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True # 训练时对带噪声的观测项实际施加噪声
            self.concatenate_terms = True # 将组内的观测项拼接成一个大向量输出

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    '''
        startup 仿真启动时执行一次
    ''' 
    # 随机化所有刚体的摩擦系数
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
   
    # 在 base 上随机增减质量，模拟负载变化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # 随机偏移 base 的质心位置，模拟质量分布不均
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    '''
       reset 每次 episode 重置时执行
    '''
    # 随机化施加持续的恒定外力/外力矩（当前范围均为 0，相当于已关闭，保留接口）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # 随机化机器人基座的初始位姿和初始速度，防止策略只学会从固定起点恢复
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # 按比例随机化关节初始位置，初始速度为 0，防止策略依赖固定的初始姿态
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    '''
        interval 训练过程中周期性触发
    '''
    # 每 10~15 秒随机推一次机器人
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:

    # -- task
    track_lin_vel_xy_exp = RewTerm( 
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    ) # 鼓励机器人跟踪指令里的平面线速度
    track_ang_vel_z_exp = RewTerm( 
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    ) # 鼓励跟踪指令里的偏航角速度
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) # 惩罚竖直方向速度，避免上下
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # 惩罚 roll/pitch 角速度，避免机身晃动过大
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5) #惩罚关节力矩过大，降低能耗/暴力控制
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # 惩罚关节加速度过大，减少冲击
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # 惩罚动作变化过快，鼓励平滑控制
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.4,
            # "threshold": 0.5,
        },
    ) # 鼓励足部有一定的空中时间，避免一直贴地滑行
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        # weight=0.075,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "clearance_margin": 0.04,
            "std": 0.05,
            "obstacle_threshold": 0.02,
            "look_ahead_distance": 0.6,
        },
    ) # 在摆动相奖励足端越过前方检测到的障碍高度，减少拖脚和无意义高抬腿
    stumble_penalty = RewTerm(
        func=mdp.stumble_penalty,
        weight=-0.3,
        # weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "horizontal_force_threshold": 1.0,
            "horizontal_to_vertical_ratio": 4.0,
            "air_time_threshold": 0.05,
        },
    ) # 惩罚摆动相前摆脚撞到障碍，减少门槛前的绊脚和试探式乱蹭
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.75,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    ) # 惩罚不希望的接触，避免机器人与环境中的障碍物发生不必要的接触
    undesired_shank_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*SHANK"), "threshold": 1.0},
    ) # 额外惩罚小腿擦碰障碍，避免只抬脚尖不抬小腿
   
    # -- optional penalties
    # 惩罚机身姿态偏离“水平”的程度
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # 惩罚关节位置接近或超过关节极限
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)



@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    )

@configclass
class CurriculumCfg:
    # 动态调整地形难度，增加训练环境的多样性，提升策略的鲁棒性
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class VelEnvCfg(ManagerBasedRLEnvCfg):

    scene: TestSceneCfg = TestSceneCfg(num_envs=4096, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:

        self.decimation = 4 # 每4个物理步长输出一次动作
        self.episode_length_s = 20.0 

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15 # 提高 GPU 物理接触 patch 上限，避免并行环境多接触时溢出或性能问题
        
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

