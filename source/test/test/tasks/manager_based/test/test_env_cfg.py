from isaaclab.utils import configclass

from .velocity_env_cfg import VelEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


@configclass
class AnymalDTestEnvCfg(VelEnvCfg):
    def __post_init__(self):

        super().__post_init__()
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AnymalDTestEnvCfg_PLAY(AnymalDTestEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 8      # 把并行环境数量降到 50、环境间距设为 2.5，减少算力和显存压力
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 3
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = True

        # 关闭观测扰动和外力/推搡事件，保证 play 过程更稳定可复现
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
