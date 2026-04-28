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
        self.scene.terrain.max_init_terrain_level = 10
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = True

        # 关闭观测扰动和外力/推搡事件，保证 play 过程更稳定可复现
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AnymalDTestEnvCfg_EVAL(AnymalDTestEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5

        # 评估时关闭观测噪声和外部扰动，但保留随机地形与随机速度指令
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 评估时不再在线调整地形 curriculum，避免不同 checkpoint 的分布不一致
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False



# 如果要改 loss，我会优先动这几件事。先把 learning_rate 从 1e-3 降到 3e-4 到 5e-4，再把 entropy_coef 降到 0.001 到 0.002。
# 然后如果 value loss 还是高，就把 num_steps_per_env 从 24 提到 32 或 48，让 advantage 和 return 估计更稳一些。最后再去看是否要打开观测归一化，
# 或者把 event 型奖励先放缓一点，因为 stumble_penalty 和 contact 这类稀疏冲击项很容易把 critic 学崩。

# 第一，先把探索噪声降下来。你现在 entropy_coef=0.005，且新 run 的 Loss/entropy 和 Policy/mean_std 都显著高于旧 run，这说明策略到 1500 iter 还很“散”，
# 没有收紧。最先试的是把 entropy_coef 降到 0.001 到 0.002。第二，弱化过障碍 shaping 的激进程度。foot_clearance 可以先从 0.075 降到 0.04 到 0.06，
# stumble_penalty 从 -0.5 降到 -0.2 到 -0.3。第三，稍微加强稳态约束。action_rate_l2 可以从 -0.01 加到 -0.015 或 -0.02，
# ang_vel_xy_l2 可以从 -0.05 加到 -0.07 或 -0.1。第四，feet_air_time 这项值得单独注意，它现在长期是负的，说明 threshold=0.5 
# 对当前步态偏高，实际上它更像在持续施压而不是奖励；这项可以把 threshold 从 0.5 降到 0.35 到 0.4，或者把 weight 从 0.125 降到 0.08 到 0.1。