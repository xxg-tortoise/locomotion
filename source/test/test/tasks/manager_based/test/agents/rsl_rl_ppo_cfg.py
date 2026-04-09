# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticCfg, 
    RslRlPpoAlgorithmCfg,
    RslRlMLPModelCfg
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # Critic loss 权重
        use_clipped_value_loss=True, 
        clip_param=0.2, # PPO 裁剪范围，限制策略更新幅度
        entropy_coef=0.005, # 熵正则系数，鼓励探索
        num_learning_epochs=5, # 每批数据重复训练 5 遍
        num_mini_batches=4, # 每遍分 4 个 mini-batch
        learning_rate=1.0e-3, # 初始学习率
        schedule="adaptive", # 根据 KL 散度自动调整学习率
        gamma=0.99, # 折扣因子，关注长期回报
        lam=0.95, # GAE λ，平衡 bias/variance
        desired_kl=0.01, # 目标 KL 散度，超过则降低学习率
        max_grad_norm=1.0, # 梯度裁剪，防止梯度爆炸
    ) # 每次迭代的训练步数：5 epochs × 4 mini-batches = 20 次梯度更新
