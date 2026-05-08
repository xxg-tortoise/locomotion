from __future__ import annotations

import isaaclab.terrains as terrain_utils
from isaaclab.terrains import TerrainGeneratorCfg
from .mesh_base_cfg import MeshTianziTerrainCfg


# THRESHOLD_CFG = TerrainGeneratorCfg(
#     size = (10.0, 10.0), # 每个子地形的尺寸
#     border_width = 20.0, # 整体地形的边界宽度
#     num_rows = 10, # 行数
#     num_cols = 20, # 列数
#     horizontal_scale = 0.1, # 地形沿x y的离散化尺度
#     vertical_scale = 0.005, # 地形沿z的离散化尺度
#     slope_threshold = 0.5, # 坡度阈值，高于该值则为垂直
#     use_cache = False, # 是否使用缓存的地形数据
#     sub_terrains = {
#         "threshold": terrain_utils.MeshRailsTerrainCfg(
#             proportion=1.0,
#             rail_thickness_range=(0.2, 0.2),
#             rail_height_range=(0.05, 0.3),
#             platform_width=3.0,
#             # flat_patch_sampling=1.0
#         )
#     }
# )


# Terrain_custom_cfg = TerrainGeneratorCfg(
#     size=(10.0, 10.0),
#     border_width=20.0,  # 只作用于整张 terrain 外围
#     num_rows=10,
#     num_cols=20,
#     use_cache=False,
#     sub_terrains={
#         "tianzi": MeshTianziTerrainCfg(
#             proportion=1.0,
#             horizontal_beam_count_range=(1, 4),
#             vertical_beam_count_range=(1, 4),
#             beam_height_range=(0.02, 0.10),
#             beam_width_range=(0.02, 0.10),
#             border_width=1.0,
#             add_outer_frame=True,
#             frame_width=0.20,
#             ground_height=1.0,
#             origin_on_free_cell=True,
#         )
#     },
# )






def _build_multi_terrain_cfg(
    *,
    threshold_proportion: float,
    pyramid_stairs_proportion: float,
    pyramid_stairs_inv_proportion: float,
    boxes_proportion: float,
    random_rough_proportion: float,
    hf_pyramid_slope_proportion: float,
    hf_pyramid_slope_inv_proportion: float,
    difficulty_range: tuple[float, float],
) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        size=(10.0, 10.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=difficulty_range,
        use_cache=False,
        sub_terrains={
            # 旧门槛地形保留为主体，降低二次续训时遗忘门槛能力的风险。
            "threshold": MeshTianziTerrainCfg(
                proportion=threshold_proportion,
                horizontal_beam_count_range=(1, 4),
                vertical_beam_count_range=(1, 4),
                beam_height_range=(0.02, 0.10),
                beam_width_range=(0.02, 0.10),
                border_width=1.0,
                add_outer_frame=True,
                frame_width=0.20,
                ground_height=1.0,
                origin_on_free_cell=True,
            ),
            "pyramid_stairs": terrain_utils.MeshPyramidStairsTerrainCfg(
                proportion=pyramid_stairs_proportion,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_utils.MeshInvertedPyramidStairsTerrainCfg(
                proportion=pyramid_stairs_inv_proportion,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "boxes": terrain_utils.MeshRandomGridTerrainCfg(
                proportion=boxes_proportion,
                grid_width=0.45,
                grid_height_range=(0.05, 0.20),
                platform_width=2.0,
            ),
            "random_rough": terrain_utils.HfRandomUniformTerrainCfg(
                proportion=random_rough_proportion,
                noise_range=(0.02, 0.10),
                noise_step=0.02,
                border_width=0.25,
            ),
            "hf_pyramid_slope": terrain_utils.HfPyramidSlopedTerrainCfg(
                proportion=hf_pyramid_slope_proportion,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
            "hf_pyramid_slope_inv": terrain_utils.HfInvertedPyramidSlopedTerrainCfg(
                proportion=hf_pyramid_slope_inv_proportion,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
        },
    )


# Stage 1: 先以旧门槛为主，把新地形引入训练分布，但难度上限先压住。
Terrain_custom_stage1_cfg = _build_multi_terrain_cfg(
    threshold_proportion=0.65,
    pyramid_stairs_proportion=0.08,
    pyramid_stairs_inv_proportion=0.07,
    boxes_proportion=0.08,
    random_rough_proportion=0.06,
    hf_pyramid_slope_proportion=0.03,
    hf_pyramid_slope_inv_proportion=0.03,
    difficulty_range=(0.0, 0.55),
)

# Stage 2: 旧门槛与新地形接近平衡，开始让策略学习共享越障能力。
Terrain_custom_stage2_cfg = _build_multi_terrain_cfg(
    threshold_proportion=0.50,
    pyramid_stairs_proportion=0.10,
    pyramid_stairs_inv_proportion=0.09,
    boxes_proportion=0.12,
    random_rough_proportion=0.09,
    hf_pyramid_slope_proportion=0.05,
    hf_pyramid_slope_inv_proportion=0.05,
    difficulty_range=(0.0, 0.70),
)

# Stage 3: 泛化强化阶段，保留部分旧门槛避免遗忘，同时开放更高难度。
Terrain_custom_stage3_cfg = _build_multi_terrain_cfg(
    threshold_proportion=0.30,
    pyramid_stairs_proportion=0.14,
    pyramid_stairs_inv_proportion=0.13,
    boxes_proportion=0.16,
    random_rough_proportion=0.12,
    hf_pyramid_slope_proportion=0.075,
    hf_pyramid_slope_inv_proportion=0.075,
    difficulty_range=(0.0, 0.85),
)

# Stage 4: 收尾阶段，用全难度分布做最终收敛，但仍保留 20% 旧门槛作锚点。
Terrain_custom_stage4_cfg = _build_multi_terrain_cfg(
    threshold_proportion=0.20,
    pyramid_stairs_proportion=0.15,
    pyramid_stairs_inv_proportion=0.15,
    boxes_proportion=0.1,
    random_rough_proportion=0.15,
    hf_pyramid_slope_proportion=0.15,
    hf_pyramid_slope_inv_proportion=0.1,
    difficulty_range=(0.0, 1.0),
)


Terrain_custom_stage0_cfg = _build_multi_terrain_cfg(
    threshold_proportion=1.0,
    pyramid_stairs_proportion=0.0,
    pyramid_stairs_inv_proportion=0.0,
    boxes_proportion=0.0,
    random_rough_proportion=0.0,
    hf_pyramid_slope_proportion=0.0,
    hf_pyramid_slope_inv_proportion=0.0,
    difficulty_range=(0.0, 1.0),
)


# 默认先从第一阶段续训，稳定后再把这行切到 stage2 / stage3 / stage4。
Terrain_custom_cfg = Terrain_custom_stage0_cfg







