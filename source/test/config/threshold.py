import isaaclab.terrains as terrain_utils
from isaaclab.terrains import TerrainGeneratorCfg

THRESHOLD_CFG = TerrainGeneratorCfg(
    size = (10.0, 10.0), # 每个子地形的尺寸
    border_width = 20.0, # 整体地形的边界宽度
    num_rows = 10, # 行数
    num_cols = 20, # 列数
    horizontal_scale = 0.1, # 地形沿x y的离散化尺度
    vertical_scale = 0.005, # 地形沿z的离散化尺度
    slope_threshold = 0.5, # 坡度阈值，高于该值则为垂直
    use_cache = False, # 是否使用缓存的地形数据
    sub_terrains = {
        "threshold": terrain_utils.MeshRailsTerrainCfg(
            proportion=1.0,
            rail_thickness_range=(0.05, 0.2),
            rail_height_range=(0.05, 0.2),
            platform_width=3.0,
            # flat_patch_sampling=1.0
        )
    }
)