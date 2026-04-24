from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from .mesh_base import tianzi_grid_terrain


@configclass
class MeshTianziTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a Tian-zi style mesh terrain."""

    function = tianzi_grid_terrain

    # 水平梁数量范围
    horizontal_beam_count_range: tuple[int, int] = MISSING

    # 垂直梁数量范围
    vertical_beam_count_range: tuple[int, int] = MISSING

    # 梁高度范围
    beam_height_range: tuple[float, float] = MISSING

    # 梁宽度范围
    beam_width_range: tuple[float, float] = MISSING

    # 子地形内部留白宽度，用于在相邻 env 之间留出平坦间隔
    border_width: float = 0.0

    # 是否添加外框
    add_outer_frame: bool = True

    # 外框宽度，如果为 None 则使用 beam_width
    frame_width: float | None = None

    # 基础地面厚度
    ground_height: float = 1.0

    # 是否将地形原点放在最近的空格中心，而不是梁上
    origin_on_free_cell: bool = True


