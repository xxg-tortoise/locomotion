from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
    from .mesh_base_cfg import MeshTianziTerrainCfg

def _resolve_int(value_range: tuple[int, int], difficulty: float) -> int:
    lower, upper = value_range
    if lower < 0 or upper < 0 or lower > upper:
        raise ValueError(f"Invalid integer range: {value_range}.")
    value = lower + difficulty * (upper - lower)
    return int(np.floor(value + 0.5))


def _resolve_float(value_range: tuple[float, float], difficulty: float) -> float:
    lower, upper = value_range
    if lower > upper:
        raise ValueError(f"Invalid float range: {value_range}.")
    return lower + difficulty * (upper - lower)


def _compute_axis_layout(
    total_length: float, beam_count: int, beam_width: float, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    usable_length = total_length - 2.0 * margin
    if usable_length <= 0.0:
        raise ValueError(
            f"Usable length must be positive, got total_length={total_length}, margin={margin}."
        )

    if beam_count == 0:
        return np.zeros((0,), dtype=float), np.asarray([margin + 0.5 * usable_length], dtype=float)

    free_length = usable_length - beam_count * beam_width
    if free_length <= 0.0:
        raise ValueError(
            "Beam width is too large for the requested beam count. "
            f"total_length={total_length}, beam_count={beam_count}, beam_width={beam_width}, margin={margin}."
        )

    gap = free_length / (beam_count + 1)

    beam_centers = margin + gap + 0.5 * beam_width + np.arange(beam_count) * (gap + beam_width)
    free_cell_centers = margin + 0.5 * gap + np.arange(beam_count + 1) * (gap + beam_width)

    return beam_centers, free_cell_centers


def tianzi_grid_terrain(difficulty: float, cfg: "MeshTianziTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a Tian-zi style mesh terrain.

    Behavior:
    - `horizontal_beam_count_range` controls how many full-width horizontal beams are created.
    - `vertical_beam_count_range` controls how many full-height vertical beams are created.
    - `beam_height_range` controls beam height.
    - `beam_width_range` controls beam width.
    - `border_width` controls the flat inset inside each sub-terrain so adjacent env tiles do not touch.
    - When both count ranges are `(1, 1)` and `add_outer_frame=True`, the result is the classic `田` shape.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    num_horizontal_beams = _resolve_int(cfg.horizontal_beam_count_range, difficulty)
    num_vertical_beams = _resolve_int(cfg.vertical_beam_count_range, difficulty)
    beam_height = _resolve_float(cfg.beam_height_range, difficulty)
    beam_width = _resolve_float(cfg.beam_width_range, difficulty)

    if beam_height <= 0.0:
        raise ValueError(f"Beam height must be positive. Got: {beam_height}.")
    if beam_width <= 0.0:
        raise ValueError(f"Beam width must be positive. Got: {beam_width}.")
    if cfg.border_width < 0.0:
        raise ValueError(f"Border width must be non-negative. Got: {cfg.border_width}.")
    if cfg.ground_height <= 0.0:
        raise ValueError(f"Ground height must be positive. Got: {cfg.ground_height}.")

    frame_width = beam_width if cfg.frame_width is None else cfg.frame_width
    if frame_width < 0.0:
        raise ValueError(f"Frame width must be non-negative. Got: {frame_width}.")

    active_size = (cfg.size[0] - 2.0 * cfg.border_width, cfg.size[1] - 2.0 * cfg.border_width)
    if active_size[0] <= 0.0 or active_size[1] <= 0.0:
        raise ValueError(
            f"Border width {cfg.border_width} is too large for terrain size {cfg.size}."
        )

    active_origin_x = cfg.border_width
    active_origin_y = cfg.border_width

    x_margin = frame_width if cfg.add_outer_frame else 0.0
    y_margin = frame_width if cfg.add_outer_frame else 0.0

    vertical_centers_x, free_cell_centers_x = _compute_axis_layout(
        active_size[0], num_vertical_beams, beam_width, x_margin
    )
    horizontal_centers_y, free_cell_centers_y = _compute_axis_layout(
        active_size[1], num_horizontal_beams, beam_width, y_margin
    )

    meshes_list: list[trimesh.Trimesh] = []

    ground_dims = (cfg.size[0], cfg.size[1], cfg.ground_height)
    ground_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.ground_height)
    ground = trimesh.creation.box(ground_dims, trimesh.transformations.translation_matrix(ground_pos))
    meshes_list.append(ground)

    beam_span_x = active_size[0]
    beam_span_y = active_size[1]

    if cfg.add_outer_frame and frame_width > 0.0:
        inner_size = (active_size[0] - 2.0 * frame_width, active_size[1] - 2.0 * frame_width)
        if inner_size[0] <= 0.0 or inner_size[1] <= 0.0:
            raise ValueError(
                f"Frame width {frame_width} is too large for active terrain size {active_size}."
            )
        frame_center = (
            active_origin_x + 0.5 * active_size[0],
            active_origin_y + 0.5 * active_size[1],
            0.5 * beam_height,
        )
        meshes_list += make_border(active_size, inner_size, beam_height, frame_center)
        beam_span_x = inner_size[0]
        beam_span_y = inner_size[1]

    vertical_dims = (beam_width, beam_span_y, beam_height)
    for center_x in vertical_centers_x:
        beam_pos = (center_x + active_origin_x, active_origin_y + 0.5 * active_size[1], 0.5 * beam_height)
        beam = trimesh.creation.box(vertical_dims, trimesh.transformations.translation_matrix(beam_pos))
        meshes_list.append(beam)

    horizontal_dims = (beam_span_x, beam_width, beam_height)
    for center_y in horizontal_centers_y:
        beam_pos = (active_origin_x + 0.5 * active_size[0], center_y + active_origin_y, 0.5 * beam_height)
        beam = trimesh.creation.box(horizontal_dims, trimesh.transformations.translation_matrix(beam_pos))
        meshes_list.append(beam)

    if cfg.origin_on_free_cell:
        origin_x = active_origin_x + free_cell_centers_x[np.argmin(np.abs(free_cell_centers_x - 0.5 * active_size[0]))]
        origin_y = active_origin_y + free_cell_centers_y[np.argmin(np.abs(free_cell_centers_y - 0.5 * active_size[1]))]
    else:
        origin_x = 0.5 * cfg.size[0]
        origin_y = 0.5 * cfg.size[1]

    origin = np.asarray([origin_x, origin_y, 0.0], dtype=float)
    return meshes_list, origin

