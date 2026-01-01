from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .grid import VoxelGrid


@dataclass
class Universe:
    """需覆盖点集 U。

    attrs
    ------
    name: 名称，例如 "all_air" / "ground2"。
    coords: List[(x, y, z)]，按索引顺序存储 Universe 中所有格点坐标。
    index_map: np.ndarray[int32]，shape 与 grid 相同，若格点在 U 中则为其索引，否则为 -1。
    """

    name: str
    coords: List[Tuple[int, int, int]]
    index_map: np.ndarray

    @property
    def size(self) -> int:
        return len(self.coords)


def _init_index_map(grid: VoxelGrid) -> np.ndarray:
    """构造默认的 index_map = -1 数组。"""

    return np.full(grid.shape, -1, dtype=np.int32)


def build_universe_all_air(grid: VoxelGrid) -> Universe:
    """Universe 模式 1：所有空气格都作为 U。

    这是最简单、也是默认的安全约束：所有 Air 都需满足 block light >= 1。
    """

    index_map = _init_index_map(grid)
    coords: List[Tuple[int, int, int]] = []

    idx = 0
    M, N, H = grid.shape
    for x in range(M):
        for y in range(N):
            for z in range(H):
                if grid.is_air(x, y, z):
                    index_map[x, y, z] = idx
                    coords.append((x, y, z))
                    idx += 1

    return Universe(name="all_air", coords=coords, index_map=index_map)


def build_universe_ground2(grid: VoxelGrid) -> Universe:
    """Universe 模式 2：地面上方两格空气空间。

    更贴近“刷怪空间”的建模：
    - 选择所有满足：下方为 Solid，当前和上方连续两格均为 Air 的位置；
    - 将“脚下”和“头顶”这两个空气格都纳入 U，要求二者都被照亮。
    """

    index_map = _init_index_map(grid)
    coords: List[Tuple[int, int, int]] = []

    idx = 0
    M, N, H = grid.shape

    # 只可能在 1 <= z <= H-2 处作为“脚下”格
    for x in range(M):
        for y in range(N):
            for z in range(1, H - 1):
                # 下方为 Solid
                if not grid.is_solid(x, y, z - 1):
                    continue
                # 当前和上方两格为 Air
                if not (grid.is_air(x, y, z) and grid.is_air(x, y, z + 1)):
                    continue

                # 将脚下和头顶两个空气格都加入 Universe
                for zz in (z, z + 1):
                    if index_map[x, y, zz] >= 0:
                        # 已经加入过（可能多个脚位重合），跳过
                        continue
                    index_map[x, y, zz] = idx
                    coords.append((x, y, zz))
                    idx += 1

    return Universe(name="ground2", coords=coords, index_map=index_map)
