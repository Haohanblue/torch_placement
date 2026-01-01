from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .grid import VoxelGrid

# 六邻接方向 (dx, dy, dz)
_NEIGHBORS_6: Tuple[Tuple[int, int, int], ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


@dataclass
class LightParams:
    """光照参数配置。"""

    L0: int = 14          # 火把源光照等级
    threshold: int = 1    # 安全阈值：block light >= threshold 视为安全


class LightSimulator:
    """在给定网格上进行方块光 BFS 传播的模拟器。

    该类会复用内部的 best_level 缓冲区，避免为每个火把候选重复分配大数组，
    从而在中大规模地图上保持可接受的性能。

    光照传播规则：
    - 起点为火把所在的空气方块；
    - 采用 6 邻接（Manhattan 距离），每走一步光照等级减 1；
    - 不透明方块 Solid 阻挡传播；
    - 记录每个格子到达的最大 light level；
    - 若某格子的 light level >= threshold，则视为被该火把覆盖（安全）。
    """

    def __init__(self, grid: VoxelGrid, params: LightParams | None = None) -> None:
        self.grid = grid
        self.params = params or LightParams()
        # 使用 int8 存储 best light level，-1 表示未访问
        self._best_level = np.full(grid.shape, -1, dtype=np.int8)
        self.bfs_calls: int = 0

    def reset_stats(self) -> None:
        self.bfs_calls = 0

    def coverage_mask_from_source(
        self,
        source: Tuple[int, int, int],
        universe_index: np.ndarray,
    ) -> int:
        """从给定光源（空气方块坐标）进行 BFS，返回覆盖 Universe 的位图掩码。

        参数
        ------
        source: (x, y, z)
            火把光源所在空气方块坐标。
        universe_index: np.ndarray[int32]
            与 grid.shape 相同，若该格子在 U 中则为其索引，否则为 -1。

        返回
        ------
        mask: int
            Python 整型表示的位图，bit i = 1 表示 Universe 中索引 i 的点被照亮。
        """

        x0, y0, z0 = source
        if not self.grid.in_bounds(x0, y0, z0):
            return 0
        if self.grid.is_solid(x0, y0, z0):
            # 光源必须位于空气格
            return 0

        self.bfs_calls += 1

        # 重置 best_level
        self._best_level.fill(-1)

        q: deque[Tuple[int, int, int]] = deque()
        self._best_level[x0, y0, z0] = self.params.L0
        q.append((x0, y0, z0))

        mask: int = 0

        while q:
            x, y, z = q.popleft()
            level = int(self._best_level[x, y, z])

            # 记录 Universe 覆盖
            uid = int(universe_index[x, y, z])
            if uid >= 0 and level >= self.params.threshold:
                mask |= 1 << uid

            if level <= 1:
                # 传播到此为止
                continue

            next_level = level - 1
            for dx, dy, dz in _NEIGHBORS_6:
                nx, ny, nz = x + dx, y + dy, z + dz
                if not self.grid.in_bounds(nx, ny, nz):
                    continue
                if self.grid.is_solid(nx, ny, nz):
                    # 不透明方块阻挡
                    continue
                if self._best_level[nx, ny, nz] >= next_level:
                    # 已有更高或相同光照，无需更新
                    continue
                self._best_level[nx, ny, nz] = next_level
                q.append((nx, ny, nz))

        return mask

    def batch_coverage_masks(
        self,
        sources: Iterable[Tuple[int, int, int]],
        universe_index: np.ndarray,
    ) -> List[int]:
        """对一批火把候选源点计算覆盖位图，返回列表。"""

        masks: List[int] = []
        for s in sources:
            m = self.coverage_mask_from_source(s, universe_index)
            masks.append(m)
        return masks
