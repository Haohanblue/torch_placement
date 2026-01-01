from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# 方块类型：这里只区分 Solid / Air
BLOCK_AIR = 0
BLOCK_SOLID = 1


@dataclass
class VoxelGrid:
    """三维体素网格。

    使用 shape = (M, N, H)，分别对应 x, y, z 方向尺寸。
    solid[x, y, z] == True 表示该方块为实心方块(Solid)。
    """

    solid: np.ndarray  # bool 类型

    def __post_init__(self) -> None:
        if self.solid.ndim != 3:
            raise ValueError("solid 数组必须是三维的 (M, N, H)")
        if self.solid.dtype != np.bool_:
            # 统一转换为 bool，便于后续运算
            self.solid = self.solid.astype(np.bool_)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.solid.shape  # (M, N, H)

    @property
    def M(self) -> int:
        return self.solid.shape[0]

    @property
    def N(self) -> int:
        return self.solid.shape[1]

    @property
    def H(self) -> int:
        return self.solid.shape[2]

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.M and 0 <= y < self.N and 0 <= z < self.H

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return bool(self.solid[x, y, z])

    def is_air(self, x: int, y: int, z: int) -> bool:
        return not self.solid[x, y, z]

    @classmethod
    def full_solid(cls, M: int, N: int, H: int) -> "VoxelGrid":
        """构造一个全部为 Solid 的网格。"""

        solid = np.ones((M, N, H), dtype=bool)
        return cls(solid=solid)

    def clone(self) -> "VoxelGrid":
        return VoxelGrid(self.solid.copy())
