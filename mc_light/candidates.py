from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .grid import VoxelGrid

# 方向命名与坐标偏移（不允许朝下放置火把）
FACE_OFFSETS = {
    "UP": (0, 0, 1),
    "N": (0, 1, 0),   # 正 y
    "S": (0, -1, 0),  # 负 y
    "E": (1, 0, 0),   # 正 x
    "W": (-1, 0, 0),  # 负 x
}


@dataclass
class TorchCandidate:
    """火把候选点。

    这里将“火把所在的空气格”视为一个候选点，但仍记录一个附着的 Solid 方块
    及其朝向，用于保持与真实机制的一致性。
    """

    idx: int
    source: Tuple[int, int, int]          # 火把实际占用的空气格坐标
    attached_block: Optional[Tuple[int, int, int]] = None
    face: Optional[str] = None            # "UP" / "N" / "S" / "E" / "W"


def generate_candidates(grid: VoxelGrid) -> List[TorchCandidate]:
    """在给定网格上生成所有合法火把候选。

    规则：
    - 火把实际放置在某个空气格 (x, y, z)；
    - 该空气格必须与至少一个 Solid 方块相邻，且不是下方相邻（不允许悬挂在方块底面）；
    - 对每个符合条件的空气格，仅生成一个候选（避免同一格出现多个候选）。
    """

    M, N, H = grid.shape
    candidates: List[TorchCandidate] = []

    # 预先构造所有邻接方向（包括下方），用于判断是否有可附着的实心方块
    neighbor_dirs = [
        ("DOWN", (0, 0, -1)),
        ("UP", (0, 0, 1)),
        ("N", (0, 1, 0)),
        ("S", (0, -1, 0)),
        ("E", (1, 0, 0)),
        ("W", (-1, 0, 0)),
    ]

    idx_counter = 0
    for x in range(M):
        for y in range(N):
            for z in range(H):
                if grid.is_solid(x, y, z):
                    continue

                # 对于每个空气格，寻找至少一个可附着的 Solid 方块（非下方）
                attached_block = None
                attached_face: Optional[str] = None
                for face_name, (dx, dy, dz) in neighbor_dirs:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if not grid.in_bounds(nx, ny, nz):
                        continue
                    if not grid.is_solid(nx, ny, nz):
                        continue
                    if face_name == "DOWN":
                        # 不允许挂在方块底面
                        continue

                    attached_block = (nx, ny, nz)
                    attached_face = face_name
                    break

                if attached_block is None:
                    # 不存在合法附着面，则该空气格不能放火把
                    continue

                candidates.append(
                    TorchCandidate(
                        idx=idx_counter,
                        source=(x, y, z),
                        attached_block=attached_block,
                        face=attached_face,
                    )
                )
                idx_counter += 1

    return candidates
