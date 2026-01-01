from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .grid import VoxelGrid


# 三种尺度的统一尺寸设置
SCALE_DIMS: Dict[str, Tuple[int, int, int]] = {
    "small": (20, 20, 3),
    "medium": (50, 50, 5),
    "large": (100, 100, 7),
}


@dataclass
class MapInfo:
    """地图元信息。"""

    map_type: str  # "corridor" / "rooms" / "cave"
    scale: str     # "small" / "medium" / "large"
    seed: int


def _base_grid(scale: str) -> VoxelGrid:
    """生成给定尺度下的全 Solid 网格。"""

    if scale not in SCALE_DIMS:
        raise ValueError(f"未知 scale: {scale}")
    M, N, H = SCALE_DIMS[scale]
    return VoxelGrid.full_solid(M, N, H)


def generate_corridor_map(scale: str, seed: int) -> Tuple[VoxelGrid, MapInfo]:
    """直线矿道：1×k 或 2×k 走廊。

    - 在整体 Solid 网格中挖出一条高 2 格的走廊；
    - 对 small 使用宽度 1，对 medium/large 使用宽度 2。"""

    rng = np.random.default_rng(seed)
    grid = _base_grid(scale)
    M, N, H = grid.shape

    # 走廊高度：保留底层为地面，从 z=1 到 z=min(2, H-1) 挖空
    z_start = 1
    z_end = min(2, H - 1)

    # small: 宽度 1；中大图：宽度 2
    if scale == "small":
        width = 1
    else:
        width = 2

    # 走廊沿 x 方向，从 (0, y0) 到 (M-1, y0 .. y0+width-1)
    y0 = N // 2 - width // 2
    y0 = max(0, min(y0, N - width))

    for x in range(M):
        for dy in range(width):
            y = y0 + dy
            for z in range(z_start, z_end + 1):
                grid.solid[x, y, z] = False

    return grid, MapInfo(map_type="corridor", scale=scale, seed=seed)


def generate_rooms_map(scale: str, seed: int) -> Tuple[VoxelGrid, MapInfo]:
    """房间 + 走廊场景。

    - 随机生成若干矩形房间；
    - 用 1 格宽的走廊连接房间中心；
    - 房间和走廊都是高 2 格的可通行空间。
    """

    rng = np.random.default_rng(seed)
    grid = _base_grid(scale)
    M, N, H = grid.shape

    z_start = 1
    z_end = min(2, H - 1)

    if scale == "small":
        num_rooms = 3
        room_min, room_max = 5, 8
    elif scale == "medium":
        num_rooms = 5
        room_min, room_max = 8, 12
    else:
        num_rooms = 8
        room_min, room_max = 10, 16

    centers = []

    for _ in range(num_rooms):
        w = int(rng.integers(room_min, room_max + 1))
        h = int(rng.integers(room_min, room_max + 1))

        x0 = int(rng.integers(1, max(2, M - w - 1)))
        y0 = int(rng.integers(1, max(2, N - h - 1)))
        x1 = min(M - 2, x0 + w)
        y1 = min(N - 2, y0 + h)

        # 挖房间内部
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z_start, z_end + 1):
                    grid.solid[x, y, z] = False

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        centers.append((cx, cy))

    # 用 L 型走廊连接房间中心
    if centers:
        cx0, cy0 = centers[0]
        for cx, cy in centers[1:]:
            # 先沿 x 再沿 y（或反之，随机）
            if rng.random() < 0.5:
                # x -> y
                xs = range(min(cx0, cx), max(cx0, cx) + 1)
                ys = range(min(cy0, cy), max(cy0, cy) + 1)
            else:
                ys = range(min(cy0, cy), max(cy0, cy) + 1)
                xs = range(min(cx0, cx), max(cx0, cx) + 1)

            for x in xs:
                for z in range(z_start, z_end + 1):
                    grid.solid[x, cy0, z] = False
            for y in ys:
                for z in range(z_start, z_end + 1):
                    grid.solid[cx, y, z] = False

            cx0, cy0 = cx, cy

    return grid, MapInfo(map_type="rooms", scale=scale, seed=seed)


def generate_cave_map(scale: str, seed: int) -> Tuple[VoxelGrid, MapInfo]:
    """迷宫/洞穴：随机障碍并保证整体连通。

    简化实现：
    - 先在地面层以上随机挖空形成噪声洞穴；
    - 然后从中心做一次 BFS，只保留与中心连通的空气区域，其余重新填回 Solid。
    """

    rng = np.random.default_rng(seed)
    grid = _base_grid(scale)
    M, N, H = grid.shape

    z_start = 1
    z_end = min(2, H - 1)

    # 不同尺度使用不同的开洞概率
    if scale == "small":
        p_open = 0.45
    elif scale == "medium":
        p_open = 0.40
    else:
        p_open = 0.38

    # 随机挖空形成洞穴
    noise = rng.random((M, N))
    for x in range(M):
        for y in range(N):
            if noise[x, y] < p_open:
                for z in range(z_start, z_end + 1):
                    grid.solid[x, y, z] = False

    # 选择一个靠近中心的空气格作为洞穴主连通分量的起点
    cx, cy = M // 2, N // 2
    start = None
    for radius in range(0, max(M, N)):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < M and 0 <= y < N and not grid.solid[x, y, z_start]:
                    start = (x, y, z_start)
                    break
            if start is not None:
                break
        if start is not None:
            break

    if start is None:
        # 极端情况下没有任何空气，直接返回
        return grid, MapInfo(map_type="cave", scale=scale, seed=seed)

    # BFS 保留主连通分量，其余空气重新填回 Solid
    visited = np.zeros(grid.shape, dtype=bool)
    stack = [start]
    visited[start] = True

    from collections import deque

    q: deque[Tuple[int, int, int]] = deque([start])
    while q:
        x, y, z = q.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            nz = z
            if not grid.in_bounds(nx, ny, nz):
                continue
            if visited[nx, ny, nz]:
                continue
            if grid.solid[nx, ny, nz]:
                continue
            visited[nx, ny, nz] = True
            q.append((nx, ny, nz))

    # 将不在主连通分量中的空气重新填回 Solid
    for x in range(M):
        for y in range(N):
            for z in range(z_start, z_end + 1):
                if not grid.solid[x, y, z] and not visited[x, y, z]:
                    grid.solid[x, y, z] = True

    return grid, MapInfo(map_type="cave", scale=scale, seed=seed)


def generate_map(map_type: str, scale: str, seed: int) -> Tuple[VoxelGrid, MapInfo]:
    """统一入口，根据 map_type 和 scale 生成对应地图。"""

    if map_type == "corridor":
        return generate_corridor_map(scale, seed)
    if map_type == "rooms":
        return generate_rooms_map(scale, seed)
    if map_type == "cave":
        return generate_cave_map(scale, seed)
    raise ValueError(f"未知 map_type: {map_type}")
