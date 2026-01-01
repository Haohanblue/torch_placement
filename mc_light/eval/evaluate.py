from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Dict, List

import numpy as np

from ..candidates import generate_candidates
from ..grid import VoxelGrid
from ..light import LightParams, LightSimulator
from ..maps import SCALE_DIMS, MapInfo, generate_map
from ..universe import Universe, build_universe_all_air
from ..algorithms.exact import ExactSetCoverSolver
from ..algorithms.greedy import GreedyResult, greedy_set_cover
from ..algorithms.improve import ImproveResult, improve_solution
from .charts import plot_all_charts


@dataclass
class ExperimentResult:
    """单次实验（地图 × Universe 模式 × 算法）的指标记录。"""

    map_type: str
    scale: str
    universe_mode: str
    algorithm: str  # "exact" / "greedy" / "improve"

    torch_count: int
    optimal_torch_count: int | None
    gap: float | None  # 相对最优差距 (|S|-|S*|)/|S*|

    coverage_rate: float
    runtime_ms: float

    greedy_iterations: int | None
    prune_iterations: int | None
    swap_attempts: int | None
    swaps_performed: int | None

    bfs_light_calls: int
    num_candidates: int
    universe_size: int


def _ensure_output_dirs(base_dir: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "charts"), exist_ok=True)


def _build_candidate_masks(
    grid: VoxelGrid,
    universe: Universe,
    light_params: LightParams,
) -> tuple[list[int], int, int]:
    """生成候选火把集合的覆盖位图列表。

    返回 (candidate_masks, bfs_calls, num_candidates)。
    """

    candidates = generate_candidates(grid)
    sources = [c.source for c in candidates]

    simulator = LightSimulator(grid, params=light_params)
    masks = simulator.batch_coverage_masks(sources, universe.index_map)

    # 过滤掉完全不覆盖 U 的候选
    filtered_masks: List[int] = []
    for m in masks:
        if m != 0:
            filtered_masks.append(m)

    return filtered_masks, simulator.bfs_calls, len(filtered_masks)


def run_all_experiments(output_dir: str = "output") -> List[ExperimentResult]:
    _ensure_output_dirs(output_dir)

    light_params = LightParams(L0=14, threshold=1)

    map_types = ["corridor", "rooms", "cave"]
    scales = ["small", "medium", "large"]

    results: List[ExperimentResult] = []

    # 固定随机种子，保证可复现
    base_seed = 42

    for scale_idx, scale in enumerate(scales):
        for map_idx, map_type in enumerate(map_types):
            seed = base_seed + scale_idx * 100 + map_idx

            grid, info = generate_map(map_type, scale, seed)

            # Universe：默认使用 all_air 模式
            universe = build_universe_all_air(grid)

            # 预计算候选覆盖
            candidate_masks, bfs_calls, num_candidates = _build_candidate_masks(
                grid, universe, light_params
            )

            universe_size = universe.size

            # 如果 U 为空，则所有算法都是 0 火把 + 100% 覆盖
            if universe_size == 0:
                for algo in ("exact", "greedy", "improve"):
                    results.append(
                        ExperimentResult(
                            map_type=map_type,
                            scale=scale,
                            universe_mode=universe.name,
                            algorithm=algo,
                            torch_count=0,
                            optimal_torch_count=0,
                            gap=0.0,
                            coverage_rate=1.0,
                            runtime_ms=0.0,
                            greedy_iterations=0,
                            prune_iterations=0,
                            swap_attempts=0,
                            swaps_performed=0,
                            bfs_light_calls=bfs_calls,
                            num_candidates=num_candidates,
                            universe_size=universe_size,
                        )
                    )
                continue

            # ---------- 贪心算法 ----------
            greedy_res: GreedyResult = greedy_set_cover(universe_size, candidate_masks)

            # ---------- 改进算法（冗余删除 + 1-交换） ----------
            # 为控制复杂度，大图只开启冗余删除，小图开启 1-交换
            enable_swap = universe_size <= 3000
            improve_res: ImproveResult = improve_solution(
                universe_size,
                candidate_masks,
                greedy_res.selected_indices,
                enable_one_swap=enable_swap,
                max_swaps=50,
            )

            # ---------- 小图上运行精确算法 ----------
            exact_torch_count: int | None = None
            exact_gap = None
            exact_coverage = None
            exact_runtime = None

            if scale == "small":
                solver = ExactSetCoverSolver(
                    universe_size=universe_size,
                    candidate_masks=candidate_masks,
                    time_limit_sec=60.0,
                    greedy_upper_bound=greedy_res.torch_count,
                )
                exact_res = solver.solve()
                if exact_res.feasible and exact_res.optimal_found:
                    exact_torch_count = exact_res.torch_count
                    exact_coverage = exact_res.coverage_rate
                    exact_runtime = exact_res.runtime_ms

                    results.append(
                        ExperimentResult(
                            map_type=map_type,
                            scale=scale,
                            universe_mode=universe.name,
                            algorithm="exact",
                            torch_count=exact_res.torch_count,
                            optimal_torch_count=exact_res.torch_count,
                            gap=0.0,
                            coverage_rate=exact_res.coverage_rate,
                            runtime_ms=exact_res.runtime_ms,
                            greedy_iterations=None,
                            prune_iterations=None,
                            swap_attempts=None,
                            swaps_performed=None,
                            bfs_light_calls=bfs_calls,
                            num_candidates=num_candidates,
                            universe_size=universe_size,
                        )
                    )

            # 对贪心和改进算法计算相对最优差距（仅在最优已知且可行时）
            def _gap(count: int) -> float | None:
                if exact_torch_count is None or exact_torch_count == 0:
                    return None
                return (count - exact_torch_count) / exact_torch_count

            # 记录贪心结果
            results.append(
                ExperimentResult(
                    map_type=map_type,
                    scale=scale,
                    universe_mode=universe.name,
                    algorithm="greedy",
                    torch_count=greedy_res.torch_count,
                    optimal_torch_count=exact_torch_count,
                    gap=_gap(greedy_res.torch_count),
                    coverage_rate=greedy_res.coverage_rate,
                    runtime_ms=greedy_res.runtime_ms,
                    greedy_iterations=greedy_res.iterations,
                    prune_iterations=None,
                    swap_attempts=None,
                    swaps_performed=None,
                    bfs_light_calls=bfs_calls,
                    num_candidates=num_candidates,
                    universe_size=universe_size,
                )
            )

            # 记录改进算法结果
            results.append(
                ExperimentResult(
                    map_type=map_type,
                    scale=scale,
                    universe_mode=universe.name,
                    algorithm="improve",
                    torch_count=improve_res.torch_count,
                    optimal_torch_count=exact_torch_count,
                    gap=_gap(improve_res.torch_count),
                    coverage_rate=improve_res.coverage_rate,
                    runtime_ms=improve_res.runtime_ms,
                    greedy_iterations=None,
                    prune_iterations=improve_res.prune_iterations,
                    swap_attempts=improve_res.swap_attempts,
                    swaps_performed=improve_res.swaps_performed,
                    bfs_light_calls=bfs_calls,
                    num_candidates=num_candidates,
                    universe_size=universe_size,
                )
            )

    # 写出 CSV 与 JSON 摘要
    csv_path = os.path.join(output_dir, "results_table.csv")
    json_path = os.path.join(output_dir, "summary.json")

    fieldnames = [
        "map_type",
        "scale",
        "universe_mode",
        "algorithm",
        "torch_count",
        "optimal_torch_count",
        "gap",
        "coverage_rate",
        "runtime_ms",
        "greedy_iterations",
        "prune_iterations",
        "swap_attempts",
        "swaps_performed",
        "bfs_light_calls",
        "num_candidates",
        "universe_size",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # 生成图表
    charts_dir = os.path.join(output_dir, "charts")
    plot_all_charts(results, charts_dir)

    return results


def main() -> None:
    run_all_experiments()


if __name__ == "__main__":
    main()
