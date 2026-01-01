from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Set


@dataclass
class ImproveResult:
    """改进版（贪心 + 后处理）结果。"""

    selected_indices: List[int]
    torch_count: int
    coverage_rate: float
    runtime_ms: float
    prune_iterations: int
    swap_attempts: int
    swaps_performed: int


def _compute_cover_mask(solution: List[int], candidate_masks: List[int]) -> int:
    mask = 0
    for idx in solution:
        mask |= candidate_masks[idx]
    return mask


def improve_solution(
    universe_size: int,
    candidate_masks: List[int],
    initial_solution: List[int],
    enable_one_swap: bool = True,
    max_swaps: int = 50,
) -> ImproveResult:
    """在贪心解基础上的改进算法：

    1. 冗余删除（post-pruning）：
       反复尝试移除任意一个火把，若剩余集合仍能完全覆盖 U，则删除之；
    2. 1-交换（可选）：
       对于每个无法直接删除的火把 t，尝试移除 t 后的未覆盖集合，若存在单个候选
       c 能补全这部分未覆盖，则用 c 替换 t（一次 1-交换）。
    """

    start = perf_counter()

    if universe_size == 0:
        runtime_ms = (perf_counter() - start) * 1000.0
        return ImproveResult(
            selected_indices=[],
            torch_count=0,
            coverage_rate=1.0,
            runtime_ms=runtime_ms,
            prune_iterations=0,
            swap_attempts=0,
            swaps_performed=0,
        )

    universe_mask = (1 << universe_size) - 1

    # 当前解及其覆盖
    current_solution: List[int] = list(dict.fromkeys(initial_solution))  # 去重并保持顺序
    current_cover = _compute_cover_mask(current_solution, candidate_masks)

    # ---------- 阶段 1：冗余删除 ----------
    prune_iterations = 0
    changed = True
    while changed:
        changed = False
        # 遍历时复制一份列表，避免在迭代中修改原列表造成问题
        for torch_idx in list(current_solution):
            if len(current_solution) <= 1:
                break
            # 尝试删除 torch_idx
            tentative = [t for t in current_solution if t != torch_idx]
            cover_without = _compute_cover_mask(tentative, candidate_masks)
            if cover_without.bit_count() == universe_size:
                current_solution = tentative
                current_cover = cover_without
                prune_iterations += 1
                changed = True
                break  # 重启循环

    # ---------- 阶段 2：1-交换（局部搜索） ----------
    swap_attempts = 0
    swaps_performed = 0

    if enable_one_swap and current_solution:
        all_indices: Set[int] = set(range(len(candidate_masks)))
        used: Set[int] = set(current_solution)

        # 限制最多进行 max_swaps 次成功的 1-交换
        for torch_idx in list(current_solution):
            if swaps_performed >= max_swaps:
                break

            # 移除 torch_idx 后的覆盖
            tentative = [t for t in current_solution if t != torch_idx]
            cover_without = _compute_cover_mask(tentative, candidate_masks)
            uncovered = universe_mask & ~cover_without

            if uncovered == 0:
                # 实际上该 torch 已在冗余删除阶段被删干净，此处略过
                continue

            # 搜索是否存在单个候选即可覆盖 uncovered
            candidate_pool = list(all_indices - set(tentative))
            found_replacement = None

            for cand in candidate_pool:
                swap_attempts += 1
                if (candidate_masks[cand] & uncovered) == uncovered:
                    found_replacement = cand
                    break

            if found_replacement is not None:
                # 完成一次 1-交换
                tentative.append(found_replacement)
                current_solution = tentative
                current_cover = _compute_cover_mask(current_solution, candidate_masks)
                swaps_performed += 1

    coverage_rate = current_cover.bit_count() / universe_size
    runtime_ms = (perf_counter() - start) * 1000.0

    return ImproveResult(
        selected_indices=current_solution,
        torch_count=len(current_solution),
        coverage_rate=coverage_rate,
        runtime_ms=runtime_ms,
        prune_iterations=prune_iterations,
        swap_attempts=swap_attempts,
        swaps_performed=swaps_performed,
    )
