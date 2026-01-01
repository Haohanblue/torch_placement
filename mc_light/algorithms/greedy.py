from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List


@dataclass
class GreedyResult:
    """贪心集合覆盖结果。"""

    selected_indices: List[int]
    torch_count: int
    coverage_rate: float
    runtime_ms: float
    iterations: int


def greedy_set_cover(
    universe_size: int,
    candidate_masks: List[int],
) -> GreedyResult:
    """经典贪心 Set Cover：每次选取能覆盖最多未覆盖元素的候选。

    使用 Python int 作为位图表示集合，gain = (mask & uncovered).bit_count()。
    """

    start = perf_counter()

    universe_mask = (1 << universe_size) - 1
    uncovered = universe_mask

    selected: List[int] = []
    iterations = 0

    # 预先合并所有候选，检查整体可覆盖性
    global_cover = 0
    for m in candidate_masks:
        global_cover |= m

    if global_cover.bit_count() == 0 or universe_size == 0:
        runtime_ms = (perf_counter() - start) * 1000.0
        coverage_rate = 0.0 if universe_size > 0 else 1.0
        return GreedyResult(
            selected_indices=[],
            torch_count=0,
            coverage_rate=coverage_rate,
            runtime_ms=runtime_ms,
            iterations=0,
        )

    while uncovered:
        best_gain = 0
        best_idx = -1

        for idx, mask in enumerate(candidate_masks):
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_gain == 0 or best_idx < 0:
            # 已无法进一步覆盖未覆盖元素
            break

        selected.append(best_idx)
        uncovered &= ~candidate_masks[best_idx]
        iterations += 1

    cover_mask = universe_mask & ~uncovered
    coverage_rate = cover_mask.bit_count() / universe_size if universe_size > 0 else 1.0
    runtime_ms = (perf_counter() - start) * 1000.0

    return GreedyResult(
        selected_indices=selected,
        torch_count=len(selected),
        coverage_rate=coverage_rate,
        runtime_ms=runtime_ms,
        iterations=iterations,
    )
