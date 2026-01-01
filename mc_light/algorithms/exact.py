from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional


@dataclass
class ExactResult:
    """精确求解结果。"""

    selected_indices: List[int]
    torch_count: int
    coverage_rate: float
    runtime_ms: float
    nodes_expanded: int
    optimal_found: bool
    feasible: bool


class ExactSetCoverSolver:
    """基于回溯 + 分支限界的 0-1 集合覆盖精确求解器。

    仅用于小规模地图（例如 20×20×3），用于给贪心与改进算法提供最优基线。"""

    def __init__(
        self,
        universe_size: int,
        candidate_masks: List[int],
        time_limit_sec: float = 60.0,
        greedy_upper_bound: Optional[int] = None,
    ) -> None:
        self.universe_size = universe_size
        self.candidate_masks = candidate_masks
        self.time_limit_sec = time_limit_sec

        # Universe 的完整位图（所有 bit 置 1）
        self.universe_mask = (1 << universe_size) - 1

        # 全部候选联合覆盖
        global_cover = 0
        for m in candidate_masks:
            global_cover |= m
        self.global_cover = global_cover

        self.start_time: float = 0.0
        self.nodes_expanded: int = 0

        # 当前最优解
        if greedy_upper_bound is not None and greedy_upper_bound > 0:
            self.best_size: int = greedy_upper_bound
        else:
            self.best_size = len(candidate_masks) + 1
        self.best_solution: List[int] = []

        # 元素 -> 能覆盖它的候选索引 列表
        self.elem_to_cands: List[List[int]] = [[] for _ in range(universe_size)]
        self._build_elem_to_cands()

        # 预计算单个候选能覆盖的元素个数，用于下界估计
        self.max_cover_per_cand: int = max((m.bit_count() for m in candidate_masks), default=0)

    def _build_elem_to_cands(self) -> None:
        for cand_idx, mask in enumerate(self.candidate_masks):
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if bit_idx < self.universe_size:
                    self.elem_to_cands[bit_idx].append(cand_idx)
                m ^= lsb

    def _choose_uncovered_element(self, uncovered: int) -> int:
        """返回 uncovered 位图中最低位的元素索引。"""

        lsb = uncovered & -uncovered
        return lsb.bit_length() - 1

    def _search(
        self,
        uncovered: int,
        current_solution: List[int],
    ) -> None:
        # 时间剪枝
        if perf_counter() - self.start_time > self.time_limit_sec:
            return

        # 已覆盖全部 Universe
        if uncovered == 0:
            if len(current_solution) < self.best_size:
                self.best_size = len(current_solution)
                self.best_solution = list(current_solution)
            return

        # 简单的下界：至少还需要 ceil(剩余元素数 / 单个候选覆盖数上界) 个候选
        remaining = uncovered.bit_count()
        if self.max_cover_per_cand == 0:
            return
        lower_bound = (remaining + self.max_cover_per_cand - 1) // self.max_cover_per_cand
        if len(current_solution) + lower_bound > self.best_size:
            return

        # 选择一个尚未覆盖的元素 e，并在所有能覆盖 e 的候选上分支
        e = self._choose_uncovered_element(uncovered)
        cand_list = self.elem_to_cands[e]
        if not cand_list:
            # 无法覆盖该元素，当前分支不可能得到可行解
            return

        self.nodes_expanded += 1

        for cand_idx in cand_list:
            # 分支：选择 cand_idx
            current_solution.append(cand_idx)
            new_uncovered = uncovered & ~self.candidate_masks[cand_idx]
            self._search(new_uncovered, current_solution)
            current_solution.pop()

    def solve(self) -> ExactResult:
        # 若整体不可覆盖，直接返回不可行
        if self.global_cover.bit_count() < self.universe_size:
            return ExactResult(
                selected_indices=[],
                torch_count=0,
                coverage_rate=self.global_cover.bit_count() / max(1, self.universe_size),
                runtime_ms=0.0,
                nodes_expanded=0,
                optimal_found=False,
                feasible=False,
            )

        self.start_time = perf_counter()
        self.nodes_expanded = 0

        uncovered0 = self.universe_mask
        self._search(uncovered0, [])
        runtime_ms = (perf_counter() - self.start_time) * 1000.0

        if not self.best_solution:
            # 理论上 global_cover 已包含全部 Universe，此时不会发生
            coverage_rate = 0.0
            feasible = False
        else:
            cover_mask = 0
            for idx in self.best_solution:
                cover_mask |= self.candidate_masks[idx]
            coverage_rate = cover_mask.bit_count() / self.universe_size
            feasible = coverage_rate >= 1.0 - 1e-9

        optimal_found = feasible and self.best_size <= len(self.candidate_masks)

        return ExactResult(
            selected_indices=self.best_solution,
            torch_count=len(self.best_solution),
            coverage_rate=coverage_rate,
            runtime_ms=runtime_ms,
            nodes_expanded=self.nodes_expanded,
            optimal_found=optimal_found,
            feasible=feasible,
        )
