from __future__ import annotations

import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from ..maps import SCALE_DIMS


def _group_by(
    results: Iterable[dict], key: str
) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for r in results:
        k = r[key]
        grouped.setdefault(k, []).append(r)
    return grouped


def _plot_bar_per_scale(
    results: List[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    filename_prefix: str,
    output_dir: str,
) -> None:
    """按 scale 生成柱状图（横轴为地形类型，柱为算法）。"""

    map_types = ["corridor", "rooms", "cave"]
    scales = ["small", "medium", "large"]

    algo_order = ["exact", "greedy", "improve"]
    algo_labels = {"exact": "精确", "greedy": "贪心", "improve": "改进"}

    for scale in scales:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # 找出该尺度下实际存在的算法
        algos_for_scale = set()
        for r in results:
            if r["scale"] == scale:
                algos_for_scale.add(r["algorithm"])
        algos = [a for a in algo_order if a in algos_for_scale]
        if not algos:
            plt.close(fig)
            continue

        x = np.arange(len(map_types))
        width = 0.25

        for i, algo in enumerate(algos):
            vals = []
            for mt in map_types:
                value = np.nan
                for r in results:
                    if (
                        r["scale"] == scale
                        and r["map_type"] == mt
                        and r["algorithm"] == algo
                    ):
                        value = r[metric_key]
                        break
                vals.append(value)
            offset = (i - (len(algos) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=algo_labels.get(algo, algo),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(["直线矿道", "房间+走廊", "洞穴"], rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}（{scale}）")

        # 图例放在下方，避免遮挡图形
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=len(algos), frameon=False)

        # 预留底部空间给图例
        fig.subplots_adjust(bottom=0.3, top=0.88)

        out_path = os.path.join(output_dir, f"{filename_prefix}_{scale}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def _plot_gap_small(results: List[dict], output_dir: str) -> None:
    """小规模图上的近似差距对比（柱状图）。"""

    small_results = [r for r in results if r["scale"] == "small"]
    map_types = ["corridor", "rooms", "cave"]
    algo_order = ["greedy", "improve"]
    algo_labels = {"greedy": "贪心", "improve": "改进"}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = np.arange(len(map_types))
    width = 0.3

    for i, algo in enumerate(algo_order):
        vals = []
        for mt in map_types:
            value = np.nan
            for r in small_results:
                if r["map_type"] == mt and r["algorithm"] == algo:
                    value = r["gap"] if r["gap"] is not None else np.nan
                    break
            vals.append(value)
        offset = (i - (len(algo_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=algo_labels.get(algo, algo),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["直线矿道", "房间+走廊", "洞穴"])
    ax.set_ylabel("相对最优差距 gap")
    ax.set_title("小规模地图：近似差距对比")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=len(algo_order), frameon=False)
    fig.subplots_adjust(bottom=0.3, top=0.88)

    out_path = os.path.join(output_dir, "gap_small.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_charts(results_dataclasses, output_dir: str) -> None:
    """从 ExperimentResult 列表生成所有图表。"""

    os.makedirs(output_dir, exist_ok=True)

    # dataclass -> dict
    results: List[dict] = [
        r if isinstance(r, dict) else r.__dict__ for r in results_dataclasses
    ]

    # 1) 火把数量对比
    _plot_bar_per_scale(
        results,
        metric_key="torch_count",
        ylabel="火把数量",
        title="各算法火把数量对比",
        filename_prefix="torch_count",
        output_dir=output_dir,
    )

    # 2) 运行时间对比
    _plot_bar_per_scale(
        results,
        metric_key="runtime_ms",
        ylabel="运行时间 (ms)",
        title="各算法运行时间对比",
        filename_prefix="runtime",
        output_dir=output_dir,
    )

    # 3) 小规模 map 上的最优差距
    _plot_gap_small(results, output_dir)
