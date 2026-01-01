"""Minecraft torch placement 实验代码包。

子模块：
- grid: 三维网格与方块类型
- light: 光照 BFS 传播
- candidates: 火把候选生成
- universe: 需覆盖点集 U 的建模
- maps: 测试地图生成
- algorithms: 精确 / 贪心 / 改进算法
- eval: 统一评估与制图
"""

__all__ = [
    "grid",
    "light",
    "candidates",
    "universe",
    "maps",
    "algorithms",
]
