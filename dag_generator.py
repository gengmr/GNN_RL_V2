# -*- coding: utf-8 -*-
"""
一个专为强化学习调度任务设计的高性能、高泛化性问题生成器。

该模块的核心是 `ProblemGenerator` 类，它能够创建包含随机有向无环图（DAG）
和异构处理器环境的复杂调度问题实例。其设计旨在为强化学习Agent提供
多样化且高度规范化的训练数据，模拟在有限资源下处理单项目或多项目的真实场景。

核心设计原则:
1.  无偏见的DAG生成:
    采用“随机逐边添加与周期检测”方法。该算法从一个空图开始，在一个完全
    随机打乱的候选边池中逐一尝试添加边。只有当一条边不会引入环路时，它才
    会被采纳。这种方法从根本上避免了任何结构性偏见，理论上能够生成任何
    可能的DAG拓扑结构。

2.  结构标准化的传递性约简:
    所有生成的图都经过严格的传递性约简处理。该步骤移除了图中的冗余依赖
    （例如，若存在 A->B->C 的路径，则直接的 A->C 边是冗余的），确保了
    输出的DAG是其所代表依赖关系的最简、最本质的表示，极大地有利于Agent的特征学习。

3.  高性能的纯NumPy实现:
    整个生成过程，包括图结构的创建、约简和成本分配，完全基于NumPy的
    向量化操作实现，避免了传统图库（如NetworkX）的开销，为大规模、
    高通量的训练数据生成提供了性能保障。

4.  高度参数化的泛化能力:
    问题的几乎所有维度，包括任务数、处理器数、图的依赖密度等，都可以
    在指定的范围内随机生成，使得Agent能够在训练中接触到从简单到复杂、
    从资源充裕到紧张的各种调度场景，从而提升其泛化能力。
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Set


@dataclass(frozen=True)
class SchedulingProblem:
    """
    一个不可变的、标准化的调度问题实例的数据容器。

    该结构封装了强化学习环境在一次`reset()`中所需的所有静态信息。所有数据均
    为NumPy数组，被设计为可直接输入机器学习模型，无需额外转换。

    Attributes:
        adj_matrix (np.ndarray):
            邻接矩阵, 形状为 (N, N), 类型为 int。
            `adj_matrix[i, j] = 1` 表示存在一条从任务i到任务j的直接依赖边。
            该矩阵是经过传递性约简的。
        comp_costs (np.ndarray):
            任务基础计算成本, 形状为 (N,), 类型为 int。
            `comp_costs[i]` 代表任务i的基础工作量。任务i在处理器p上的实际
            执行时间为: `comp_costs[i] / proc_speeds[p]`。
        comm_costs (np.ndarray):
            通信成本矩阵, 形状为 (N, N), 类型为 int。
            `comm_costs[i, j]` 代表当任务i和j被调度到**不同**处理器上时，
            在它们之间传输数据所需的时间成本。如果调度到同一处理器，此成本为0。
        num_tasks (int):
            任务（DAG节点）的总数 N。
        num_processors (int):
            处理器的总数 M。
        proc_speeds (np.ndarray):
            处理器相对速度, 形状为 (M,), 类型为 float。
            `proc_speeds[p]` 代表处理器p的计算能力系数。数值越大，处理速度越快。
    """
    adj_matrix: np.ndarray
    comp_costs: np.ndarray
    comm_costs: np.ndarray
    num_tasks: int
    num_processors: int
    proc_speeds: np.ndarray


class ProblemGenerator:
    """
    高效的、可参数化的调度问题实例生成器。
    """

    def __init__(self,
                 num_tasks_range: Tuple[int, int],
                 num_processors_range: Tuple[int, int],
                 density_range: Tuple[float, float],
                 comp_cost_range: Tuple[int, int],
                 comm_cost_range: Tuple[int, int],
                 proc_speed_range: Tuple[float, float]):
        """
        初始化问题生成器，配置生成问题的参数范围。

        Args:
            num_tasks_range (Tuple[int, int]):
                描述: [min, max]，任务数量的随机范围。
                作用: 控制调度问题的规模和复杂度。
                示例: `(20, 50)`

            num_processors_range (Tuple[int, int]):
                描述: [min, max]，处理器数量的随机范围。
                作用: 控制可用资源的数量，影响任务的并行执行能力。
                示例: `(4, 8)`

            density_range (Tuple[float, float]):
                描述: [min, max]，图的密度随机范围。
                作用: 控制任务间的依赖复杂度。密度定义为实际边数与
                      N*(N-1)/2 (N个节点的DAG中最大可能边数) 的比率。
                示例: `(0.2, 0.4)`

            comp_cost_range (Tuple[int, int]):
                描述: [min, max]，任务基础计算成本的随机范围。
                作用: 代表任务的工作量大小，影响节点“权重”。
                示例: `(20, 150)`

            comm_cost_range (Tuple[int, int]):
                描述: [min, max]，任务间通信成本的随机范围。
                作用: 代表数据传输的开销，影响边“权重”。
                示例: `(10, 80)`

            proc_speed_range (Tuple[float, float]):
                描述: [min, max]，处理器相对速度的随机范围。
                作用: 用于模拟异构计算环境。
                示例: `(1.0, 2.5)`
        """
        self.num_tasks_range = num_tasks_range
        self.num_processors_range = num_processors_range
        self.density_range = density_range
        self.comp_cost_range = comp_cost_range
        self.comm_cost_range = comm_cost_range
        self.proc_speed_range = proc_speed_range

    @staticmethod
    def _path_exists(adj: np.ndarray, start_node: int, end_node: int) -> bool:
        """
        使用广度优先搜索(BFS)高效检查图中是否存在从`start_node`到`end_node`的路径。

        Args:
            adj (np.ndarray): 图的邻接矩阵。
            start_node (int): 路径的起始节点。
            end_node (int): 路径的目标节点。

        Returns:
            bool: 如果路径存在，则返回True，否则返回False。
        """
        if start_node == end_node:
            return True

        num_tasks = adj.shape[0]
        queue: List[int] = [start_node]
        visited: Set[int] = {start_node}

        while queue:
            current_node = queue.pop(0)

            # 找到当前节点的所有邻居
            neighbors = np.where(adj[current_node, :] == 1)[0]
            for neighbor in neighbors:
                if neighbor == end_node:
                    return True  # 找到了目标路径
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    @staticmethod
    def _transitive_reduction(adj: np.ndarray) -> np.ndarray:
        """
        对邻接矩阵执行传递性约简 (基于NumPy的高效实现)。

        该算法的目的是移除图中的冗余“捷径”边。如果从节点i到节点j已经存在
        一条长度大于1的路径 (例如 i -> k -> j)，那么直接的边 i -> j 就是
        冗余的，因为它所代表的依赖关系已经由更长的路径蕴含。

        Args:
            adj (np.ndarray): 原始的、可能包含冗余边的邻接矩阵。

        Returns:
            np.ndarray: 经过传递性约简后的、结构更简洁的邻接矩阵。
        """
        num_tasks = adj.shape[0]
        path_matrix = (adj > 0)

        # 使用Floyd-Warshall算法思想计算传递闭包，比迭代矩阵乘法更直接。
        for k in range(num_tasks):
            for i in range(num_tasks):
                for j in range(num_tasks):
                    path_matrix[i, j] = path_matrix[i, j] or \
                                        (path_matrix[i, k] and path_matrix[k, j])

        # 如果 adj[i, j] 存在，并且还存在一条从 i 到 j 的更长路径，则其为冗余。
        # 一条更长的路径等价于 path_matrix[i, k] 和 adj[k, j] 都为真，
        # 对于某个中间节点k。这正是矩阵乘法 (path_matrix @ adj) 的定义。
        redundant_edges_mask = (path_matrix @ adj) > 0

        reduced_adj = adj.copy()
        reduced_adj[redundant_edges_mask] = 0

        return reduced_adj

    def generate(self, seed: int = None) -> SchedulingProblem:
        """
        生成一个全新的、随机的、无偏见的调度问题实例。

        该方法是与外部交互的主要接口。在强化学习训练循环中，环境的`reset()`
        方法通常会调用此函数来生成一个新的“关卡”。

        Args:
            seed (int, optional):
                随机种子。提供一个确定的种子可以确保生成可复现的问题实例，
                对于调试和算法验证非常有用。如果为`None`，则每次生成都是
                完全随机的。默认为 `None`。

        Returns:
            SchedulingProblem: 一个包含完整DAG和处理器信息的、立即可用的问题实例。
        """
        rng = np.random.default_rng(seed)

        # --- 步骤 1: 实例参数随机化 ---
        num_tasks = rng.integers(self.num_tasks_range[0], self.num_tasks_range[1], endpoint=True)
        num_processors = rng.integers(self.num_processors_range[0], self.num_processors_range[1], endpoint=True)
        density = rng.uniform(self.density_range[0], self.density_range[1])

        # --- 步骤 2: 无偏见DAG结构生成 ---
        # a. 初始化空图并确定目标边数。
        adj_matrix = np.zeros((num_tasks, num_tasks), dtype=int)
        num_edges_added = 0
        # 对于N个节点的DAG，最大边数为 N*(N-1)/2
        max_possible_edges = num_tasks * (num_tasks - 1) // 2
        target_num_edges = int(density * max_possible_edges)

        # b. 创建一个包含所有可能边 (N*(N-1)条) 的候选池并完全随机打乱。
        #    这是确保生成过程无任何结构性偏见的关键步骤。
        nodes = np.arange(num_tasks)
        candidate_edges = np.array(np.meshgrid(nodes, nodes)).T.reshape(-1, 2)
        candidate_edges = candidate_edges[candidate_edges[:, 0] != candidate_edges[:, 1]]
        rng.shuffle(candidate_edges)

        # c. 迭代添加边，同时通过周期检测确保图的无环性。
        for u, v in candidate_edges:
            if num_edges_added >= target_num_edges:
                break

            # 检查添加边 (u, v) 是否会形成环路。
            # 环路形成当且仅当图中已存在一条从 v 到 u 的路径。
            if not self._path_exists(adj_matrix, start_node=v, end_node=u):
                # 若不形成环路，则采纳该边。
                adj_matrix[u, v] = 1
                num_edges_added += 1

        # d. 标准化结构: 执行传递性约简，确保得到最简DAG表示。
        final_adj_matrix = self._transitive_reduction(adj_matrix)

        # --- 步骤 3: 成本参数分配 ---
        comp_costs = rng.integers(
            self.comp_cost_range[0], self.comp_cost_range[1], size=num_tasks, endpoint=True
        )

        comm_costs_matrix = rng.integers(
            self.comm_cost_range[0], self.comm_cost_range[1], size=(num_tasks, num_tasks), endpoint=True
        )
        comm_costs_matrix *= final_adj_matrix

        # --- 步骤 4: 异构处理器环境生成 ---
        proc_speeds = rng.uniform(
            self.proc_speed_range[0], self.proc_speed_range[1], size=num_processors
        )
        proc_speeds.sort()

        # --- 步骤 5: 封装并返回标准化的数据对象 ---
        problem_instance = SchedulingProblem(
            adj_matrix=final_adj_matrix,
            comp_costs=comp_costs,
            comm_costs=comm_costs_matrix,
            num_tasks=num_tasks,
            num_processors=num_processors,
            proc_speeds=proc_speeds
        )

        return problem_instance


if __name__ == '__main__':
    """
    演示与验证区域：

    本区域展示了如何使用 `ProblemGenerator`，并验证其生成的数据是否符合预期。
    您可以修改 `generator_config` 来测试不同参数下的生成效果。
    """

    # 1. 配置生成器的参数范围
    generator_config = {
        "num_tasks_range": (1, 20),  # 问题规模: 任务数量范围
        "num_processors_range": (1, 6),  # 资源规模: 处理器数量范围
        "density_range": (0, 1),  # 依赖关系: 图的密度
        "comp_cost_range": (10, 100),  # 任务工作量
        "comm_cost_range": (5, 50),  # 数据传输开销
        "proc_speed_range": (1.0, 2.5)  # 处理器异构性
    }

    # 2. 创建问题生成器实例
    problem_generator = ProblemGenerator(**generator_config)

    # 3. 生成一个调度问题实例
    print("--- 正在生成一个无偏见的、高度泛化的调度问题实例 ---")
    # 使用固定的随机种子(如 seed=42)可以确保每次运行都得到相同的结果，便于调试
    seed = random.randint(0, 1000)
    new_problem = problem_generator.generate(seed=seed)
    print("生成完毕！\n")

    # 4. 打印生成的数据，以验证其格式和内容
    print("--- 生成的问题实例详情 ---")
    print(f"随机选择的任务数量: {new_problem.num_tasks}")
    print(f"随机选择的处理器数量: {new_problem.num_processors}")
    print(f"处理器速度分布: {np.round(new_problem.proc_speeds, 2)}")
    # print(f"\n邻接矩阵 (shape: {new_problem.adj_matrix.shape}):")
    # 由于矩阵可能很大，只打印一个概览或部分
    # print("这是一个经过约简的稀疏矩阵，1表示存在直接依赖关系。")
    # print(new_problem.adj_matrix)
    print(f"图中实际边数: {np.sum(new_problem.adj_matrix)}")

    print(f"\n通信成本矩阵 (非零元素示例):")
    rows, cols = new_problem.comm_costs.nonzero()
    if len(rows) > 0:
        for r, c in list(zip(rows, cols))[:5]:  # 最多打印5个非零成本示例
            print(f"  - 任务 {r} -> {c} 的通信成本: {new_problem.comm_costs[r, c]}")
    else:
        print("  - 图中没有依赖关系（密度极低或任务数少）。")

    # 5. (可选) 使用 networkx 验证图的结构特性
    try:
        import networkx as nx

        g = nx.from_numpy_array(new_problem.adj_matrix, create_using=nx.DiGraph)
        is_dag = nx.is_directed_acyclic_graph(g)
        print(f"\n[验证] 生成的图是否为有向无环图 (DAG): {is_dag}")
        assert is_dag, "错误：生成的图包含环路！"

        num_components = nx.number_weakly_connected_components(g)
        print(f"[验证] 该DAG包含 {num_components} 个独立的组件（项目）。")
        if num_components > 1:
            print("      -> 成功生成了包含多个独立项目的泛化场景！")
        else:
            print("      -> 本次随机生成了一个单一的、连通的项目。")
    except ImportError:
        print("\n[提示] networkx未安装，无法进行图结构特性的自动验证。")