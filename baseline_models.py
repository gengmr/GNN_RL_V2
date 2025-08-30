# -*- coding: utf-8 -*-
"""
基准调度模型模块。

本模块实现了经典的、非学习的启发式调度算法，用于与强化学习Agent
的性能进行比较和评估。
"""
import numpy as np
import networkx as nx
from dag_generator import SchedulingProblem
from typing import Dict

# [新增] 导入环境和配置以支持状态化调度器
from environment import SchedulingEnvironment, TASK_STATUS_READY
import config as cfg


class HEFTScheduler:
    """
    实现了异构最早完成时间 (Heterogeneous Earliest Finish Time, HEFT) 调度算法。

    HEFT 算法是一种高效的列表调度启发式算法，广泛用于在异构计算环境中
    调度有向无环图（DAG）表示的应用程序。它主要包括两个阶段：

    1.  **任务优先级排序**:
        通过计算每个任务的“向上排名”(upward rank)来确定其优先级。
        向上排名递归地定义为一个任务的平均计算成本加上其所有直接后继任务中
        最高的向上排名（考虑通信成本）。排名最高的任务优先被调度。

    2.  **处理器选择**:
        按照任务的优先级顺序，将每个任务分配给能够使其获得“最早完成时间”
        (Earliest Finish Time, EFT) 的处理器。计算EFT时会考虑处理器的
        当前可用时间、前驱任务的完成时间以及必要的数据通信时间。
    """

    def schedule(self, problem: SchedulingProblem) -> float:
        """
        使用HEFT算法调度给定的DAG问题，并返回最终的完工时间 (Makespan)。

        Args:
            problem (SchedulingProblem): 从 `ProblemGenerator` 生成的调度问题实例。

        Returns:
            float: 由HEFT算法产生的调度方案的完工时间。
        """
        # --- 1. 初始化和数据准备 ---
        num_tasks = problem.num_tasks
        if num_tasks == 0:
            return 0.0

        num_processors = problem.num_processors
        adj = problem.adj_matrix
        comp_costs = problem.comp_costs
        comm_costs = problem.comm_costs
        proc_speeds = problem.proc_speeds

        # 使用 NetworkX 简化图操作，如查找前驱/后继节点
        g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

        # 预计算每个任务在所有处理器上的平均计算成本，用于计算向上排名
        avg_comp_costs = np.zeros(num_tasks)
        for i in range(num_tasks):
            avg_comp_costs[i] = np.mean(comp_costs[i] / (proc_speeds + 1e-8))

        # --- 2. 阶段一：任务优先级排序 (计算向上排名) ---
        ranks = self._compute_upward_ranks(g, num_tasks, avg_comp_costs, comm_costs)

        # 按排名的降序确定任务的调度顺序
        task_schedule_order = np.argsort(ranks)[::-1]

        # --- 3. 阶段二：处理器选择 (贪心分配) ---
        processor_available_times = np.zeros(num_processors)
        task_finish_times = np.zeros(num_tasks)
        task_assignments = np.full(num_tasks, -1, dtype=int)

        for task_id in task_schedule_order:
            min_eft = float('inf')
            best_proc = -1

            # 遍历所有处理器，为当前任务找到能提供最早完成时间(EFT)的处理器
            for proc_id in range(num_processors):
                # 计算任务在当前处理器上的最早开始时间(EST)
                est = self._calculate_est(task_id, proc_id, g, processor_available_times,
                                          task_finish_times, task_assignments, comm_costs)

                # 计算实际执行时间并得到EFT
                execution_time = comp_costs[task_id] / (proc_speeds[proc_id] + 1e-8)
                eft = est + execution_time

                # 如果当前处理器的EFT更优，则更新选择
                if eft < min_eft:
                    min_eft = eft
                    best_proc = proc_id

            # 将任务分配给最佳处理器，并更新状态
            task_finish_times[task_id] = min_eft
            task_assignments[task_id] = best_proc
            processor_available_times[best_proc] = min_eft

        # --- 4. 返回最终的 Makespan ---
        return np.max(task_finish_times) if num_tasks > 0 else 0.0

    def _compute_upward_ranks(self, g: nx.DiGraph, num_tasks: int,
                              avg_comp_costs: np.ndarray, comm_costs: np.ndarray) -> np.ndarray:
        """使用带备忘录的递归方法计算所有任务的向上排名。"""
        memo: Dict[int, float] = {}

        def get_rank(task_id: int) -> float:
            if task_id in memo:
                return memo[task_id]

            successors = list(g.successors(task_id))
            if not successors:
                max_succ_val = 0.0
            else:
                max_succ_val = max([
                    float(comm_costs[task_id, succ_id] + get_rank(succ_id)) for succ_id in successors
                ])

            rank = avg_comp_costs[task_id] + max_succ_val
            # [最终修复] 显式转换为 float 以匹配 memo 的类型提示
            memo[task_id] = float(rank)
            return float(rank)

        ranks = np.array([get_rank(i) for i in range(num_tasks)])
        return ranks

    def _calculate_est(self, task_id: int, proc_id: int, g: nx.DiGraph, proc_avail_times: np.ndarray,
                       task_finish_times: np.ndarray, task_assignments: np.ndarray,
                       comm_costs: np.ndarray) -> float:
        """计算一个任务在给定处理器上的最早开始时间 (EST)。"""
        processor_ready_time = float(proc_avail_times[proc_id])

        max_pred_finish_time = 0.0
        predecessors = list(g.predecessors(task_id))
        if not predecessors:
            return processor_ready_time

        for pred_id in predecessors:
            pred_proc = task_assignments[pred_id]
            communication_cost = 0.0
            if pred_proc != proc_id:
                communication_cost = comm_costs[pred_id, task_id]

            data_ready_time = task_finish_times[pred_id] + communication_cost
            max_pred_finish_time = max(max_pred_finish_time, data_ready_time)

        return max(processor_ready_time, max_pred_finish_time)


# [新增] 状态化HEFT调度器，用于MCTS引导和专家数据生成
class HEFTStatefulScheduler:
    """
    一个“状态化”的HEFT调度器，用于在MCTS的每一步提供启发式引导。

    与一次性计算整个调度的标准HEFTScheduler不同，该类在初始化时预计算
    任务优先级（向上排名），然后可以根据当前环境的实时状态（哪些任务已完成，
    哪些处理器何时可用）在任何给定步骤中推荐“下一个最佳”动作。
    """

    def __init__(self, problem: SchedulingProblem):
        """
        初始化状态化调度器。

        Args:
            problem (SchedulingProblem): 要解决的调度问题实例。
        """
        self.problem = problem
        self.num_tasks = problem.num_tasks
        self.num_processors = problem.num_processors
        self.g = nx.from_numpy_array(problem.adj_matrix, create_using=nx.DiGraph)
        self.ranks = self._compute_upward_ranks()

    def _compute_upward_ranks(self) -> np.ndarray:
        """计算所有任务的向上排名。与HEFTScheduler中的逻辑相同。"""
        if self.num_tasks == 0:
            return np.array([])

        avg_comp_costs = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            avg_comp_costs[i] = np.mean(self.problem.comp_costs[i] / (self.problem.proc_speeds + 1e-8))

        memo: Dict[int, float] = {}

        def get_rank(task_id: int) -> float:
            if task_id in memo:
                return memo[task_id]

            successors = list(self.g.successors(task_id))
            if not successors:
                max_succ_val = 0.0
            else:
                max_succ_val = max([
                    float(self.problem.comm_costs[task_id, succ_id] + get_rank(succ_id)) for succ_id in successors
                ])

            rank = avg_comp_costs[task_id] + max_succ_val
            # [最终修复] 显式转换为 float 以匹配 memo 的类型提示
            memo[task_id] = float(rank)
            return float(rank)

        return np.array([get_rank(i) for i in range(self.num_tasks)])

    def get_next_action(self, env: SchedulingEnvironment) -> int:
        """
        根据当前环境状态，计算并返回HEFT推荐的下一个动作。

        Args:
            env (SchedulingEnvironment): 当前的调度环境实例。

        Returns:
            int: 展平后的最佳动作索引 (task_id * M_MAX + proc_id)，如果没有合法动作则返回-1。
        """
        # 1. 识别当前所有处于“就绪”状态的任务
        ready_tasks = np.where(env.task_status[:self.num_tasks] == TASK_STATUS_READY)[0]

        if ready_tasks.size == 0:
            return -1

        # 2. 在所有就绪任务中，找到向上排名最高的任务
        highest_rank_task = ready_tasks[np.argmax(self.ranks[ready_tasks])]

        # 3. 为这个选定的任务找到能提供最早完成时间(EFT)的处理器
        min_eft = float('inf')
        best_proc = -1

        for proc_id in range(self.num_processors):
            processor_ready_time = float(env.processor_available_times[proc_id])
            predecessor_finish_time = 0.0
            predecessors = list(self.g.predecessors(highest_rank_task))

            if predecessors:
                for pred_id in predecessors:
                    pred_proc = env.task_assignments[pred_id]
                    comm_cost = 0.0
                    if pred_proc != proc_id:
                        comm_cost = self.problem.comm_costs[pred_id, highest_rank_task]

                    data_ready_time = env.task_finish_times[pred_id] + comm_cost
                    predecessor_finish_time = max(predecessor_finish_time, data_ready_time)

            est = max(processor_ready_time, predecessor_finish_time)
            execution_time = self.problem.comp_costs[highest_rank_task] / (self.problem.proc_speeds[proc_id] + 1e-8)
            eft = est + execution_time

            if eft < min_eft:
                min_eft = eft
                best_proc = proc_id

        if best_proc == -1:
            return -1

        # 4. 将 (task, processor) 对转换为扁平化的动作索引
        action = np.ravel_multi_index((highest_rank_task, best_proc), (cfg.N_MAX, cfg.M_MAX))
        return int(action)