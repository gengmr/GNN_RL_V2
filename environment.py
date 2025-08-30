# -*- coding: utf-8 -*-
"""
调度环境模块，遵循OpenAI Gym风格的接口设计。
...
"""
import numpy as np
# [修改] 导入 Optional 用于类型提示
from typing import Tuple, Dict, Any, Set, Optional

from dag_generator import SchedulingProblem
# ============================ [ 代码修改 1/4 - 已删除 ] ============================
# [原因] 解决循环导入问题。
# [方案] 此处的 import 语句已被移动到 reset 方法内部，在实际使用时才导入。
# from baseline_models import HEFTScheduler  <-- 此行已被删除
# ========================= [ 修改结束 ] =========================
import config as cfg

# 定义任务状态的常量，便于阅读和维护
TASK_STATUS_UNSCHEDULED = 0
TASK_STATUS_READY = 1
TASK_STATUS_RUNNING = 2
TASK_STATUS_DONE = 3


class SchedulingEnvironment:
    """
    一个用于DAG任务调度的、确定性的、单玩家游戏环境。
    """

    def __init__(self, n_max: int, m_max: int):
        """
        初始化环境。

        Args:
            n_max (int): 模型支持的最大任务数，用于状态张量的填充。
            m_max (int): 模型支持的最大处理器数，用于状态张量的填充。
        """
        self.n_max = n_max
        self.m_max = m_max
        # [修改] 将类型提示从 SchedulingProblem 修改为 Optional[SchedulingProblem]
        # 这明确表示 self.problem 属性在初始化时可以为 None。
        self.problem: Optional[SchedulingProblem] = None

        # 状态分离：将属性分为静态和动态两部分
        # 1. 静态属性 (Static Attributes): 在一次游戏中 (一个episode) 永不改变。
        self.num_tasks = 0
        self.num_processors = 0
        self.adj = None
        self.comm_costs = None
        self.comp_costs = None
        self.proc_speeds = None
        self.task_mask = None
        self.proc_mask = None
        self.predecessors: Dict[int, Set[int]] = {}
        self.successors: Dict[int, Set[int]] = {}
        # ============================ [ 代码修改 2/4 ] ============================
        # [原因] 根据“模块一”，引入一个在单局游戏内静态的、问题内蕴的尺度因子。
        # [方案] 新增 self.normalization_scale 属性，并初始化为安全的默认值1.0。
        self.normalization_scale = 1.0
        # ========================= [ 修改结束 ] =========================

        # 2. 动态属性 (Dynamic Attributes): 在每个 `step` 中都会改变。
        self.task_status = None
        self.task_finish_times = None
        self.task_assignments = None
        self.processor_available_times = None
        self.num_done_tasks = 0
        self.predecessor_counts = None

    def reset(self, problem: SchedulingProblem) -> Dict[str, np.ndarray]:
        """
        重置环境以开始一个新游戏。

        Args:
            problem (SchedulingProblem): 从ProblemGenerator生成的新问题实例。

        Returns:
            Dict[str, np.ndarray]: 初始状态的字典。
        """
        self.problem = problem
        self.num_tasks = problem.num_tasks
        self.num_processors = problem.num_processors

        # --- 初始化静态状态 (Static State) ---
        self.adj = np.zeros((self.n_max, self.n_max), dtype=np.float32)
        self.adj[:self.num_tasks, :self.num_tasks] = problem.adj_matrix

        self.comm_costs = np.zeros((self.n_max, self.n_max), dtype=np.float32)
        self.comm_costs[:self.num_tasks, :self.num_tasks] = problem.comm_costs

        self.comp_costs = np.zeros(self.n_max, dtype=np.float32)
        self.comp_costs[:self.num_tasks] = problem.comp_costs

        self.proc_speeds = np.zeros(self.m_max, dtype=np.float32)
        self.proc_speeds[:self.num_processors] = problem.proc_speeds

        self.task_mask = np.zeros(self.n_max, dtype=bool)
        self.task_mask[:self.num_tasks] = True
        self.proc_mask = np.zeros(self.m_max, dtype=bool)
        self.proc_mask[:self.num_processors] = True

        self.predecessors = {
            i: set(np.where(self.adj[:, i] == 1)[0]) for i in range(self.num_tasks)
        }
        self.successors = {
            i: set(np.where(self.adj[i, :] == 1)[0]) for i in range(self.num_tasks)
        }

        # ============================ [ 代码修改 3/4 ] ============================
        # [原因] 根据“模块一”，需要在每个新 episode 开始时计算并设置固定的归一化尺度。
        # [方案] 实例化HEFT调度器，计算当前问题的heft_makespan，并将其赋值给
        #        self.normalization_scale，同时处理heft_makespan为0的边缘情况。

        # [FIX] 导入语句被移动到此处以解决循环导入问题。
        from baseline_models import HEFTScheduler

        heft_scheduler = HEFTScheduler()
        heft_makespan = heft_scheduler.schedule(problem)
        self.normalization_scale = heft_makespan if heft_makespan > 0 else 1.0
        # ========================= [ 修改结束 ] =========================

        # --- 初始化动态状态 (Dynamic State) ---
        self.task_status = np.full(self.n_max, TASK_STATUS_UNSCHEDULED, dtype=int)
        self.task_finish_times = np.zeros(self.n_max, dtype=np.float32)
        self.task_assignments = np.full(self.n_max, -1, dtype=int)
        self.processor_available_times = np.zeros(self.m_max, dtype=np.float32)

        self.predecessor_counts = np.zeros(self.n_max, dtype=int)
        for i in range(self.num_tasks):
            self.predecessor_counts[i] = len(self.predecessors.get(i, set()))

        self.num_done_tasks = 0
        self._update_initial_ready_tasks()

        return self.get_state()

    def get_state(self) -> Dict[str, np.ndarray]:
        """将当前环境状态打包成一个字典，供神经网络使用。"""
        status_one_hot = np.eye(4)[self.task_status].astype(np.float32)

        # ============================ [ 代码修改 4/4 ] ============================
        # [原因] 根据“模块一”，所有与绝对时间相关的状态变量在输出前，都必须
        #        使用当前 episode 的静态尺度因子 self.normalization_scale 进行归一化。
        # [方案] 将 task_finish_times 和 processor_available_times 除以 self.normalization_scale。
        state = {
            "adj_matrix": self.adj.copy(), "comp_costs": self.comp_costs.copy(),
            "comm_costs": self.comm_costs.copy(), "proc_speeds": self.proc_speeds.copy(),
            "task_mask": self.task_mask.copy(), "proc_mask": self.proc_mask.copy(),
            "task_status": status_one_hot,
            "task_finish_times": self.task_finish_times.copy() / self.normalization_scale,
            "processor_available_times": self.processor_available_times.copy() / self.normalization_scale
        }
        # ========================= [ 修改结束 ] =========================
        return state

    def get_legal_actions_mask(self) -> np.ndarray:
        """计算并返回一个布尔掩码，指示哪些动作是合法的。"""
        legal_mask = np.zeros((self.n_max, self.m_max), dtype=bool)
        ready_tasks = np.where(self.task_status == TASK_STATUS_READY)[0]
        if ready_tasks.size > 0:
            legal_mask[np.ix_(ready_tasks, self.proc_mask)] = True
        return legal_mask.flatten()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """执行一个动作并推进环境状态。"""
        task_id_np, proc_id_np = np.unravel_index(action, (self.n_max, self.m_max))
        task_id, proc_id = int(task_id_np), int(proc_id_np)

        # --- 1. 计算任务开始时间 ---
        processor_ready_time = self.processor_available_times[proc_id]
        predecessor_finish_time = 0.0
        task_predecessors = self.predecessors.get(task_id, set())

        if task_predecessors:
            pred_indices = list(task_predecessors)
            pred_finish_times = self.task_finish_times[pred_indices]
            pred_proc_ids = self.task_assignments[pred_indices]
            comm_costs = self.comm_costs[pred_indices, task_id]
            is_remote_comm = (pred_proc_ids != proc_id)
            ready_times_from_preds = pred_finish_times + comm_costs * is_remote_comm
            if ready_times_from_preds.size > 0:
                predecessor_finish_time = np.max(ready_times_from_preds)

        start_time = max(processor_ready_time, predecessor_finish_time)

        # --- 2. 计算任务完成时间 ---
        computation_time = self.comp_costs[task_id] / (self.proc_speeds[proc_id] + 1e-8)
        finish_time = start_time + computation_time

        # --- 3. 更新动态状态 ---
        self.task_status[task_id] = TASK_STATUS_DONE
        self.task_finish_times[task_id] = finish_time
        self.task_assignments[task_id] = proc_id
        self.processor_available_times[proc_id] = finish_time
        self.num_done_tasks += 1

        # --- 4. 高效更新就绪任务 ---
        self._update_ready_tasks_optimized(finished_task_id=task_id)

        # --- 5. 检查终止条件 ---
        done = (self.num_done_tasks == self.num_tasks)
        reward = 0.0
        if done:
            makespan = self.get_makespan()
            reward = -makespan

        return self.get_state(), reward, done, {}

    def _update_initial_ready_tasks(self):
        """内部方法，在reset时调用，找到所有没有前驱的初始任务。"""
        initial_ready_tasks = np.where(self.predecessor_counts[:self.num_tasks] == 0)[0]
        if initial_ready_tasks.size > 0:
            self.task_status[initial_ready_tasks] = TASK_STATUS_READY

    def _update_ready_tasks_optimized(self, finished_task_id: int):
        """内部方法（优化版），通过递减后继任务的前驱计数器来高效更新就绪状态。"""
        for succ_id in self.successors.get(finished_task_id, set()):
            if self.task_status[succ_id] == TASK_STATUS_UNSCHEDULED:
                self.predecessor_counts[succ_id] -= 1
                if self.predecessor_counts[succ_id] == 0:
                    self.task_status[succ_id] = TASK_STATUS_READY

    def get_makespan(self) -> float:
        """计算最终的完工时间。"""
        return np.max(self.task_finish_times[:self.num_tasks]) if self.num_tasks > 0 else 0.0

    def clone(self):
        """
        [核心性能优化] 创建一个超轻量级的环境副本，用于MCTS模拟。
        """
        new_env = SchedulingEnvironment(self.n_max, self.m_max)

        # 引用传递静态数据
        new_env.problem = self.problem
        new_env.num_tasks = self.num_tasks
        new_env.num_processors = self.num_processors
        new_env.adj = self.adj
        new_env.comm_costs = self.comm_costs
        new_env.comp_costs = self.comp_costs
        new_env.proc_speeds = self.proc_speeds
        new_env.task_mask = self.task_mask
        new_env.proc_mask = self.proc_mask
        new_env.predecessors = self.predecessors
        new_env.successors = self.successors
        # ============================ [ 代码修改 (克隆逻辑) ] ============================
        # [原因] 克隆的环境也必须拥有与原始环境相同的、固定的归一化尺度。
        # [方案] 将 normalization_scale 属性一同传递给新副本。
        new_env.normalization_scale = self.normalization_scale
        # ========================= [ 修改结束 ] =========================

        # 深度复制动态数据
        new_env.task_status = self.task_status.copy()
        new_env.task_finish_times = self.task_finish_times.copy()
        new_env.task_assignments = self.task_assignments.copy()
        new_env.processor_available_times = self.processor_available_times.copy()
        new_env.num_done_tasks = self.num_done_tasks
        new_env.predecessor_counts = self.predecessor_counts.copy()

        return new_env