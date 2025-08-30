# FILE: mcts.py

# -*- coding: utf-8 -*-
"""
蒙特卡洛树搜索 (MCTS) 模块。
...
"""
import numpy as np
import torch
import torch.nn.functional as F
import math
import config as cfg
from torch_geometric.utils import dense_to_sparse
# [修改 1/2] 导入 Union 和 Optional 以支持更灵活的类型提示
from typing import TYPE_CHECKING, Union, Optional
# [新增] 导入状态化HEFT调度器用于引导
from baseline_models import HEFTStatefulScheduler

if TYPE_CHECKING:
    # [修改] 导入 Trainer 和 MockTrainer 以便进行类型提示
    from trainer import Trainer, MockTrainer
    # [新增] 导入环境类以进行类型提示
    from environment import SchedulingEnvironment


class Node:
    """MCTS树中的节点。"""

    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # action -> Node
        self.n_visits = 0
        self.q_value = 0.0
        # [优化] 移除 u_value 状态。PUCT将在选择时即时计算，
        # 避免了存储和更新所有兄弟节点的开销，并修复了潜在的逻辑错误。
        self.p_value = prior_p  # 先验概率


class MCTS:
    """
    [核心性能重构]
    蒙特卡洛树搜索的高性能批处理实现 (Batch MCTS)。
    """

    # [修复] 更新类型提示，允许 trainer 参数为 Trainer 或 MockTrainer
    def __init__(self, model, trainer: Union['Trainer', 'MockTrainer'], mcts_config):
        self.model = model
        self.trainer = trainer
        self.mcts_config = mcts_config  # [FIX] Store the whole config
        self.c_puct = mcts_config['c_puct']
        self.dirichlet_alpha = mcts_config['dirichlet_alpha']
        self.dirichlet_epsilon = mcts_config['dirichlet_epsilon']
        # [新增] 从配置中获取虚拟工作者数量
        self.virtual_workers = mcts_config['mcts_virtual_workers']
        # [新增] 引导参数
        self.guidance_gating_threshold = mcts_config['guidance_gating_threshold']

    def _select_child(self, node: Node, heft_action: int, current_epsilon: float):
        """
        [修改] 使用混合了HEFT引导的PUCT公式选择子节点。
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent_visits = math.sqrt(node.n_visits)

        # --- 软性概率引导 ---
        # 1. 获取神经网络的原始策略概率分布
        p_nn_values = np.array([child.p_value for child in node.children.values()])
        actions = list(node.children.keys())

        # 2. 创建混合概率分布
        p_hybrid = (1 - current_epsilon) * p_nn_values

        # 3. 将epsilon权重“添加”到HEFT建议的动作上
        if heft_action in actions:
            heft_action_index = actions.index(heft_action)
            p_hybrid[heft_action_index] += current_epsilon

        for i, (action, child) in enumerate(node.children.items()):
            # 4. 使用混合概率计算U值
            prior_p = p_hybrid[i]
            u_value = self.c_puct * prior_p * sqrt_parent_visits / (1 + child.n_visits)
            score = child.q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _evaluate_batch(self, leaves):
        """
        [核心新函数]
        对一批叶子节点进行批处理神经网络评估。
        """
        # 1. 准备批处理数据
        states_list = [leaf['state'] for leaf in leaves]
        if not states_list:
            return

        # 使用trainer中的高效collate函数来构建批次
        from trainer import collate_fn
        batched_states, _, _ = collate_fn([(s, np.array([]), 0.0) for s in states_list])

        # 2. 模型推理
        with torch.no_grad():
            model_device = next(self.model.parameters()).device
            for key in batched_states:
                batched_states[key] = batched_states[key].to(model_device)

            # ============================ [ 代码修改 - Bug 修复 ] ============================
            # [原因] 模型 forward() 现在返回4个值，但这里只解包了2个，导致报错。
            # [方案] 解包所有4个返回值，并忽略在MCTS评估中不需要的后两个（log_vars）。
            policy_logits_batch, value_batch, _, _ = self.model(batched_states)
            # ========================= [ 修改结束 ] =========================

            policies = F.softmax(policy_logits_batch, dim=1).cpu().numpy()
            values = value_batch.cpu().numpy().flatten()

        # 3. 将结果分发回每个叶子节点
        for i, leaf_info in enumerate(leaves):
            node = leaf_info['node']
            legal_mask = leaf_info['mask']

            policy = policies[i]
            value = values[i]

            # 过滤非法动作
            policy *= legal_mask
            policy_sum = np.sum(policy)
            if policy_sum > 1e-8:
                policy /= policy_sum
            else:  # 如果所有合法动作概率都接近0，则赋予均匀概率
                num_legal = np.sum(legal_mask)
                if num_legal > 0:
                    policy = legal_mask / num_legal

            # 展开节点
            legal_actions = np.where(legal_mask)[0]
            for action in legal_actions:
                if action not in node.children:
                    node.children[action] = Node(parent=node, prior_p=policy[action])

            # 反向传播评估值
            self._backpropagate(leaf_info['path'], value)

    def _backpropagate(self, path, value: float):
        """反向传播更新节点的Q值和访问次数。"""
        for node in reversed(path):
            node.n_visits += 1
            node.q_value += (value - node.q_value) / node.n_visits

    def search(self, env: 'SchedulingEnvironment', num_simulations: int, temperature: float, is_self_play: bool = True,
               heft_scheduler: Optional[HEFTStatefulScheduler] = None):
        """
        [核心重构]
        在给定状态下执行高性能的批处理MCTS，并集成自适应引导。
        """
        root_state = env.get_state()
        root_mask = env.get_legal_actions_mask()
        root = Node(parent=None, prior_p=1.0)

        if not np.any(root_mask):
            pi = np.zeros(cfg.N_MAX * cfg.M_MAX, dtype=float)
            return -1, pi

        # --- [新增] 初始化启发式引导 ---
        if heft_scheduler is None:
            heft_scheduler = HEFTStatefulScheduler(env.problem)

        # [FIX] 从实例的配置中读取epsilon，而不是全局配置
        current_epsilon = self.mcts_config['guidance_epsilon']

        # --- 初始根节点扩展 ---
        initial_leaves_to_eval = [{'node': root, 'state': root_state, 'mask': root_mask, 'path': [root]}]
        self._evaluate_batch(initial_leaves_to_eval)

        if root.n_visits == 0:
            root.n_visits = 1

        # --- [新增] (高级) 置信度门控 ---
        current_search_epsilon = current_epsilon
        if is_self_play and current_epsilon > 0 and root.children:
            root_policy_probs = np.array([child.p_value for child in root.children.values()])
            root_actions = list(root.children.keys())
            root_heft_action = heft_scheduler.get_next_action(env)
            nn_best_action_idx = np.argmax(root_policy_probs)
            nn_best_action = root_actions[nn_best_action_idx]
            nn_max_confidence = root_policy_probs[nn_best_action_idx]

            if nn_max_confidence > self.guidance_gating_threshold and nn_best_action != root_heft_action:
                current_search_epsilon = current_epsilon * 0.1

        # --- 为自对弈添加狄利克雷噪声 ---
        if is_self_play:
            legal_actions = np.where(root_mask)[0]
            if len(legal_actions) > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
                for i, action in enumerate(legal_actions):
                    if action in root.children:
                        child_node = root.children[action]
                        child_node.p_value = (1 - self.dirichlet_epsilon) * child_node.p_value + \
                                             self.dirichlet_epsilon * noise[i]

        # --- 主模拟循环 ---
        remaining_sims = num_simulations - 1
        for _ in range(math.ceil(remaining_sims / self.virtual_workers)):
            leaves_to_eval = []
            for _ in range(self.virtual_workers):
                sim_env = env.clone()
                node = root
                path = [root]
                done = False
                while node.children:
                    # [修改] 调用HEFT获取当前状态的引导动作，并传入_select_child
                    heft_action_for_node = heft_scheduler.get_next_action(sim_env)
                    action, node = self._select_child(node, heft_action_for_node, current_search_epsilon)
                    if action == -1:
                        break
                    _, _, done, _ = sim_env.step(action)
                    path.append(node)
                    if done:
                        break

                value = 0.0
                if done:
                    # [注意] MCTS内部的价值评估仍使用原始奖励信号，
                    # 而非扩展后的价值目标。扩展价值目标仅用于最终的监督学习。
                    # trainer.normalize_reward 的存在是为了在MCTS搜索期间稳定Q值。
                    # 随着模型V头输出范围变为[-1, 1]，此处的归一化效果可能会改变，
                    # 但保留它作为稳定搜索的机制是合理的。
                    final_reward = -sim_env.get_makespan()
                    value = self.trainer.normalize_reward(final_reward) if self.trainer else final_reward
                    self._backpropagate(path, value)
                else:
                    leaf_state = sim_env.get_state()
                    leaf_mask = sim_env.get_legal_actions_mask()
                    if np.any(leaf_mask):
                        leaves_to_eval.append({'node': node, 'state': leaf_state, 'mask': leaf_mask, 'path': path})
                    else: # Dead end, no legal moves but not done
                        final_reward = -sim_env.get_makespan()
                        value = self.trainer.normalize_reward(final_reward) if self.trainer else final_reward
                        self._backpropagate(path, value)
            if leaves_to_eval:
                self._evaluate_batch(leaves_to_eval)

        # --- 决定最终动作 ---
        visits = np.zeros(cfg.N_MAX * cfg.M_MAX)
        for action, child in root.children.items():
            visits[action] = child.n_visits

        visit_sum = np.sum(visits)
        if visit_sum > 0:
            pi = visits / visit_sum
        else:
            num_legal_actions = np.sum(root_mask)
            pi = root_mask / num_legal_actions if num_legal_actions > 0 else np.zeros_like(root_mask, dtype=float)

        if temperature == 0:
            action = np.argmax(visits)
        else:
            action_probs = pi / np.sum(pi)
            action_indices = np.arange(len(pi))
            action = np.random.choice(action_indices, p=action_probs)

        return action, pi