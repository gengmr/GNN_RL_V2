# =============================== FILE: trainer.py ===============================
# -*- coding: utf-8 -*-
"""
训练器与评估器模块。
...
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from typing import Tuple, List, Dict, Optional
import torch.multiprocessing as mp
from multiprocessing import cpu_count
import functools
from collections import deque

import config as cfg
from model import DualHeadGNN
from mcts import MCTS
from environment import SchedulingEnvironment
from dag_generator import ProblemGenerator, SchedulingProblem
from replay_buffer import ReplayBuffer
from baseline_models import HEFTScheduler, HEFTStatefulScheduler


class MockTrainer:
    """一个模拟训练器，仅用于在并行工作进程中提供奖励归一化所需的方法和属性。"""

    def __init__(self, stats: Dict):
        self.reward_n = stats.get('n', 0)
        self.reward_mean = stats.get('mean', 0.0)
        self.reward_m2 = stats.get('m2', 0.0)

    def normalize_reward(self, raw_reward: float) -> float:
        if self.reward_n < 2: return 0.0
        variance = self.reward_m2 / (self.reward_n - 1)
        std_dev = math.sqrt(variance)
        if std_dev < 1e-6: return 0.0
        return (raw_reward - self.reward_mean) / std_dev


def play_game(model: DualHeadGNN, mcts: MCTS, problem: SchedulingProblem, is_self_play: bool,
              heft_scheduler: Optional[HEFTStatefulScheduler] = None) -> Tuple[List[Tuple], float]:
    """通用函数：运行一局完整的游戏。"""
    env = SchedulingEnvironment(cfg.N_MAX, cfg.M_MAX)
    _ = env.reset(problem)

    if heft_scheduler is None:
        heft_scheduler = HEFTStatefulScheduler(problem)

    game_history = []
    done = False
    step = 0
    temperature_final_step = int(problem.num_tasks * cfg.MCTS_CONFIG['temperature_decay_ratio'])

    while not done:
        temp = 0.0
        if is_self_play:
            temp = cfg.MCTS_CONFIG['temperature_initial'] if step < temperature_final_step else 0.0
        sim_count = cfg.MCTS_CONFIG['num_simulations'] if is_self_play else cfg.TEST_CONFIG['mcts_simulations_for_test']

        action, pi = mcts.search(env, sim_count, temperature=temp, is_self_play=is_self_play,
                                 heft_scheduler=heft_scheduler)

        if action == -1:
            break

        if is_self_play:
            game_history.append((env.get_state(), pi))

        _, _, done, _ = env.step(action)
        step += 1

    return game_history, env.get_makespan()


def run_single_match(problem: SchedulingProblem, candidate_state_dict, best_state_dict, device_str: str,
                     reward_stats: Dict) -> Tuple[int, float, float, float]:
    """在一个独立的进程中运行单场比赛，并与HEFT基准进行比较。"""
    device = torch.device(device_str)
    candidate_model = DualHeadGNN(cfg.N_MAX, cfg.M_MAX, cfg.NN_CONFIG)
    candidate_model.load_state_dict(candidate_state_dict)
    candidate_model.to(device).eval()

    best_model = DualHeadGNN(cfg.N_MAX, cfg.M_MAX, cfg.NN_CONFIG)
    best_model.load_state_dict(best_state_dict)
    best_model.to(device).eval()

    mock_trainer = MockTrainer(reward_stats)
    candidate_mcts = MCTS(candidate_model, mock_trainer, cfg.MCTS_CONFIG)
    best_mcts = MCTS(best_model, mock_trainer, cfg.MCTS_CONFIG)

    heft_scheduler_instance = HEFTStatefulScheduler(problem)

    _, makespan_candidate = play_game(candidate_model, candidate_mcts, problem, is_self_play=False,
                                      heft_scheduler=heft_scheduler_instance)
    _, makespan_best = play_game(best_model, best_mcts, problem, is_self_play=False,
                                 heft_scheduler=heft_scheduler_instance)

    heft_scheduler = HEFTScheduler()
    makespan_heft = heft_scheduler.schedule(problem)

    candidate_win = 1 if makespan_candidate < makespan_best else 0
    return candidate_win, makespan_candidate, makespan_best, makespan_heft


def run_single_test(problem: SchedulingProblem, model_state_dict: Dict, device_str: str, reward_stats: Dict) -> Tuple[
    float, float, int, int]:
    """在一个独立进程中运行单个测试问题，并返回Agent和HEFT的makespan。"""
    device = torch.device(device_str)
    model = DualHeadGNN(cfg.N_MAX, cfg.M_MAX, cfg.NN_CONFIG)
    model.load_state_dict(model_state_dict)
    model.to(device).eval()

    mock_trainer = MockTrainer(reward_stats)
    mcts = MCTS(model, mock_trainer, cfg.MCTS_CONFIG)
    heft_scheduler = HEFTScheduler()

    heft_scheduler_instance = HEFTStatefulScheduler(problem)

    _, agent_makespan = play_game(model, mcts, problem, is_self_play=False, heft_scheduler=heft_scheduler_instance)
    heft_makespan = heft_scheduler.schedule(problem)

    return agent_makespan, heft_makespan, problem.num_tasks, problem.num_processors


def collate_fn(batch: List[Tuple[Dict, np.ndarray, float]]) -> Tuple[
    Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """自定义的collate函数，用于批处理。"""
    states_list, policies_list, values_list = zip(*[exp for exp in batch])
    policies = torch.tensor(np.array(policies_list), dtype=torch.float32)
    values = torch.tensor(np.array(values_list), dtype=torch.float32).view(-1, 1)
    batched_states = {}
    if states_list:
        first_state_keys = states_list[0].keys()
        for key in first_state_keys:
            numpy_list = [s[key] for s in states_list]
            batched_states[key] = torch.from_numpy(np.stack(numpy_list))
    return batched_states, policies, values


class ExpertDataset(Dataset):
    """一个简单的PyTorch数据集包装器，用于专家数据列表。"""

    def __init__(self, data: List[Tuple[Dict, np.ndarray, float]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Dict, np.ndarray, float]:
        return self.data[idx]


class Trainer:
    """负责模型训练、评估、保存和加载的核心类。"""

    def __init__(self, model: DualHeadGNN, device):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.NN_CONFIG['learning_rate'],
                                     weight_decay=cfg.NN_CONFIG['weight_decay'])

        warmup_steps = cfg.NN_CONFIG.get('warmup_steps', 0)
        total_training_steps = cfg.TRAIN_CONFIG['num_iterations'] * cfg.TRAIN_CONFIG['num_training_steps']

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.use_amp = cfg.NN_CONFIG.get('use_mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("[INFO] Automatic Mixed Precision (AMP) training is ENABLED.")
        else:
            print("[INFO] Automatic Mixed Precision (AMP) training is DISABLED.")

        self.start_iteration = 0
        self.metrics_df = pd.DataFrame()
        self.total_training_time = 0.0
        self._setup_dirs()
        # ============================ [ 代码修改 1/4 - 新增 ] ============================
        # [原因] 需要在日志中记录自适应学习到的损失权重，以便监控训练过程。
        # [方案] 在 expected_columns 列表中添加 'policy_weight' 和 'value_weight'。
        self.expected_columns = [
            'iteration', 'avg_total_loss', 'avg_value_loss', 'avg_policy_loss',
            'learning_rate', 'policy_weight', 'value_weight', 'promoted',
            'avg_cand_makespan', 'avg_best_makespan', 'avg_heft_makespan',
            'improvement_vs_heft', 'guidance_epsilon', 'reward_mean', 'reward_std_dev'
        ]
        # ========================= [ 修改结束 ] =========================
        self._ensure_metrics_file_exists()
        self._load_metrics()
        self.reward_n = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        self.improvement_history = deque(maxlen=3)

    def _setup_dirs(self):
        """创建用于保存模型和日志的目录。"""
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        eval_dir = os.path.dirname(cfg.EVAL_DATASET_PATH)
        if eval_dir:
            os.makedirs(eval_dir, exist_ok=True)

    def _ensure_metrics_file_exists(self):
        """检查指标文件是否存在，如果不存在则创建带有正确表头的空文件。"""
        if not os.path.exists(cfg.METRICS_FILE):
            pd.DataFrame(columns=self.expected_columns).to_csv(cfg.METRICS_FILE, index=False)

    def _load_metrics(self):
        """如果存在，则加载之前的训练指标文件，并确保所有期望的列都存在。"""
        if os.path.exists(cfg.METRICS_FILE) and os.path.getsize(cfg.METRICS_FILE) > 0:
            try:
                self.metrics_df = pd.read_csv(cfg.METRICS_FILE)
                for col in self.expected_columns:
                    if col not in self.metrics_df.columns:
                        self.metrics_df[col] = np.nan
                self.metrics_df = self.metrics_df[self.expected_columns]
            except Exception as e:
                print(f"[WARNING] Could not load metrics file. Starting fresh. Error: {e}")
                self.metrics_df = pd.DataFrame(columns=self.expected_columns)
        else:
            self.metrics_df = pd.DataFrame(columns=self.expected_columns)

    def update_reward_stats(self, raw_reward: float):
        """使用Welford's online algorithm在线更新奖励的均值和方差。"""
        self.reward_n += 1
        delta = raw_reward - self.reward_mean
        self.reward_mean += delta / self.reward_n
        delta2 = raw_reward - self.reward_mean
        self.reward_m2 += delta * delta2

    def normalize_reward(self, raw_reward: float) -> float:
        """使用当前的运行统计数据对奖励进行标准化 (z-score)。"""
        if self.reward_n < 2: return 0.0
        variance = self.reward_m2 / (self.reward_n - 1)
        std_dev = math.sqrt(variance)
        if std_dev < 1e-6: return 0.0
        return (raw_reward - self.reward_mean) / std_dev

    def pretrain_with_expert_data(self, data_dir: str):
        """使用专家数据集对策略头和价值头进行监督学习预训练。"""
        print(f"[INFO] Loading expert data from directory {data_dir}...")
        if not os.path.isdir(data_dir):
            print(f"[ERROR] Expert data directory not found at {data_dir}. Aborting pre-training.")
            return

        expert_dataset = []
        try:
            part_files = sorted(
                [f for f in os.listdir(data_dir) if f.startswith('expert_data_part_') and f.endswith('.pkl')])
            if not part_files:
                print(f"[ERROR] No expert data files ('expert_data_part_*.pkl') found in {data_dir}. Aborting.")
                return

            print(f"[INFO] Found {len(part_files)} data part files. Merging...")
            for filename in tqdm(part_files, desc="[PROGRESS] Loading data parts"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'rb') as f:
                    batch_data = pickle.load(f)
                    expert_dataset.extend(batch_data)
        except Exception as e:
            print(f"[ERROR] Failed to load or merge expert data parts. Error: {e}")
            return

        print(f"[INFO] Loaded {len(expert_dataset)} total expert samples.")
        pretrain_optimizer = optim.AdamW(self.model.parameters(), lr=cfg.PRETRAIN_CONFIG['pretrain_lr'])

        start_epoch = 0
        pretrain_checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, cfg.PRETRAIN_CHECKPOINT_NAME)
        if os.path.exists(pretrain_checkpoint_path):
            try:
                print(f"[INFO] Found pre-training checkpoint at '{pretrain_checkpoint_path}'. Resuming...")
                checkpoint = torch.load(pretrain_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                pretrain_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"[SUCCESS] Resumed pre-training from epoch {start_epoch}.")
            except Exception as e:
                print(f"[WARNING] Could not load pre-training checkpoint. Starting from scratch. Error: {e}")

        print("[INFO] Initializing reward stats based on expert data...")
        for _, _, raw_reward in expert_dataset:
            self.update_reward_stats(raw_reward)
        std_dev = math.sqrt(self.reward_m2 / (self.reward_n - 1)) if self.reward_n > 1 else 0
        print(f"[DETAIL] Initialized reward stats: Mean={self.reward_mean:.2f}, StdDev={std_dev:.2f}")

        pretrain_dataset = [(s, p, 0.0) for s, p, _ in expert_dataset]

        pytorch_dataset = ExpertDataset(pretrain_dataset)
        data_loader = DataLoader(pytorch_dataset, batch_size=cfg.PRETRAIN_CONFIG['pretrain_batch_size'],
                                 shuffle=True, collate_fn=collate_fn)

        self.model.train()
        for epoch in range(start_epoch, cfg.PRETRAIN_CONFIG['pretrain_epochs']):
            total_p_loss, total_v_loss, total_loss = 0, 0, 0
            pbar = tqdm(data_loader,
                        desc=f"[PROGRESS] Pre-train Epoch {epoch + 1}/{cfg.PRETRAIN_CONFIG['pretrain_epochs']}")
            for batch_states, expert_policies, expert_returns in pbar:
                for key in batch_states:
                    batch_states[key] = batch_states[key].to(self.device)
                expert_policies = expert_policies.to(self.device)
                expert_returns = expert_returns.to(self.device)

                pretrain_optimizer.zero_grad()

                # ============================ [ 代码修改 2/4 - 修改 ] ============================
                # [原因] 模型 forward 方法的返回值已改变，需要解包新的对数方差参数。
                #        同时，需要实现基于不确定性的自适应损失函数。
                # [方案]
                # 1. 解包 log_var_policy 和 log_var_value。
                # 2. 计算原始的 policy_loss 和 value_loss。
                # 3. 使用自适应损失公式计算 precision (1/variance) 和最终的加权损失。
                # 4. total_loss 是两个加权损失之和。
                predicted_policy_logits, predicted_values, log_var_p, log_var_v = self.model(batch_states)

                # 计算原始损失
                policy_loss = F.cross_entropy(predicted_policy_logits, expert_policies)
                value_loss = F.mse_loss(predicted_values.squeeze(), expert_returns.squeeze())

                # 计算自适应加权损失
                precision_p = torch.exp(-log_var_p)
                weighted_p_loss = precision_p * policy_loss + log_var_p

                precision_v = torch.exp(-log_var_v)
                weighted_v_loss = precision_v * value_loss + log_var_v

                # 最终总损失
                current_total_loss = 0.5 * (weighted_p_loss + weighted_v_loss)
                current_total_loss = current_total_loss.mean()

                current_total_loss.backward()
                # ========================= [ 修改结束 ] =========================
                pretrain_optimizer.step()

                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_loss += current_total_loss.item()
                pbar.set_postfix({"P_Loss": f"{policy_loss.item():.4f}", "V_Loss": f"{value_loss.item():.4f}",
                                  "P_weight": f"{precision_p.item():.2f}", "V_weight": f"{precision_v.item():.2f}"})

            avg_p_loss = total_p_loss / len(data_loader)
            avg_v_loss = total_v_loss / len(data_loader)
            print(
                f"[RESULT] Epoch {epoch + 1} complete. Avg Policy Loss: {avg_p_loss:.4f}, Avg Value Loss: {avg_v_loss:.4f}")

            save_interval = cfg.PRETRAIN_CONFIG.get('pretrain_save_interval', 0)
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                pretrain_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': pretrain_optimizer.state_dict(),
                }
                torch.save(pretrain_state, pretrain_checkpoint_path)
                print(f"[INFO] Saved pre-training checkpoint at epoch {epoch + 1} to '{pretrain_checkpoint_path}'.")

        if os.path.exists(pretrain_checkpoint_path):
            os.remove(pretrain_checkpoint_path)
            print(f"[INFO] Pre-training complete. Removed temporary checkpoint '{pretrain_checkpoint_path}'.")

    def update_improvement_history(self, improvement_rate: float):
        """更新最近N次评估的性能改善率列表。"""
        self.improvement_history.append(improvement_rate)

    def get_avg_improvement(self) -> float:
        """获取最近N次评估的平均改善率。"""
        if not self.improvement_history:
            return 0.0
        return float(np.mean(list(self.improvement_history)))

    def train_step(self, batch) -> dict:
        """执行单次训练更新，并集成了自动混合精度训练。"""
        batch_states, target_policies, target_values = batch
        for key in batch_states:
            batch_states[key] = batch_states[key].to(self.device, non_blocking=True)
        target_policies = target_policies.to(self.device, non_blocking=True)
        target_values = target_values.to(self.device, non_blocking=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            # ============================ [ 代码修改 3/4 - 修改 ] ============================
            # [原因] 与预训练阶段相同，需要实现自适应损失平衡。
            # [方案]
            # 1. 解包模型返回的 log_var_policy 和 log_var_value。
            # 2. 实现自适应损失函数，计算加权后的总损失。
            # 3. 将学习到的权重也返回，用于日志记录。
            pred_policy_logits, pred_values, log_var_p, log_var_v = self.model(batch_states)

            # 计算原始损失
            value_loss = F.mse_loss(pred_values, target_values)
            policy_loss = -torch.sum(target_policies * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()

            # 计算自适应加权损失
            precision_p = torch.exp(-log_var_p)
            weighted_p_loss = precision_p * policy_loss + log_var_p

            precision_v = torch.exp(-log_var_v)
            weighted_v_loss = precision_v * value_loss + log_var_v

            total_loss = 0.5 * (weighted_p_loss + weighted_v_loss)
            total_loss = total_loss.mean()

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "policy_weight": precision_p.item(),
            "value_weight": precision_v.item()
        }
        # ========================= [ 修改结束 ] =========================

    def perform_training_steps(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """执行一个迭代中的所有训练步骤。"""
        if len(replay_buffer) < cfg.TRAIN_CONFIG['train_batch_size']:
            print("  [WARNING] Replay buffer has fewer samples than batch size, skipping training.")
            return {"avg_total_loss": 0, "avg_value_loss": 0, "avg_policy_loss": 0,
                    "learning_rate": self.scheduler.get_last_lr()[0]}
        self.model.train()
        losses = []
        pbar = tqdm(range(cfg.TRAIN_CONFIG['num_training_steps']), desc="  [PROGRESS] Training Steps")
        for step in pbar:
            batch = collate_fn(replay_buffer.sample(cfg.TRAIN_CONFIG['train_batch_size']))
            loss_dict = self.train_step(batch)
            losses.append(loss_dict)
            pbar.set_postfix({"Loss": f"{np.mean([l['total_loss'] for l in losses]):.4f}",
                              "P_W": f"{losses[-1]['policy_weight']:.2f}",
                              "V_W": f"{losses[-1]['value_weight']:.2f}"})
        pbar.close()
        avg_losses = {key: np.mean([d[key] for d in losses]) for key in losses[0]} if losses else {}
        return {f"avg_{k}": v for k, v in avg_losses.items()}

    def log_and_save_metrics(self, iteration: int, metrics: dict):
        """将单次迭代的指标更新到内存中的DataFrame，并覆盖写入CSV文件。"""
        new_row_data = {col: metrics.get(col, np.nan) for col in self.expected_columns}
        new_row_data['iteration'] = iteration + 1
        new_row_df = pd.DataFrame([new_row_data])

        if not new_row_df.empty:
            existing_rows = self.metrics_df[self.metrics_df['iteration'] == new_row_data['iteration']]
            if not existing_rows.empty:
                self.metrics_df = self.metrics_df[self.metrics_df['iteration'] != new_row_data['iteration']]
            self.metrics_df = pd.concat([self.metrics_df, new_row_df], ignore_index=True)

        self.metrics_df.sort_values(by='iteration', inplace=True)
        self.metrics_df = self.metrics_df.reset_index(drop=True)

        try:
            self.metrics_df[self.expected_columns].to_csv(cfg.METRICS_FILE, index=False, float_format='%.6f')
            print(f"  [INFO] Metrics updated and saved to {cfg.METRICS_FILE}")
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred while saving metrics: {e}")

    def save_model_checkpoint(self, iteration: int, model_name: str, replay_buffer: ReplayBuffer,
                              guidance_epsilon: float):
        """保存模型、优化器、调度器、回放池以及混合精度缩放器的状态。"""
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        state = {'iteration': iteration, 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict(), 'reward_n': self.reward_n,
                 'reward_mean': self.reward_mean,
                 'reward_m2': self.reward_m2, 'total_training_time': self.total_training_time,
                 'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
                 'guidance_epsilon': guidance_epsilon,
                 'improvement_history': list(self.improvement_history)}
        torch.save(state, checkpoint_path)
        print(f"  [INFO] Saving {model_name} checkpoint...")

        buffer_path = os.path.join(cfg.CHECKPOINT_DIR, "replay_buffer.pkl")
        try:
            with open(buffer_path, 'wb') as f:
                pickle.dump(replay_buffer.memory, f)
            print(f"  [INFO] Replay buffer saved successfully.")
        except Exception as e:
            print(f"  [ERROR] Could not save replay buffer: {e}")

    def load_checkpoint(self, model_name: str) -> Tuple[bool, Optional[float]]:
        """加载模型、优化器、调度器以及混合精度缩放器的状态。"""
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint and checkpoint[
                    'scaler_state_dict'] is not None and self.use_amp: self.scaler.load_state_dict(
                    checkpoint['scaler_state_dict'])
                self.start_iteration = checkpoint.get('iteration', -1) + 1
                self.reward_n = checkpoint.get('reward_n', 0)
                self.reward_mean = checkpoint.get('reward_mean', 0.0)
                self.reward_m2 = checkpoint.get('reward_m2', 0.0)
                self.total_training_time = checkpoint.get('total_training_time', 0.0)

                loaded_epsilon = None
                if 'guidance_epsilon' in checkpoint:
                    loaded_epsilon = checkpoint['guidance_epsilon']
                    print(f"[INFO] Restored guidance epsilon to {loaded_epsilon:.4f}")

                if 'improvement_history' in checkpoint:
                    self.improvement_history = deque(checkpoint['improvement_history'],
                                                     maxlen=self.improvement_history.maxlen)
                print(
                    f"[SUCCESS] Resumed training from checkpoint: {checkpoint_path} at iteration {self.start_iteration}")
                return True, loaded_epsilon
            except Exception as e:
                print(f"[ERROR] Error loading checkpoint: {e}. Starting from scratch.")
                self.start_iteration = 0
                return False, None
        else:
            print(f"[INFO] No checkpoint found at {checkpoint_path}. Starting from scratch.")
            self.start_iteration = 0
            return False, None

    def evaluate_and_promote(self, best_model, iteration: int) -> Tuple[bool, Dict[str, float]]:
        """在竞技场中并行评估候选模型，并保存详细的比赛日志。"""
        if not os.path.exists(cfg.EVAL_DATASET_PATH):
            print(f"  [INFO] Evaluation dataset not found. Generating at {cfg.EVAL_DATASET_PATH}...")
            problem_gen = ProblemGenerator(**cfg.GENERATOR_CONFIG)
            eval_problems = [problem_gen.generate(seed=i) for i in range(cfg.TRAIN_CONFIG['eval_num_games'])]
            with open(cfg.EVAL_DATASET_PATH, 'wb') as f:
                pickle.dump(eval_problems, f)
        else:
            with open(cfg.EVAL_DATASET_PATH, 'rb') as f:
                eval_problems = pickle.load(f)

        candidate_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        best_state = {k: v.cpu() for k, v in best_model.state_dict().items()}
        reward_stats = {'n': self.reward_n, 'mean': self.reward_mean, 'm2': self.reward_m2}
        task = functools.partial(run_single_match, candidate_state_dict=candidate_state, best_state_dict=best_state,
                                 device_str=str(self.device), reward_stats=reward_stats)

        configured_workers = cfg.TRAIN_CONFIG.get('eval_parallel_workers', 0)
        if configured_workers > 0:
            num_workers = configured_workers
        else:
            num_workers = cfg.TRAIN_CONFIG['default_gpu_workers'] if self.device.type == 'cuda' else cpu_count()
        num_workers = min(num_workers, len(eval_problems))

        print(
            f"  [INFO] Evaluating candidate model vs best on {len(eval_problems)} games using {num_workers} workers...")

        all_results = []
        with mp.Pool(processes=num_workers) as pool:
            pbar = tqdm(pool.imap(task, eval_problems), total=len(eval_problems),
                        desc="  [PROGRESS] Evaluating")
            for result in pbar:
                all_results.append(result)
                win_rate = sum(r[0] for r in all_results) / len(all_results)
                pbar.set_postfix({"Win Rate": f"{win_rate:.2%}"})

        games_played = len(all_results)
        candidate_wins = sum(r[0] for r in all_results)
        win_rate_threshold = cfg.TRAIN_CONFIG['eval_win_rate_threshold']
        promoted = games_played > 0 and (candidate_wins / games_played) >= win_rate_threshold

        if games_played > 0:
            detailed_results = [{'iteration': iteration + 1, 'problem_index': i, 'candidate_makespan': r[1],
                                 'best_model_makespan': r[2], 'heft_makespan': r[3]} for i, r in enumerate(all_results)]
            results_df = pd.DataFrame(detailed_results)
            try:
                file_exists = os.path.exists(cfg.EVAL_RESULTS_DETAILS_FILE)
                results_df.to_csv(cfg.EVAL_RESULTS_DETAILS_FILE, mode='a', header=not file_exists, index=False,
                                  float_format='%.4f')
            except Exception as e:
                print(f"  [ERROR] Could not save detailed evaluation results: {e}")

        avg_candidate_makespan = np.mean([r[1] for r in all_results]) if all_results else 0
        avg_best_makespan = np.mean([r[2] for r in all_results]) if all_results else 0
        avg_heft_makespan = np.mean([r[3] for r in all_results]) if all_results else 0

        improvement_vs_heft = (1 - avg_candidate_makespan / avg_heft_makespan) * 100 if avg_heft_makespan > 0 else 0
        eval_metrics = {
            "avg_cand_makespan": avg_candidate_makespan,
            "avg_best_makespan": avg_best_makespan,
            "avg_heft_makespan": avg_heft_makespan,
            "improvement_vs_heft": improvement_vs_heft
        }

        final_win_rate = candidate_wins / games_played if games_played > 0 else 0.0
        print(
            f"  [RESULT] Candidate Wins: {candidate_wins}/{games_played} (Win Rate: {final_win_rate:.2%}) | Threshold: {win_rate_threshold:.2%}")
        print(
            f"  [RESULT] Avg Makespan (Candidate / Best / HEFT): {avg_candidate_makespan:.1f} / {avg_best_makespan:.1f} / {avg_heft_makespan:.1f}")
        print(f"  [RESULT] Improvement vs HEFT: {improvement_vs_heft:.2f}%")

        if promoted:
            print("  [PROMOTION] 🏆 New best model promoted!")
            return True, eval_metrics
        else:
            print("  [INFO] Candidate model did not meet the promotion threshold.")
            return False, eval_metrics

    def test(self, test_set_path: str = None):
        """在给定的测试集上并行测试最终模型的性能，并保存包含HEFT的详细结果。"""
        print("\n" + "=" * 50)
        print("[PHASE] Final Performance Test")
        print("=" * 50)
        self.model.eval()
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if test_set_path and os.path.exists(test_set_path):
            with open(test_set_path, 'rb') as f:
                test_problems = pickle.load(f)
            print(f"[INFO] Loaded {len(test_problems)} problems from {test_set_path}")
        else:
            print(f"[INFO] Generating {cfg.TEST_CONFIG['num_test_problems']} new problems for testing...")
            problem_gen = ProblemGenerator(**cfg.GENERATOR_CONFIG)
            test_problems = [problem_gen.generate(seed=cfg.TEST_CONFIG['test_seed'] + i)
                             for i in range(cfg.TEST_CONFIG['num_test_problems'])]

        reward_stats = {'n': self.reward_n, 'mean': self.reward_mean, 'm2': self.reward_m2}
        task = functools.partial(run_single_test, model_state_dict=model_state_dict,
                                 device_str=str(self.device), reward_stats=reward_stats)

        configured_workers = cfg.TRAIN_CONFIG.get('eval_parallel_workers', 0)
        if configured_workers > 0:
            num_workers = configured_workers
        else:
            num_workers = cfg.TRAIN_CONFIG['default_gpu_workers'] if self.device.type == 'cuda' else cpu_count()
        num_workers = min(num_workers, len(test_problems))

        print(f"[INFO] Starting parallel testing with {num_workers} worker processes...")

        all_results = []
        with mp.Pool(processes=num_workers) as pool:
            pbar = tqdm(pool.imap(task, test_problems), total=len(test_problems), desc="[PROGRESS] Testing")
            for result in pbar:
                all_results.append(result)

        detailed_results = [{'problem_index': i, 'agent_makespan': r[0], 'heft_makespan': r[1],
                             'num_tasks': r[2], 'num_processors': r[3]} for i, r in enumerate(all_results)]

        try:
            results_df = pd.DataFrame(detailed_results)
            results_df.to_csv(cfg.TEST_RESULTS_FILE, index=False, float_format='%.4f')
            print(f"\n[SUCCESS] Detailed test results saved to {cfg.TEST_RESULTS_FILE}")
        except Exception as e:
            print(f"\n[ERROR] Could not save detailed test results: {e}")

        print("\n--- Test Results Summary ---")
        print(f"Problems Tested: {len(all_results)}")
        avg_agent_makespan = np.mean([r[0] for r in all_results]) if all_results else 0.0
        avg_heft_makespan = np.mean([r[1] for r in all_results]) if all_results else 0.0
        print(f"Agent Average Makespan: {avg_agent_makespan:.4f}")
        print(f"HEFT Baseline Average Makespan: {avg_heft_makespan:.4f}")