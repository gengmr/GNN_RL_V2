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
import copy
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


def run_single_pretrain_test(problem: SchedulingProblem, model_state_dict: Dict, device_str: str, reward_stats: Dict) -> \
        Tuple[float, float]:
    """在一个独立进程中为预训练评估运行单个测试问题。"""
    device = torch.device(device_str)
    model = DualHeadGNN(cfg.N_MAX, cfg.M_MAX, cfg.NN_CONFIG)
    model.load_state_dict(model_state_dict)
    model.to(device).eval()
    mock_trainer = MockTrainer(reward_stats)
    mcts = MCTS(model, mock_trainer, cfg.MCTS_CONFIG)
    heft_scheduler = HEFTScheduler()
    heft_stateful_scheduler = HEFTStatefulScheduler(problem)
    _, model_makespan = play_game(model, mcts, problem, is_self_play=False, heft_scheduler=heft_stateful_scheduler)
    heft_makespan = heft_scheduler.schedule(problem)
    return model_makespan, heft_makespan


# ============================ [ 代码修改 1/4 - 新增专用 Collate Fn ] ============================
# [原因] 预训练数据格式已变为 (State, Policy)，需要一个专门的函数来处理这种格式的批次。
# [方案] 创建一个新函数 collate_fn_pretrain_policy_only，保持原 collate_fn 不变以兼容强化学习阶段。
def collate_fn_pretrain_policy_only(batch: List[Tuple[Dict, np.ndarray]]) -> Tuple[
    Dict[str, torch.Tensor], torch.Tensor]:
    """自定义的collate函数，用于仅包含策略目标的预训练批处理。"""
    states_list, policies_list = zip(*[exp for exp in batch])
    policies = torch.tensor(np.array(policies_list), dtype=torch.float32)
    batched_states = {}
    if states_list:
        first_state_keys = states_list[0].keys()
        for key in first_state_keys:
            numpy_list = [s[key] for s in states_list]
            batched_states[key] = torch.from_numpy(np.stack(numpy_list))
    return batched_states, policies


# ========================= [ 修改结束 ] =========================


def collate_fn(batch: List[Tuple[Dict, np.ndarray, float]]) -> Tuple[
    Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """自定义的collate函数，用于批处理 (强化学习阶段)。"""
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


# ============================ [ 代码修改 2/4 - 更新 ExpertDataset ] ============================
# [原因] ExpertDataset 需要反映新的数据元组格式。
# [方案] 更新 __init__ 和 __getitem__ 的类型提示。
class ExpertDataset(Dataset):
    """一个简单的PyTorch数据集包装器，用于专家数据列表。"""

    def __init__(self, data: List[Tuple[Dict, np.ndarray]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Dict, np.ndarray]:
        return self.data[idx]


# ========================= [ 修改结束 ] =========================


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
        self.expected_columns = [
            'iteration', 'avg_total_loss', 'avg_value_loss', 'avg_policy_loss',
            'learning_rate', 'promoted', 'avg_cand_makespan', 'avg_best_makespan',
            'avg_heft_makespan', 'improvement_vs_heft', 'guidance_epsilon',
            'value_target_mean', 'value_target_std'
        ]
        self._ensure_metrics_file_exists()
        self._load_metrics()
        self.reward_n = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        self.improvement_history = deque(maxlen=3)

    def _setup_dirs(self):
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        eval_dir = os.path.dirname(cfg.EVAL_DATASET_PATH)
        if eval_dir:
            os.makedirs(eval_dir, exist_ok=True)

    def _ensure_metrics_file_exists(self):
        if not os.path.exists(cfg.METRICS_FILE):
            pd.DataFrame(columns=self.expected_columns).to_csv(cfg.METRICS_FILE, index=False)

    def _load_metrics(self):
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
        self.reward_n += 1
        delta = raw_reward - self.reward_mean
        self.reward_mean += delta / self.reward_n
        delta2 = raw_reward - self.reward_mean
        self.reward_m2 += delta * delta2

    def normalize_reward(self, raw_reward: float) -> float:
        if self.reward_n < 2: return 0.0
        variance = self.reward_m2 / (self.reward_n - 1)
        std_dev = math.sqrt(variance)
        if std_dev < 1e-6: return 0.0
        return (raw_reward - self.reward_mean) / std_dev

    def _run_pretrain_evaluation(self, epoch: int):
        print(f"\n[INFO] Running pre-train evaluation for epoch {epoch + 1}...")
        eval_problems = []
        if not os.path.exists(cfg.EVAL_DATASET_PATH):
            print(f"  [INFO] Evaluation dataset not found. Generating at {cfg.EVAL_DATASET_PATH}...")
            problem_gen = ProblemGenerator(**cfg.GENERATOR_CONFIG)
            eval_problems = [problem_gen.generate(seed=i) for i in range(cfg.TRAIN_CONFIG['eval_num_games'])]
            with open(cfg.EVAL_DATASET_PATH, 'wb') as f:
                pickle.dump(eval_problems, f)
        else:
            with open(cfg.EVAL_DATASET_PATH, 'rb') as f:
                eval_problems = pickle.load(f)
        self.model.eval()
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        reward_stats = {'n': self.reward_n, 'mean': self.reward_mean, 'm2': self.reward_m2}
        task = functools.partial(run_single_pretrain_test, model_state_dict=model_state_dict,
                                 device_str=str(self.device), reward_stats=reward_stats)
        num_workers = min(cpu_count(), len(eval_problems))
        all_results = []
        with mp.Pool(processes=num_workers) as pool:
            pbar = tqdm(pool.imap(task, eval_problems), total=len(eval_problems),
                        desc="  [PROGRESS] Pre-train Evaluating")
            for result in pbar:
                all_results.append(result)
        avg_model_makespan = np.mean([r[0] for r in all_results]) if all_results else 0
        avg_heft_makespan = np.mean([r[1] for r in all_results]) if all_results else 0
        improvement = (1 - avg_model_makespan / avg_heft_makespan) * 100 if avg_heft_makespan > 0 else 0
        print(f"  [RESULT] Epoch {epoch + 1} Evaluation | Avg Model Makespan: {avg_model_makespan:.2f} | "
              f"Avg HEFT Makespan: {avg_heft_makespan:.2f} | Improvement: {improvement:.2f}%")
        detailed_results = [{'epoch': epoch + 1, 'problem_index': i, 'model_makespan': r[0], 'heft_makespan': r[1]}
                            for i, r in enumerate(all_results)]
        results_df = pd.DataFrame(detailed_results)
        try:
            file_exists = os.path.exists(cfg.PRETRAIN_EVAL_RESULTS_FILE)
            results_df.to_csv(cfg.PRETRAIN_EVAL_RESULTS_FILE, mode='a', header=not file_exists, index=False,
                              float_format='%.4f')
            print(f"  [INFO] Pre-train evaluation details saved to {cfg.PRETRAIN_EVAL_RESULTS_FILE}")
        except Exception as e:
            print(f"  [ERROR] Could not save pre-train evaluation results: {e}")
        self.model.train()

    # ============================ [ 代码修改 3/4 - 重写预训练方法 ] ============================
    # [原因] 实现仅 P_loss 训练的核心逻辑。
    # [方案] 移除所有与 V_loss 相关的代码，包括数据归一化、损失计算和日志记录。
    #        使用新创建的 collate_fn_pretrain_policy_only。
    def pretrain_with_expert_data(self, data_dir: str):
        """
        使用专家数据集对策略头进行监督学习预训练 (仅模仿动作，不训练价值头)。
        """
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

        # [已删除] 所有与价值目标相关的归一化逻辑均已移除。

        pytorch_dataset = ExpertDataset(expert_dataset)
        data_loader = DataLoader(pytorch_dataset, batch_size=cfg.PRETRAIN_CONFIG['pretrain_batch_size'],
                                 shuffle=True, collate_fn=collate_fn_pretrain_policy_only)  # <-- 使用专用collate_fn

        self.model.train()
        for epoch in range(start_epoch, cfg.PRETRAIN_CONFIG['pretrain_epochs']):
            total_p_loss = 0
            pbar = tqdm(data_loader,
                        desc=f"[PROGRESS] Pre-train Epoch {epoch + 1}/{cfg.PRETRAIN_CONFIG['pretrain_epochs']}")

            for batch_states, expert_policies in pbar:
                for key in batch_states:
                    batch_states[key] = batch_states[key].to(self.device)
                expert_policies = expert_policies.to(self.device)

                pretrain_optimizer.zero_grad()

                # 模型仍然会输出价值，但我们忽略它
                predicted_policy_logits, _, _, _ = self.model(batch_states)

                # 损失函数现在只包含策略损失
                policy_loss = F.cross_entropy(predicted_policy_logits, expert_policies)

                # 直接反向传播策略损失
                policy_loss.backward()
                pretrain_optimizer.step()

                total_p_loss += policy_loss.item()
                pbar.set_postfix({"P_Loss": f"{policy_loss.item():.4f}"})

            avg_p_loss = total_p_loss / len(data_loader)
            print(
                f"[RESULT] Epoch {epoch + 1} complete. Avg Policy Loss: {avg_p_loss:.4f}")

            eval_interval = cfg.PRETRAIN_CONFIG.get('pretrain_eval_interval', 0)
            if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
                self._run_pretrain_evaluation(epoch)

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

    # ========================= [ 修改结束 ] =========================

    def update_improvement_history(self, improvement_rate: float):
        self.improvement_history.append(improvement_rate)

    def get_avg_improvement(self) -> float:
        if not self.improvement_history:
            return 0.0
        return float(np.mean(list(self.improvement_history)))

    # ============================ [ 代码修改 4/4 - 保持不变 ] ============================
    # [原因] 强化学习阶段的训练逻辑不受影响，它仍然需要处理价值目标。
    # [方案] train_step 方法保持原样，继续使用自适应损失函数同时训练策略和价值头。
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
            pred_policy_logits, pred_values, log_var_policy, log_var_value = self.model(batch_states)
            raw_value_loss = F.mse_loss(pred_values, target_values)
            raw_policy_loss = -torch.sum(target_policies * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()
            precision_policy = torch.exp(-log_var_policy)
            policy_loss_term = precision_policy * raw_policy_loss + log_var_policy
            precision_value = torch.exp(-log_var_value)
            value_loss_term = precision_value * raw_value_loss + log_var_value
            total_loss = (policy_loss_term + value_loss_term) * 0.5

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "value_loss": raw_value_loss.item(),
            "policy_loss": raw_policy_loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0]
        }

    # ========================= [ 修改结束 ] =========================

    def perform_training_steps(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        if len(replay_buffer) < cfg.TRAIN_CONFIG['train_batch_size']:
            print("  [WARNING] Replay buffer has fewer samples than batch size, skipping training.")
            return {"avg_total_loss": 0, "avg_value_loss": 0, "avg_policy_loss": 0,
                    "learning_rate": self.scheduler.get_last_lr()[0]}
        self.model.train()
        losses = []
        pbar = tqdm(range(cfg.TRAIN_CONFIG['num_training_steps']), desc="  [PROGRESS] Training Steps")
        for step in pbar:
            batch_experiences = replay_buffer.sample(cfg.TRAIN_CONFIG['train_batch_size'])
            batch_states, target_policies, target_values = collate_fn(batch_experiences)
            mean_val = torch.mean(target_values)
            std_val = torch.std(target_values)
            normalized_values = (target_values - mean_val) / (std_val + 1e-8)
            batch = (batch_states, target_policies, normalized_values)
            loss_dict = self.train_step(batch)
            losses.append(loss_dict)
            pbar.set_postfix({"T_Loss": f"{np.mean([l['total_loss'] for l in losses]):.4f}",
                              "P_Loss": f"{loss_dict['policy_loss']:.4f}",
                              "V_Loss": f"{loss_dict['value_loss']:.4f}"})
        pbar.close()
        avg_losses = {key: np.mean([d[key] for d in losses]) for key in losses[0]} if losses else {}
        return {f"avg_{k}": v for k, v in avg_losses.items()}

    def log_and_save_metrics(self, iteration: int, metrics: dict):
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