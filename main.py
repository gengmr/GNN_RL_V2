# FILE: main.py

# -*- coding: utf-8 -*-
"""
主执行脚本：AlphaGo Zero for DAG Scheduling 训练与测试管线
...
"""
import argparse
import numpy as np
import os
from tqdm import tqdm
import torch
import math
import torch.multiprocessing as mp
from multiprocessing import cpu_count
import time
import datetime
import functools
import copy
import pickle

import config as cfg
from dag_generator import ProblemGenerator, SchedulingProblem
from environment import SchedulingEnvironment
from model import DualHeadGNN
from mcts import MCTS
from replay_buffer import ReplayBuffer
from trainer import Trainer, play_game, MockTrainer
import expert_data_generator
# ============================ [ 代码修改 1/3 ] ============================
# [原因] 需要HEFT调度器来计算价值目标所依赖的基准完工时间。
# [方案] 导入HEFTScheduler。
from baseline_models import HEFTScheduler
# ========================= [ 修改结束 ] =========================


def run_self_play_game(args_bundle):
    """在一个独立的进程中运行单场自对弈游戏。"""
    (model_state_dict, mcts_config, generator_config, nn_config,
     n_max, m_max, reward_stats, device_str, current_epsilon) = args_bundle

    mcts_config_local = copy.deepcopy(mcts_config)
    mcts_config_local['guidance_epsilon'] = current_epsilon

    device = torch.device(device_str)
    model = DualHeadGNN(n_max, m_max, nn_config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    mock_trainer = MockTrainer(reward_stats)
    mcts = MCTS(model, mock_trainer, mcts_config_local)
    problem_generator = ProblemGenerator(**generator_config)

    problem = problem_generator.generate()
    game_history, final_makespan = play_game(
        model=model,
        mcts=mcts,
        problem=problem,
        is_self_play=True
    )
    # ============================ [ 代码修改 2/3 ] ============================
    # [原因] 根据“模块二”，为了计算新的价值目标，父进程需要知道每个游戏
    #        所使用的具体问题实例 (problem)。
    # [方案] 修改函数返回值，将生成的 problem 对象一并返回。
    return game_history, final_makespan, problem
    # ========================= [ 修改结束 ] =========================


def self_play(best_model, problem_generator, replay_buffer, trainer, guidance_epsilon: float):
    """自对弈阶段，使用并行工作进程高效生成训练数据。"""
    best_model.eval()
    model_state_dict = {k: v.cpu() for k, v in best_model.state_dict().items()}

    num_games = cfg.TRAIN_CONFIG['num_self_play_games']
    configured_workers = cfg.TRAIN_CONFIG.get('self_play_parallel_workers', 0)
    if configured_workers > 0:
        num_workers = configured_workers
    else:
        num_workers = cfg.TRAIN_CONFIG['default_gpu_workers'] if cfg.DEVICE.type == 'cuda' else cpu_count()
    num_workers = min(num_workers, num_games)

    print(f"  [INFO] Generating {num_games} games using {num_workers} worker processes...")
    reward_stats = {'n': trainer.reward_n, 'mean': trainer.reward_mean, 'm2': trainer.reward_m2}

    task_bundle = (model_state_dict, cfg.MCTS_CONFIG, cfg.GENERATOR_CONFIG, cfg.NN_CONFIG,
                   cfg.N_MAX, cfg.M_MAX, reward_stats, str(cfg.DEVICE), guidance_epsilon)
    tasks = [copy.deepcopy(task_bundle) for _ in range(num_games)]

    all_results = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(pool.imap(run_self_play_game, tasks), total=num_games, desc="  [PROGRESS] Self-Play Games")
        for result in pbar:
            all_results.append(result)
            pbar.set_postfix_str(f"Collected {len(all_results)}/{num_games} results")

    # ============================ [ 代码修改 3/3 ] ============================
    # [原因] 实施“模块二”的扩展价值目标计算逻辑。
    # [方案]
    # 1. 实例化一个 HEFTScheduler 用于计算基准 makespan。
    # 2. 更改 for 循环以解包新的三元组返回结果 (game_history, final_makespan, problem)。
    # 3. 对每个游戏结果，计算 HEFT makespan。
    # 4. 根据方案中定义的分段线性映射函数，计算新的 `target_value`。
    # 5. 将 `(state, policy, target_value)` 推入回放池，取代旧的归一化奖励。
    # 6. 保留 `update_reward_stats` 以继续为MCTS和日志提供原始奖励的统计信息。
    heft_solver = HEFTScheduler()
    new_experiences = 0
    for game_history, final_makespan, problem in all_results:
        if not game_history: continue

        # 步骤 1 & 2: 获取双方表现并计算原始性能比率
        makespan_RL = final_makespan
        makespan_HEFT = heft_solver.schedule(problem)

        if makespan_HEFT > 0:
            performance_ratio = makespan_RL / makespan_HEFT
        else: # 避免除零错误
            performance_ratio = 1.0 if makespan_RL == 0 else float('inf')

        # 步骤 3: 平滑的非对称区间映射
        if performance_ratio <= 1.0:
            # 性能超越HEFT: [0.95, 1.0] -> [+1.0, 0.0]
            expanded_score = (1.0 - performance_ratio) / (1.0 - 0.95) if 1.0 - 0.95 > 0 else 0.0
        else:
            # 性能落后HEFT: (1.0, 1.2] -> (0.0, -1.0]
            expanded_score = (1.0 - performance_ratio) / (1.2 - 1.0) if 1.2 - 1.0 > 0 else 0.0

        # 步骤 4: 最终非线性压缩
        final_score = np.clip(expanded_score, -3.0, 3.0)
        target_value = np.tanh(final_score)

        # 为MCTS和日志记录更新原始奖励统计数据（此步骤保持不变）
        raw_reward = -final_makespan
        trainer.update_reward_stats(raw_reward)

        # 将带有新价值目标的经验存入回放池
        for state, policy in game_history:
            replay_buffer.push(state, policy, target_value)
            new_experiences += 1
    # ========================= [ 修改结束 ] =========================

    print(f"  [INFO] Collected {new_experiences} new experiences.")
    print(f"  [INFO] Replay Buffer size: {len(replay_buffer)} / {replay_buffer.memory.maxlen}.")
    std_dev = math.sqrt(trainer.reward_m2 / (trainer.reward_n - 1)) if trainer.reward_n > 1 else 0
    print(f"  [INFO] Updated reward stats: Mean={trainer.reward_mean:.1f}, StdDev={std_dev:.1f}.")


def main(args):
    # --- 1. 初始化核心组件 ---
    print("=" * 50)
    print("AlphaGo Zero for DAG Scheduling - Initialization")
    print("=" * 50)
    print(f"[INFO] Using device: {cfg.DEVICE}")
    MCTS_CONFIG_INITIAL = copy.deepcopy(cfg.MCTS_CONFIG)

    guidance_epsilon = MCTS_CONFIG_INITIAL['guidance_epsilon']

    problem_generator = ProblemGenerator(**cfg.GENERATOR_CONFIG)
    replay_buffer = ReplayBuffer(cfg.TRAIN_CONFIG['replay_buffer_size'])
    best_model = DualHeadGNN(cfg.N_MAX, cfg.M_MAX, cfg.NN_CONFIG).to(cfg.DEVICE)
    candidate_model = DualHeadGNN(cfg.N_MAX, cfg.M_MAX, cfg.NN_CONFIG).to(cfg.DEVICE)
    trainer = Trainer(candidate_model, cfg.DEVICE)

    # --- 专家知识预训练阶段 ---
    best_model_path = os.path.join(cfg.CHECKPOINT_DIR, cfg.BEST_MODEL_NAME)
    if args.pretrain or not os.path.exists(best_model_path):
        pretrain_phase_start = time.time()
        print("\n" + "=" * 50)
        print("[PHASE 0] Expert Knowledge Injection")
        print("=" * 50)

        expert_data_dir = cfg.EXPERT_DATA_DIR
        os.makedirs(expert_data_dir, exist_ok=True)

        expert_data_files = [f for f in os.listdir(expert_data_dir) if f.endswith('.pkl')]
        if not expert_data_files:
            print(f"[INFO] Expert data not found in '{expert_data_dir}'. Generating now...")
            expert_data_generator.generate_expert_data(
                num_problems=cfg.PRETRAIN_CONFIG['num_expert_problems'],
                output_dir=expert_data_dir,
                augment=True
            )

        print("\n[INFO] Starting supervised pre-training...")
        trainer.pretrain_with_expert_data(expert_data_dir)

        trainer.save_model_checkpoint(iteration=-1, model_name=cfg.BEST_MODEL_NAME, replay_buffer=replay_buffer,
                                      guidance_epsilon=guidance_epsilon)
        trainer.save_model_checkpoint(iteration=-1, model_name=cfg.LATEST_MODEL_NAME, replay_buffer=replay_buffer,
                                      guidance_epsilon=guidance_epsilon)
        duration = str(datetime.timedelta(seconds=int(time.time() - pretrain_phase_start)))
        print(f"[SUCCESS] Pre-training complete. Initial models saved. Phase duration: {duration}")

    # --- 加载检查点和回放池 ---
    print("\n" + "=" * 50)
    print("Loading Checkpoints and State")
    print("=" * 50)
    _, loaded_epsilon = trainer.load_checkpoint(cfg.LATEST_MODEL_NAME)
    if loaded_epsilon is not None:
        guidance_epsilon = loaded_epsilon
    best_model.load_state_dict(candidate_model.state_dict())

    buffer_path = os.path.join(cfg.CHECKPOINT_DIR, "replay_buffer.pkl")
    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, 'rb') as f:
                replay_buffer.memory = pickle.load(f)
            print(f"[SUCCESS] Replay buffer loaded from {buffer_path}. Current size: {len(replay_buffer)}")
        except Exception as e:
            print(f"[WARNING] Could not load replay buffer. Starting empty. Error: {e}")

    if args.test:
        trainer.test(args.test_set)
        return

    # --- 主训练循环 ---
    for i in range(trainer.start_iteration, cfg.TRAIN_CONFIG['num_iterations']):
        iter_start_time = time.time()
        total_duration_formatted = str(datetime.timedelta(seconds=int(trainer.total_training_time)))
        print("\n" + "=" * 70)
        print(f"[ITERATION {i + 1} / {cfg.TRAIN_CONFIG['num_iterations']}] | Total Time: {total_duration_formatted}")
        print("=" * 70)

        phase_start_time = time.time()
        print("  [PHASE 1/4] Self-Play")
        print("  " + "-" * 21)
        self_play(best_model, problem_generator, replay_buffer, trainer, guidance_epsilon)
        duration = str(datetime.timedelta(seconds=int(time.time() - phase_start_time)))
        print(f"  [SUCCESS] Self-Play phase completed in {duration}.")

        if len(replay_buffer) < cfg.TRAIN_CONFIG['train_batch_size']:
            print("  [WARNING] Replay buffer too small, skipping training phase for this iteration.")
            continue

        phase_start_time = time.time()
        print("\n  [PHASE 2/4] Training")
        print("  " + "-" * 21)
        metrics = trainer.perform_training_steps(replay_buffer)
        metrics.update({
            'promoted': 0,
            'avg_cand_makespan': np.nan,
            'avg_best_makespan': np.nan,
            'avg_heft_makespan': np.nan,
            'improvement_vs_heft': np.nan,
            'guidance_epsilon': guidance_epsilon,
            'reward_mean': trainer.reward_mean,
            'reward_std_dev': math.sqrt(trainer.reward_m2 / (trainer.reward_n - 1)) if trainer.reward_n > 1 else 0.0
        })
        print(
            f"  [RESULT] Avg Total Loss: {metrics.get('avg_total_loss', 0):.4f} | Avg Policy Loss: {metrics.get('avg_policy_loss', 0):.4f} | Avg Value Loss: {metrics.get('avg_value_loss', 0):.4f}")
        duration = str(datetime.timedelta(seconds=int(time.time() - phase_start_time)))
        print(f"  [SUCCESS] Training phase completed in {duration}.")
        promoted_this_iter = False

        if (i + 1) % cfg.TRAIN_CONFIG['evaluation_interval'] == 0:
            phase_start_time = time.time()
            print("\n  [PHASE 3/4] Arena Evaluation")
            print("  " + "-" * 26)
            promoted, eval_metrics = trainer.evaluate_and_promote(best_model, iteration=i)
            metrics.update(eval_metrics)
            if promoted:
                promoted_this_iter = True
                metrics['promoted'] = 1
                best_model.load_state_dict(candidate_model.state_dict())
            duration = str(datetime.timedelta(seconds=int(time.time() - phase_start_time)))
            print(f"  [SUCCESS] Evaluation phase completed in {duration}.")

            if eval_metrics.get('avg_heft_makespan', 0) > 0 and eval_metrics.get('avg_cand_makespan') is not None:
                improvement_rate = 1.0 - (eval_metrics['avg_cand_makespan'] / eval_metrics['avg_heft_makespan'])
                trainer.update_improvement_history(improvement_rate)

            decay_factor = cfg.MCTS_CONFIG['guidance_epsilon_decay']
            if trainer.get_avg_improvement() > 0.15:
                decay_factor **= 2

            new_epsilon = max(guidance_epsilon * decay_factor, cfg.MCTS_CONFIG['guidance_min_epsilon'])
            guidance_epsilon = new_epsilon
            metrics['guidance_epsilon'] = guidance_epsilon
            print(
                f"  [DETAIL] Guidance epsilon updated to: {new_epsilon:.4f} (Avg improvement: {trainer.get_avg_improvement():.2%})")

            if not promoted and eval_metrics.get('avg_best_makespan', 0) > 0 and \
                    eval_metrics.get('avg_cand_makespan', float('inf')) > 1.1 * eval_metrics['avg_best_makespan']:
                reengaged_epsilon = min(guidance_epsilon + 0.05, MCTS_CONFIG_INITIAL['guidance_epsilon'])
                guidance_epsilon = reengaged_epsilon
                metrics['guidance_epsilon'] = guidance_epsilon
                print(
                    f"  [WARNING] 🚨 PERFORMANCE REGRESSION DETECTED! Re-engaging guidance. Epsilon set to: {reengaged_epsilon:.4f}")

        phase_start_time = time.time()
        print("\n  [PHASE 4/4] Housekeeping & Checkpointing")
        print("  " + "-" * 38)
        trainer.log_and_save_metrics(i, metrics)

        is_last_iter = (i + 1) == cfg.TRAIN_CONFIG['num_iterations']
        is_checkpoint_interval = (i + 1) % cfg.TRAIN_CONFIG['checkpoint_interval'] == 0
        if promoted_this_iter or is_checkpoint_interval or is_last_iter:
            trainer.save_model_checkpoint(i, cfg.LATEST_MODEL_NAME, replay_buffer, guidance_epsilon)
            if promoted_this_iter:
                trainer.save_model_checkpoint(i, cfg.BEST_MODEL_NAME, replay_buffer, guidance_epsilon)
        duration = str(datetime.timedelta(seconds=int(time.time() - phase_start_time)))
        print(f"  [SUCCESS] Housekeeping phase completed in {duration}.")

        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time
        trainer.total_training_time += iter_duration
        iter_duration_formatted = str(datetime.timedelta(seconds=int(iter_duration)))
        print(f"\n[SUMMARY] Iteration {i + 1} finished in {iter_duration_formatted}.")


if __name__ == "__main__":
    if cfg.DEVICE.type == 'cuda':
        try:
            mp.set_start_method('spawn', force=True)
            print("[INFO] Multiprocessing start method set to 'spawn' for CUDA safety.")
        except RuntimeError:
            print("[INFO] Multiprocessing start method already set.")

    parser = argparse.ArgumentParser(description="AlphaGo Zero for DAG Scheduling.")
    parser.add_argument("--test", action="store_true", help="Run in test-only mode, loading the best model.")
    parser.add_argument("--test_set", type=str, default=None, help="Path to a pre-generated test set file for testing.")
    parser.add_argument("--pretrain", action="store_true", help="Force pre-training even if a model exists.")
    args = parser.parse_args()
    main(args)