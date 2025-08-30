# -*- coding: utf-8 -*-
"""
专家数据集生成器。

该脚本负责生成用于第一阶段“专家知识深度注入”的训练数据。它通过以下步骤实现：
1.  使用 `ProblemGenerator` 创建大规模、多样化的问题实例。
2.  对于每个问题，利用 `HEFTScheduler` 计算出一个高质量的调度方案，并记录其
    最终的完工时间 `M_heft`，作为价值目标。
3.  使用 `HEFTStatefulScheduler` 模拟HEFT的决策过程，为问题的每一步生成
    一个 `(State, Action)` 对。
4.  (可选) 通过对问题成本施加微小扰动来进行数据增强，以提升模型的泛化能力。
5.  将所有生成的 `(State, Policy, Return)` 数据分批打包并保存为多个文件，
    以避免在生成过程中消耗过多内存。
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
import time
import datetime
from typing import List, Tuple, Dict, Any

import config as cfg
from dag_generator import ProblemGenerator, SchedulingProblem
from environment import SchedulingEnvironment
from baseline_models import HEFTScheduler, HEFTStatefulScheduler


# ============================ [ 代码修改 1/4 - 新增辅助函数 ] ============================
# [原因] 解决代码重复问题（问题2）。生成轨迹的核心逻辑在原始问题和增强问题中完全相同。
# [方案] 将此逻辑提取到一个独立的、可重用的函数中。
def _generate_trajectory_for_problem(
        problem: SchedulingProblem,
        heft_solver: HEFTScheduler
) -> List[Tuple[Dict[str, np.ndarray], np.ndarray, float]]:
    """
    为给定的调度问题生成完整的专家决策轨迹。

    Args:
        problem (SchedulingProblem): 要解决的问题实例。
        heft_solver (HEFTScheduler): 用于计算基准完工时间的求解器。

    Returns:
        List[Tuple[...]]: 一个包含 (State, Policy, Estimated_Return) 元组的列表。
                         如果处理中发生错误，则返回空列表。
    """
    trajectory = []
    try:
        # 使用完整的HEFT调度器获取该问题的基准完工时间
        heft_makespan = heft_solver.schedule(problem)
        estimated_return = -heft_makespan

        # 使用状态化调度器获取分步决策轨迹
        env = SchedulingEnvironment(cfg.N_MAX, cfg.M_MAX)
        env.reset(problem)
        heft_stateful_scheduler = HEFTStatefulScheduler(problem)

        done = False
        while not done:
            current_state = env.get_state()
            legal_actions_mask = env.get_legal_actions_mask()
            if not np.any(legal_actions_mask):
                break

            expert_action = heft_stateful_scheduler.get_next_action(env)
            if expert_action == -1: break

            # 将专家动作转换为独热编码的策略目标
            policy_target = np.zeros(cfg.N_MAX * cfg.M_MAX, dtype=np.float32)
            policy_target[expert_action] = 1.0
            trajectory.append((current_state, policy_target, estimated_return))

            _, _, done, _ = env.step(expert_action)
    except Exception as e:
        print(f"\n[WARNING] Error generating trajectory for a problem: {e}. Skipping this trajectory.")
        return []
    return trajectory


# ========================= [ 修改结束 ] =========================


# ============================ [ 代码修改 2/4 - 函数签名与变量更新 ] ============================
# [原因] 为支持分批保存，函数现在应处理目录而非单个文件。
# [方案] 将 output_path 参数重命名为 output_dir，并更新相关变量。
def generate_expert_data(num_problems: int, output_dir: str, augment: bool = True):
    """
    生成一个包含 (State, Policy, Estimated_Return) 元组的数据集，并分批保存。
    此函数现在支持断点续训，且内存占用低。

    Args:
        num_problems (int): 要生成的基础问题数量。
        output_dir (str): 保存最终数据集分片文件的目录。
        augment (bool, optional): 是否执行数据增强。默认为 True。
    """
    # ========================= [ 修改结束 ] =========================
    gen_start_time = time.time()
    print("\n" + "-" * 50)
    print("[PHASE] Expert Knowledge Data Generation")
    print("-" * 50)

    os.makedirs(output_dir, exist_ok=True)

    problem_generator = ProblemGenerator(**cfg.GENERATOR_CONFIG)
    heft_solver = HEFTScheduler()

    # ============================ [ 代码修改 3/4 - 核心逻辑重构 ] ============================
    # [原因] 实现分批保存和低内存占用的核心逻辑。
    # [方案] 引入 batch_dataset 列表，在达到SAVE_INTERVAL时保存并清空。
    #        修改断点续训逻辑，使其不再加载旧数据到内存。
    metadata_path = os.path.join(output_dir, ".meta")
    SAVE_INTERVAL = cfg.PRETRAIN_CONFIG.get('expert_data_save_interval', 500)

    # --- 断点续训逻辑 (内存优化版) ---
    start_problem_idx = 0
    if os.path.exists(metadata_path):
        print(f"[INFO] Found existing metadata at '{metadata_path}'. Attempting to resume.")
        try:
            with open(metadata_path, 'r') as f:
                last_completed_idx = int(f.read())
                start_problem_idx = last_completed_idx + 1

            # 计算已存在的文件数量以提供参考信息
            existing_files = [f for f in os.listdir(output_dir) if
                              f.startswith('expert_data_part_') and f.endswith('.pkl')]
            print(
                f"[SUCCESS] Resuming generation from problem index {start_problem_idx}. Found {len(existing_files)} existing data files.")
        except Exception as e:
            print(f"[WARNING] Could not load resume state. Starting from scratch. Error: {e}")
    else:
        print("[INFO] No existing metadata found. Starting new generation.")

    # 该列表只存储当前批次的数据
    batch_dataset = []

    if start_problem_idx >= num_problems:
        print("[INFO] Expert data generation is already complete.")
        return

    print(f"[INFO] Target number of base problems: {num_problems}")
    print(f"[INFO] Data augmentation: {'Enabled' if augment else 'Disabled'}")
    print(f"[INFO] Saving data in batches of {SAVE_INTERVAL} problems.")

    pbar = tqdm(range(start_problem_idx, num_problems),
                desc="[PROGRESS] Generating Expert Data",
                initial=start_problem_idx,
                total=num_problems)

    for i in pbar:
        # 1. 生成一个基础问题
        problem = problem_generator.generate(seed=i)

        # --- 处理原始问题 (使用辅助函数) ---
        trajectory = _generate_trajectory_for_problem(problem, heft_solver)
        batch_dataset.extend(trajectory)

        # --- 2. (可选) 数据增强 (使用辅助函数) ---
        if augment and trajectory:  # 仅当原始问题成功时才增强
            try:
                perturbed_comp = problem.comp_costs * np.random.uniform(0.95, 1.05, size=problem.comp_costs.shape)
                adj_mask = problem.adj_matrix > 0
                perturbed_comm = problem.comm_costs * np.random.uniform(0.95, 1.05, size=problem.comm_costs.shape)
                perturbed_comm[~adj_mask] = 0

                problem_prime = SchedulingProblem(
                    adj_matrix=problem.adj_matrix,
                    comp_costs=perturbed_comp.astype(int),
                    comm_costs=perturbed_comm.astype(int),
                    num_tasks=problem.num_tasks,
                    num_processors=problem.num_processors,
                    proc_speeds=problem.proc_speeds
                )

                augmented_trajectory = _generate_trajectory_for_problem(problem_prime, heft_solver)
                batch_dataset.extend(augmented_trajectory)

            except Exception as e:
                print(f"\n[WARNING] Skipping augmented problem for seed {i} due to an error: {e}")

        # --- 3. 定期保存批次数据 ---
        if (i + 1) % SAVE_INTERVAL == 0 and i > start_problem_idx:
            batch_num = i // SAVE_INTERVAL
            batch_path = os.path.join(output_dir, f"expert_data_part_{batch_num}.pkl")
            try:
                with open(batch_path, 'wb') as f:
                    pickle.dump(batch_dataset, f)
                with open(metadata_path, 'w') as f:
                    f.write(str(i))

                pbar.set_postfix_str(f"Saved batch {batch_num} ({len(batch_dataset)} samples). Mem freed.")
                batch_dataset.clear()  # 释放内存
            except Exception as e:
                print(f"\n[WARNING] Error during periodic save: {e}")

    # --- 4. 最终保存 (保存剩余数据) ---
    print(f"\n[INFO] Generation loop complete.")
    if batch_dataset:
        final_batch_num = (num_problems - 1) // SAVE_INTERVAL
        final_batch_path = os.path.join(output_dir, f"expert_data_part_{final_batch_num}.pkl")
        try:
            print(f"[INFO] Saving final batch of {len(batch_dataset)} samples to '{final_batch_path}'...")
            with open(final_batch_path, 'wb') as f:
                pickle.dump(batch_dataset, f)
            batch_dataset.clear()
        except Exception as e:
            print(f"\n[ERROR] Failed to save final batch of data: {e}")

    # 清理元数据文件
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    gen_duration = str(datetime.timedelta(seconds=int(time.time() - gen_start_time)))
    print(f"[SUCCESS] Expert dataset successfully saved to directory '{output_dir}' in {gen_duration}.")
    # ========================= [ 修改结束 ] =========================


if __name__ == '__main__':
    # ============================ [ 代码修改 4/4 - 调用更新 ] ============================
    # [原因] 适配 generate_expert_data 的新函数签名和 config.py 的新变量。
    # [方案] 使用新的配置变量 cfg.EXPERT_DATA_DIR。
    num_problems_to_gen = cfg.PRETRAIN_CONFIG.get('num_expert_problems', 1000)
    output_directory = cfg.EXPERT_DATA_DIR

    generate_expert_data(
        num_problems=num_problems_to_gen,
        output_dir=output_directory,
        augment=True
    )
    # ========================= [ 修改结束 ] =========================