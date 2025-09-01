# -*- coding: utf-8 -*-
"""
专家数据集生成器。

该脚本负责生成用于第一阶段“专家知识深度注入”的训练数据。它通过以下步骤实现：
1.  使用 `ProblemGenerator` 创建大规模、多样化的问题实例。
2.  对于每个问题，利用 `HEFTStatefulScheduler` 模拟HEFT的决策过程，为问题的每一步生成
    一个 `(State, Action)` 对，用于策略模仿学习。
3.  (可选) 通过对问题成本施加微小扰动来进行数据增强，以提升模型的泛化能力。
4.  将所有生成的 `(State, Policy)` 数据分批打包并保存为多个文件，
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


# ============================ [ 代码修改 1/2 - 核心逻辑修改 ] ============================
# [原因] 根据新方案，预训练阶段不再学习价值函数，因此无需计算和保存价值目标。
# [方案] 移除 heft_makespan 和 estimated_return 的计算，并将返回类型更新为只包含 State 和 Policy。
def _generate_trajectory_for_problem(
        problem: SchedulingProblem
) -> List[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    为给定的调度问题生成完整的专家决策轨迹 (仅策略)。

    Args:
        problem (SchedulingProblem): 要解决的问题实例。

    Returns:
        List[Tuple[Dict, np.ndarray]]: 一个包含 (State, Policy) 元组的列表。
                                       如果处理中发生错误，则返回空列表。
    """
    trajectory = []
    try:
        # [已删除] 不再需要为价值目标计算整体的完工时间。
        # heft_makespan = heft_solver.schedule(problem)
        # estimated_return = -heft_makespan

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

            # [修改] 仅追加 state 和 policy_target
            trajectory.append((current_state, policy_target))

            _, _, done, _ = env.step(expert_action)
    except Exception as e:
        print(f"\n[WARNING] Error generating trajectory for a problem: {e}. Skipping this trajectory.")
        return []
    return trajectory


# ========================= [ 修改结束 ] =========================


def generate_expert_data(num_problems: int, output_dir: str, augment: bool = True):
    """
    生成一个包含 (State, Policy) 元组的数据集，并分批保存。
    """
    gen_start_time = time.time()
    print("\n" + "-" * 50)
    print("[PHASE] Expert Knowledge Data Generation (Policy Only)")
    print("-" * 50)

    os.makedirs(output_dir, exist_ok=True)

    problem_generator = ProblemGenerator(**cfg.GENERATOR_CONFIG)
    # [修改] 虽然不计算整体 makespan，但 trajectory 生成仍需 heft_solver 实例 (尽管未使用)
    # 为保持函数签名一致性，暂时保留。或者可以重构 _generate_trajectory_for_problem，移除该参数。
    # 此处选择保留，以最小化代码改动。
    heft_solver = HEFTScheduler()

    metadata_path = os.path.join(output_dir, ".meta")
    SAVE_INTERVAL = cfg.PRETRAIN_CONFIG.get('expert_data_save_interval', 500)

    start_problem_idx = 0
    if os.path.exists(metadata_path):
        print(f"[INFO] Found existing metadata at '{metadata_path}'. Attempting to resume.")
        try:
            with open(metadata_path, 'r') as f:
                last_completed_idx = int(f.read())
                start_problem_idx = last_completed_idx + 1
            existing_files = [f for f in os.listdir(output_dir) if
                              f.startswith('expert_data_part_') and f.endswith('.pkl')]
            print(
                f"[SUCCESS] Resuming generation from problem index {start_problem_idx}. Found {len(existing_files)} existing data files.")
        except Exception as e:
            print(f"[WARNING] Could not load resume state. Starting from scratch. Error: {e}")
    else:
        print("[INFO] No existing metadata found. Starting new generation.")

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
        problem = problem_generator.generate(seed=i)

        # ============================ [ 代码修改 2/2 - 逻辑对齐 ] ============================
        # [原因] 确保调用的是更新后的、仅返回 (State, Policy) 的函数。
        # [方案] 无需修改调用代码，但要理解 batch_dataset 现在存储的是新格式的数据。
        trajectory = _generate_trajectory_for_problem(problem)
        batch_dataset.extend(trajectory)

        if augment and trajectory:
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

                augmented_trajectory = _generate_trajectory_for_problem(problem_prime)
                batch_dataset.extend(augmented_trajectory)

            except Exception as e:
                print(f"\n[WARNING] Skipping augmented problem for seed {i} due to an error: {e}")
        # ========================= [ 修改结束 ] =========================

        if (i + 1) % SAVE_INTERVAL == 0 and i > start_problem_idx:
            batch_num = i // SAVE_INTERVAL
            batch_path = os.path.join(output_dir, f"expert_data_part_{batch_num}.pkl")
            try:
                with open(batch_path, 'wb') as f:
                    pickle.dump(batch_dataset, f)
                with open(metadata_path, 'w') as f:
                    f.write(str(i))
                pbar.set_postfix_str(f"Saved batch {batch_num} ({len(batch_dataset)} samples). Mem freed.")
                batch_dataset.clear()
            except Exception as e:
                print(f"\n[WARNING] Error during periodic save: {e}")

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

    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    gen_duration = str(datetime.timedelta(seconds=int(time.time() - gen_start_time)))
    print(f"[SUCCESS] Expert dataset successfully saved to directory '{output_dir}' in {gen_duration}.")


if __name__ == '__main__':
    num_problems_to_gen = cfg.PRETRAIN_CONFIG.get('num_expert_problems', 1000)
    output_directory = cfg.EXPERT_DATA_DIR

    generate_expert_data(
        num_problems=num_problems_to_gen,
        output_dir=output_directory,
        augment=True
    )