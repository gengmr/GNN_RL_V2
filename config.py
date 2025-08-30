# =============================== FILE: config.py ================================
# -*- coding: utf-8 -*-
import torch
import os

# --- 1. 全局与路径设置 (Global & Path Settings) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_NAME = "AlphaGoZeroScheduler"
CHECKPOINT_DIR = "./checkpoints"  # 模型、优化器和指标的保存路径
METRICS_FILE = f"{CHECKPOINT_DIR}/training_metrics.csv" # 训练指标文件
TEST_RESULTS_FILE = f"{CHECKPOINT_DIR}/test_results_details.csv" # 测试结果详情文件
EVAL_RESULTS_DETAILS_FILE = f"{CHECKPOINT_DIR}/evaluation_results_details.csv" # 评估竞技场结果详情文件
# ============================ [ 代码修改 1/3 - 新增 ] ============================
# [原因] 为预训练阶段的定期评估提供独立的日志文件路径。
# [方案] 新增 PRETRAIN_EVAL_RESULTS_FILE 变量。
PRETRAIN_EVAL_RESULTS_FILE = f"{CHECKPOINT_DIR}/pretrain_evaluation_results.csv" # 预训练评估结果详情
# ========================= [ 修改结束 ] =========================
BEST_MODEL_NAME = "best_model.pth"
LATEST_MODEL_NAME = "latest_model.pth"
PRETRAIN_CHECKPOINT_NAME = "pretrain_checkpoint.pth"

EXPERT_DATA_DIR = "data/expert_data"

# --- 2. 问题生成器参数 (ProblemGenerator Config) ---
GENERATOR_CONFIG = {
    "num_tasks_range": (5, 15),
    "num_processors_range": (1, 4),
    "density_range": (0.2, 0.8),
    "comp_cost_range": (10, 100),
    "comm_cost_range": (5, 50),
    "proc_speed_range": (1.0, 2.0)
}

# --- 3. 环境与模型固定容量 (Environment & Model Capacity) ---
N_MAX = max(50, GENERATOR_CONFIG["num_tasks_range"][1])
M_MAX = max(10, GENERATOR_CONFIG["num_processors_range"][1])

# --- 4. 神经网络模型参数 (Neural Network Config) ---
NN_CONFIG = {
    "node_feature_dim": 5,
    "edge_feature_dim": 1,
    "processor_embedding_dim": 32,
    "embed_dim": 128,
    "num_layers": 4,
    "num_heads": 4,
    "ff_dim": 256,
    "dropout": 0.1,
    "policy_hidden_dim": 256,
    "value_hidden_dim": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    # ============================ [ 代码修改 2/3 - 修改 ] ============================
    # [原因] 实施了更科学的价值目标Z-Score归一化，使得P_Loss和V_Loss的量级天然对齐。
    # [方案] 将 value_loss_coeff 从一个较大的补偿值（如100.0）调整为一个更中性的、
    #        用于微调任务相对重要性的值。0.5是一个常见的、稳健的默认值。
    "value_loss_coeff": 0.5,
    # ========================= [ 修改结束 ] =========================
    "warmup_steps": 1000,
    "use_mixed_precision": True
}

# --- 5. 蒙特卡洛树搜索参数 (MCTS Config) ---
MCTS_CONFIG = {
    "num_simulations": 800,
    "c_puct": 2.5,
    "dirichlet_alpha": 0.1,
    "dirichlet_epsilon": 0.25,
    "temperature_initial": 1.0,
    "temperature_decay_ratio": 0.5,
    "mcts_virtual_workers": 48,
    "guidance_epsilon": 0.30,
    "guidance_epsilon_decay": 0.995,
    "guidance_min_epsilon": 0.01,
    "guidance_gating_threshold": 0.8,
}

# --- 6. 训练管线参数 (Training Pipeline Config) ---
TRAIN_CONFIG = {
    "num_iterations": 50000,
    "num_self_play_games": 256,
    "replay_buffer_size": 400000,
    "train_batch_size": 256,
    "default_gpu_workers": 48,
    "num_training_steps": 100,
    "checkpoint_interval": 2,
    "evaluation_interval": 10,
    "eval_num_games": 100,
    "eval_win_rate_threshold": 0.55,
    "self_play_parallel_workers": 0,
    "eval_parallel_workers": 0,
}

# --- 7. 专家知识预训练参数 (Pre-training Config) ---
PRETRAIN_CONFIG = {
    "num_expert_problems": 50000,
    "pretrain_batch_size": 2048,
    "pretrain_epochs": 100,
    "pretrain_lr": 1e-4,
    "expert_data_save_interval": 500,
    "pretrain_save_interval": 5,
    # ============================ [ 代码修改 3/3 - 新增 ] ============================
    # [原因] 实现预训练阶段的定期评估功能。
    # [方案] 新增 pretrain_eval_interval 参数，用于控制每隔多少个epoch进行一次评估。
    #        值为0表示禁用此功能。
    "pretrain_eval_interval": 10  # 每10个epoch在测试集上评估一次
    # ========================= [ 修改结束 ] =========================
}


# --- 8. 测试参数 (Testing Config) ---
TEST_CONFIG = {
    "num_test_problems": 100,
    "test_seed": 42,
    "mcts_simulations_for_test": 200
}

# 动态构建评估数据集路径
EVAL_DATASET_FILENAME = f"evaluation_set_{TRAIN_CONFIG['eval_num_games']}.pkl"
EVAL_DATASET_PATH = os.path.join("./test_sets", EVAL_DATASET_FILENAME)