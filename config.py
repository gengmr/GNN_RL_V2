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
    # ============================ [ 代码修改 1/1 - 已删除 ] ============================
    # [原因] 引入了自适应损失平衡机制，固定的 value_loss_weight 不再需要。
    # [方案] 删除此行。
    # "value_loss_weight": 1.0,
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
    "checkpoint_interval": 5,
    "evaluation_interval": 10,
    "eval_num_games": 100,
    "eval_win_rate_threshold": 0.55,
    "self_play_parallel_workers": 0,
    "eval_parallel_workers": 0,
    "generalization_check_interval": 10
}

# --- 7. 专家知识预训练参数 (Pre-training Config) ---
PRETRAIN_CONFIG = {
    "num_expert_problems": 50000,
    "pretrain_batch_size": 1024,
    "pretrain_epochs": 100,
    "pretrain_lr": 1e-4,
    "expert_data_save_interval": 500,
    "pretrain_save_interval": 10
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