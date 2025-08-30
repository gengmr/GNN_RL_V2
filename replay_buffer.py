# -*- coding: utf-8 -*-
"""
经验回放池 (Replay Buffer) 模块。

该模块的核心是一个先进先出（FIFO）的队列，用于存储在自对弈过程中
产生的训练样本 `(state, policy, value)`。

作用与优势:
1.  解耦数据生成与消耗:
    自对弈过程（数据生成）和神经网络训练过程（数据消耗）可以异步进行。
    训练器可以持续地从池中采样数据进行学习，而无需等待新的自对弈游戏完成。

2.  打破数据相关性:
    在一个完整的游戏中，连续的状态是高度相关的。如果直接按顺序使用这些数据
    进行训练，会导致模型更新不稳定且可能陷入局部最优。通过从一个大的回放池
    中随机采样，可以打破这种时间上的相关性，使得训练样本更接近独立同分布(I.I.D.)，
    从而提高训练的稳定性和效率。

3.  数据重用:
    一个有价值的经验可以被多次采样和学习，提高了数据的利用效率。
"""
import random
from collections import deque, namedtuple

# 定义经验元组的结构
Experience = namedtuple('Experience', ('state', 'policy', 'value'))

class ReplayBuffer:
    """一个固定大小的先进先出经验回放池。"""

    def __init__(self, capacity: int):
        """
        初始化回放池。

        Args:
            capacity (int): 回放池的最大容量。
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, state, policy, value):
        """
        将一个经验元组保存到内存中。

        Args:
            state (dict): 环境状态。
            policy (np.ndarray): MCTS生成的改进策略。
            value (float): 最终的游戏结果 (归一化后的奖励)。
        """
        self.memory.append(Experience(state, policy, value))

    def sample(self, batch_size: int):
        """
        从内存中随机采样一个批次的经验。

        Args:
            batch_size (int): 要采样的批次大小。

        Returns:
            list: 一个包含`batch_size`个`Experience`元组的列表。
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """返回当前内存中的经验数量。"""
        return len(self.memory)