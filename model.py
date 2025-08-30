# ================================ FILE: model.py ================================
# -*- coding: utf-8 -*-
"""
双头图注意力神经网络 (GAT) 模型。
...
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class GraphTransformerLayer(nn.Module):
    """
    一个标准的图 Transformer 层。

    该层通过全局自注意力机制使图中所有节点对之间都能进行信息交互，
    并通过前馈网络进行特征提炼。层归一化和残差连接用于稳定训练。
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x (Tensor): 输入节点嵌入, shape (B, N, D_embed).
            attn_mask (Tensor): 注意力偏置, shape (B * num_heads, N, N).
            key_padding_mask (Tensor): 填充掩码, shape (B, N).
        """
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_output)

        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output

        return x


class DualHeadGNN(nn.Module):
    """
    一个基于图Transformer的双头神经网络，用于预测调度策略和价值。
    """

    def __init__(self, n_max, m_max, nn_config):
        super().__init__()
        self.n_max = n_max
        self.m_max = m_max
        self.node_feature_dim = nn_config['node_feature_dim']
        self.edge_feature_dim = nn_config['edge_feature_dim']
        self.embed_dim = nn_config['embed_dim']
        self.policy_hidden_dim = nn_config['policy_hidden_dim']
        self.value_hidden_dim = nn_config['value_hidden_dim']
        self.processor_embedding_dim = nn_config['processor_embedding_dim']
        self.num_heads = nn_config['num_heads']

        self.feature_layernorm = nn.LayerNorm(self.node_feature_dim)
        self.proc_layernorm = nn.LayerNorm(2)
        self.node_embedder = nn.Linear(self.node_feature_dim, self.embed_dim)
        self.positional_embedding = nn.Embedding(self.n_max, self.embed_dim)
        self.edge_encoder = nn.Linear(self.edge_feature_dim, self.num_heads, bias=False)

        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=nn_config['ff_dim'],
                dropout=nn.config['dropout']
            ) for _ in range(nn_config['num_layers'])
        ])

        self.proc_encoder = nn.Sequential(
            nn.Linear(2, self.processor_embedding_dim),
            nn.ReLU()
        )

        policy_input_dim = self.embed_dim + self.processor_embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, self.policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.policy_hidden_dim, 1)
        )

        value_input_dim = self.embed_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, self.value_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.value_hidden_dim, 1),
            nn.Tanh()
        )

        # ============================ [ 代码修改 1/2 - 新增 ] ============================
        # [原因] 为实现基于不确定性的自适应损失平衡，需要为每个任务（策略和价值）
        #        定义一个可学习的对数方差参数。
        # [方案] 将 log_var_policy 和 log_var_value 初始化为 nn.Parameter。
        #        初始化为0意味着初始方差为 exp(0)=1，权重为1，这是一个中性的起点。
        self.log_var_policy = nn.Parameter(torch.zeros(1))
        self.log_var_value = nn.Parameter(torch.zeros(1))
        # ========================= [ 修改结束 ] =========================

    def forward(self, state):
        batch_size = state['task_mask'].shape[0]

        # --- 1. 构建初始节点嵌入 ---
        max_comp_cost = float(cfg.GENERATOR_CONFIG['comp_cost_range'][1])
        norm_comp_costs = state['comp_costs'] / max_comp_cost
        node_features = torch.cat([
            state['task_status'],
            norm_comp_costs.unsqueeze(-1)
        ], dim=-1)

        node_features = self.feature_layernorm(node_features)
        x = self.node_embedder(node_features)
        positions = torch.arange(0, self.n_max, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.positional_embedding(positions).to(x.dtype)

        # --- 2. 构建注意力偏置矩阵 (attn_mask) ---
        comm_costs = state['comm_costs'].unsqueeze(-1).float()
        max_comm_cost = float(cfg.GENERATOR_CONFIG['comm_cost_range'][1])
        norm_comm_costs = comm_costs / max_comm_cost
        edge_bias = self.edge_encoder(norm_comm_costs).permute(0, 3, 1, 2)
        adj = state['adj_matrix'].unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(edge_bias.dtype)

        fill_value = torch.finfo(adj.dtype).min
        attn_bias = torch.full_like(adj, fill_value)
        attn_bias[adj > 0] = edge_bias[adj > 0]
        attn_bias = attn_bias.view(batch_size * self.num_heads, self.n_max, self.n_max)

        # --- 3. 通过图 Transformer 编码器 ---
        padding_mask_bool = ~state['task_mask']
        key_padding_mask = torch.zeros_like(padding_mask_bool, dtype=attn_bias.dtype, device=x.device)
        key_padding_mask.masked_fill_(padding_mask_bool, -torch.inf)

        for layer in self.transformer_layers:
            x = layer(x, attn_mask=attn_bias, key_padding_mask=key_padding_mask)
        task_embeddings = x

        # --- 4. 策略头 ---
        max_proc_speed = float(cfg.GENERATOR_CONFIG['proc_speed_range'][1])
        norm_speeds = state['proc_speeds'] / max_proc_speed
        norm_avail_times = state['processor_available_times']
        proc_features = torch.stack([norm_speeds, norm_avail_times], dim=-1).float()
        proc_features = self.proc_layernorm(proc_features)
        proc_embeddings = self.proc_encoder(proc_features)

        task_embed_expanded = task_embeddings.unsqueeze(2).expand(-1, self.n_max, self.m_max, -1)
        proc_embed_expanded = proc_embeddings.unsqueeze(1).expand(-1, self.n_max, self.m_max, -1)
        combined_features = torch.cat([task_embed_expanded, proc_embed_expanded], dim=-1)
        policy_logits = self.policy_head(combined_features).view(batch_size, -1)

        # --- 5. 价值头 ---
        task_mask_expanded = state['task_mask'].unsqueeze(-1).expand_as(task_embeddings)
        masked_embeddings = task_embeddings * task_mask_expanded
        summed_embeddings = masked_embeddings.sum(dim=1)
        num_tasks = state['task_mask'].sum(dim=1, keepdim=True).clamp(min=1)
        graph_embedding = summed_embeddings / num_tasks
        value = self.value_head(graph_embedding)

        # ============================ [ 代码修改 2/2 - 修改 ] ============================
        # [原因] 需要将可学习的对数方差参数传递给损失函数。
        # [方案] 修改 forward 方法的返回值，额外返回 log_var_policy 和 log_var_value。
        return policy_logits, value, self.log_var_policy, self.log_var_value
        # ========================= [ 修改结束 ] =========================