import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent

import copy

class NewAgent(Agent):
    """
    基于 SAC 的连续动作智能体
    """
    
    # =========================
    # 内部：Replay Buffer
    # =========================
    class ReplayBuffer:
        def __init__(self, capacity, state_dim, action_dim):
            self.capacity = capacity
            self.ptr = 0
            self.size = 0

            self.s = np.zeros((capacity, state_dim), dtype=np.float32)
            self.a = np.zeros((capacity, action_dim), dtype=np.float32)
            self.r = np.zeros((capacity, 1), dtype=np.float32)
            self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
            self.d = np.zeros((capacity, 1), dtype=np.float32)

        def store(self, s, a, r, s2, d):
            self.s[self.ptr] = s
            self.a[self.ptr] = a
            self.r[self.ptr] = r
            self.s2[self.ptr] = s2
            self.d[self.ptr] = d

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        def sample(self, batch_size):
            idx = np.random.randint(0, self.size, size=batch_size)
            return (
                torch.tensor(self.s[idx], device=self.device),
                torch.tensor(self.a[idx], device=self.device),
                torch.tensor(self.r[idx], device=self.device),
                torch.tensor(self.s2[idx], device=self.device),
                torch.tensor(self.d[idx], device=self.device),
            )


    # =========================
    # 内部：Actor
    # =========================
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.mu = nn.Linear(256, action_dim)
            self.log_std = nn.Linear(256, action_dim)

        def forward(self, s):
            x = F.relu(self.fc1(s))
            x = F.relu(self.fc2(x))
            mu = self.mu(x)
            log_std = torch.clamp(self.log_std(x), -20, 2)
            return mu, log_std

        def sample(self, s):
            mu, log_std = self(s)
            std = log_std.exp()

            eps = torch.randn_like(std)
            z = mu + eps * std               # reparameterization
            a = torch.tanh(z)                # ✅ 关键：动作 squash 到 [-1,1]

            # log_prob with tanh correction
            logp = -0.5 * (
                ((z - mu) / std) ** 2
                + 2 * log_std
                + math.log(2 * math.pi)
            )
            logp = logp.sum(dim=-1, keepdim=True)

            # tanh Jacobian 修正
            logp -= torch.sum(
                torch.log(1 - a.pow(2) + 1e-6),
                dim=-1,
                keepdim=True
            )

            return a, logp


    # =========================
    # 内部：Critic
    # =========================
    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, s, a):
            x = torch.cat([s, a], dim=-1)
            return self.q1(x), self.q2(x)

    # =========================
    # 初始化
    # =========================
    def __init__(self, state_dim=23, action_dim=5, max_targets=3, device='cpu'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_targets = max_targets
        self.device = device

        self.actor = self.Actor(state_dim, action_dim).to(device)
        self.critic = self.Critic(state_dim, action_dim).to(device)
        self.critic_target = self.Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.target_entropy = -action_dim

        self.buffer = self.ReplayBuffer(
            capacity=200_000,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # 记录最后一次损失，用于日志
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

        # 加载训练好的模型（如果存在）
        self.model_path = 'sac_model.pth'
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha.data = checkpoint['log_alpha']
        print("Model loaded from", path)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # =========================
    # encode_state
    # =========================

    def encode_state(self, balls, my_targets, table):
        """
        将 get_observation() 返回的 (balls, my_targets, table)
        编码为固定维度的数值向量 state

        返回：
            np.ndarray, shape = (state_dim,)
        """
        state = []

        # ========= 1. 白球（Cue Ball） =========
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0][:2]  # (x, y)

        # 归一化到 [0, 1]
        cue_x = cue_pos[0] / table.w
        cue_y = cue_pos[1] / table.l

        state.extend([cue_x, cue_y])

        # ========= 2. 预取所有球袋位置 =========
        pocket_centers = [
            pocket.center[:2] for pocket in table.pockets.values()
        ]

        # ========= 3. 目标球编码 =========
        used_targets = my_targets[:self.max_targets]

        for tid in used_targets:
            ball = balls[tid]

            # --- 3.1 已进袋 ---
            if ball.state.s == 4:
                # dx, dy, dist, angle, target_to_pocket_dist, pocket_angle, mask
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                continue

            # --- 3.2 目标球相对白球几何 ---
            pos = ball.state.rvw[0][:2]
            dx = (pos[0] - cue_pos[0]) / table.w
            dy = (pos[1] - cue_pos[1]) / table.l
            dist = math.sqrt(dx * dx + dy * dy)

            angle = math.atan2(dy, dx) / math.pi  # 归一化到 [-1, 1]

            # --- 3.3 最近球袋 ---
            min_pocket_dist = float('inf')
            best_pocket_vec = None

            for pc in pocket_centers:
                vec = np.array(pc) - pos
                d = np.linalg.norm(vec)
                if d < min_pocket_dist:
                    min_pocket_dist = d
                    best_pocket_vec = vec

            pocket_dx = best_pocket_vec[0] / table.w
            pocket_dy = best_pocket_vec[1] / table.l
            pocket_dist = math.sqrt(pocket_dx ** 2 + pocket_dy ** 2)

            pocket_angle = math.atan2(pocket_dy, pocket_dx) / math.pi

            # mask = 0 表示球还在
            state.extend([
                dx, dy, dist,
                angle,
                pocket_dist,
                pocket_angle,
                0.0
            ])

        # ========= 4. 不足目标球数量时补零 =========
        missing = self.max_targets - len(used_targets)
        for _ in range(missing):
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        assert len(state) == self.state_dim

        return np.array(state, dtype=np.float32)
    
    # =============================
    # SAC select action
    # =============================
    
    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mu, log_std = self.actor(state)
            if deterministic:
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(state)

        return action.squeeze(0).numpy()
    
    # =============================
    # SAC action → 环境 action
    # =============================
    def action_postprocess(self, a: np.ndarray) -> dict:
        """
        将 SAC 输出的 [-1,1]^5 动作
        映射为 take_shot 所需的物理动作

        输入：
            a: np.ndarray, shape=(5,), 范围 [-1,1]

        输出：
            dict: 可直接传给 take_shot
        """

        # --- 1. 初速度 ---
        # [-1,1] → [0.5, 8.0]
        V0 = (a[0] + 1.0) * 0.5 * (8.0 - 0.5) + 0.5

        # --- 2. 水平角 phi（度）---
        # [-1,1] → [0,360)
        phi = (a[1] + 1.0) * 0.5 * 360.0
        phi = phi % 360.0

        # --- 3. 抬杆角 theta（度）---
        # [-1,1] → [0,90]
        theta = (a[2] + 1.0) * 0.5 * 90.0

        # --- 4. 杆头偏移 ---
        # [-1,1] → [-0.5, 0.5]
        a_offset = a[3] * 0.5
        b_offset = a[4] * 0.5

        return {
            "V0": float(V0),
            "phi": float(phi),
            "theta": float(theta),
            "a": float(a_offset),
            "b": float(b_offset),
        }
    
    # =============================
    # SAC 训练相关
    # =============================
    def store_transition(self, s, a, r, s2, done):
        self.buffer.store(s, a, r, s2, done)

    # =============================
    # SAC 更新网络
    # =============================
    def update(self, batch_size=256):
        if self.buffer.size < batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(batch_size)

        r = r / 50.0
        r = torch.clamp(r, -5.0, 5.0)


        # -------- critic --------
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            y = r + self.gamma * (1 - d) * q_t

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # -------- actor --------
        a_new, logp = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, a_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -------- alpha --------
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        '''self.alpha = 0.05'''

        # -------- target --------
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        # 记录损失
        self.last_critic_loss = critic_loss.item()
        self.last_actor_loss = actor_loss.item()


    # =============================
    # 计算 reward（SAC 专用）
    # =============================
    def compute_sac_reward(self, shot_result: dict, my_targets: list[str]):

        reward = 0.0

        # =========================
        # 1. 是否命中目标球（关键 shaping）
        # =========================
        if shot_result.get("NO_HIT", False):
            reward -= 1.0
        elif not shot_result.get("FOUL_FIRST_HIT", False):
            # 合法首球命中目标球
            reward += 0.3

        # =========================
        # 2. 进球奖励
        # =========================
        reward += 0.5 * len(shot_result.get("ME_INTO_POCKET", []))
        reward -= 0.3 * len(shot_result.get("ENEMY_INTO_POCKET", []))

        # =========================
        # 3. 白球 / 黑8
        # =========================
        if shot_result.get("WHITE_BALL_INTO_POCKET", False):
            reward -= 1.0

        if shot_result.get("BLACK_BALL_INTO_POCKET", False):
            reward += 1.0 if my_targets == ['8'] else -1.0

        # =========================
        # 4. 碰库规则
        # =========================
        if shot_result.get("NO_POCKET_NO_RAIL", False):
            reward -= 1.0

        # =========================
        # 5. 存活 shaping（轻微）
        # =========================
        if reward > -0.5:
            reward += 0.2

        return reward



    # =============================
    # 决策接口
    # =============================
    def decision(self, balls, my_targets, table):
        # 1. 编码状态
        state = self.encode_state(balls, my_targets, table)

        # 2. SAC 选择动作（[-1,1]^5），评估时使用确定性策略
        action_norm = self.select_action(state, deterministic=True)  # np.ndarray

        # 3. 映射为物理动作
        action = self.action_postprocess(action_norm)

        return action
