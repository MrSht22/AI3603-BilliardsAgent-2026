# Pool 环境 SAC 智能体训练说明

**注意！！！本仓库文件，即train_sac.py，并没有使用到最终模型中，只是应PROJECT_GUIDE中包含过往方案尝试的要求一起提交。没有使用的原因在报告中已经完整说明。如需检验，请配合agent文件夹中的oldagent.py一起使用。相应地，evaluate.py中的import部分也需要稍作更改。**

本仓库包含 `train_sac.py` 脚本，用于在中文八球台球模拟环境中训练基于 Soft Actor-Critic (SAC) 的连续动作智能体。SAC 智能体学习选择连续击球动作，同时考虑物理动态、执行噪声和规则约束。

---

## 环境要求

- Python 3.8 及以上
- PyTorch 2.0 及以上
- NumPy
- `poolenv` 模块（自定义台球模拟环境）
- 可选：带 CUDA 的 GPU（加速训练）

---

## 训练脚本说明

**文件：** `train_sac.py`  

脚本实现了 SAC 智能体在单步 episodic 环境下的训练，每个 episode 对应一杆击球。主要功能包括：

- 可选的基于 `BasicAgent` 的重放缓冲预热（目前被注释掉）
- 从 `PoolEnv` 获取状态并编码为神经网络输入
- SAC 动作选择，并将 [-1,1] 范围的输出映射到物理击球参数
- 根据击球结果计算奖励
- 将 transition 存入重放缓冲并更新 SAC 网络
- 每 `LOG_EVERY` 个 episode 打印统计信息
- 训练结束后保存模型为 `sac_model.pth`

---

## 使用方法

运行训练：

```bash
python train_sac.py
```

训练过程中每 20 个 episode 会输出统计信息，包括：

- 最近 100 个 episode 的平均奖励
- 命中目标球的次数
- 未接触任何球（NO_HIT）次数
- 首球犯规次数
- 白球进袋次数
- 平均动作值（V0, φ, θ）
- Actor / Critic 损失和熵系数 α

训练结束后，模型参数会保存为：

```
sac_model.pth
```

## 超参数说明

| 参数                       | 值         | 说明 |
|----------------------------|------------|------|
| MAX_EPISODES               | 5000       | 总训练轮数 |
| MAX_STEPS_PER_EPISODE      | 1          | 每轮 episode 步数（一杆为一轮） |
| BATCH_SIZE                 | 256        | SAC 更新时的 minibatch 大小 |
| WARMUP_STEPS               | 3000       | 重放缓冲预热步数（可选） |
| UPDATE_EVERY               | 1          | 每步更新 SAC 网络次数 |
| LOG_EVERY                  | 20         | 日志打印间隔（episode） |
| SEED                       | 42         | 随机种子，保证可复现 |
| DEVICE                     | cpu/cuda   | 训练设备 |

**SAC 网络相关超参数（在 `NewAgent` 内部定义）**：

- Actor / Critic 学习率：`3e-4`
- 折扣因子 γ：`0.99`
- Polyak 平滑系数 τ：`0.005`
- 重放缓冲容量：200,000 条 transition
- 目标熵：`-action_dim`（自动调节 α）

---

## 注意事项

- **Replay Buffer 预热**：使用 `BasicAgent` 预热缓冲可稳定早期学习，但非必需，且 `BasicAgent` 本身效果不佳时不仅起不到作用还会严重拖慢训练速度。
- **动作归一化**：SAC 输出为 [-1,1]，会在内部映射到物理击球参数（V0、φ、θ、杆偏移 a,b）。
- **奖励裁剪**：训练中奖励被裁剪到 [-10,10] 以保证数值稳定。
- **单步 Episode**：每个 episode 对应一次击球；长远规划通过奖励设计和重复 episode 实现。
- **日志输出**：统计包括命中率、犯规、白球进袋率和平均动作，方便监控学习进度。
- **模型保存**：训练结束后保存 Actor、Critic、Target Critic 和 log_alpha 参数。

---
