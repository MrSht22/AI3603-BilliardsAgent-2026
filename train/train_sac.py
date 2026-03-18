import numpy as np
import torch
import random
from collections import deque, defaultdict
from poolenv import PoolEnv
from agent import NewAgent, BasicAgent

# =============================
# 超参数
# =============================
MAX_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1
BATCH_SIZE = 256
WARMUP_STEPS = 3000
UPDATE_EVERY = 1

LOG_EVERY = 20          # ★ 日志频率
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = PoolEnv()
agent = NewAgent(device=DEVICE)
basic_agent = BasicAgent()

# =============================
# Warmup
# =============================
'''print("Warmup replay buffer with BasicAgent...")
env.reset(target_ball='solid')
obs = env.get_observation()

for step in range(WARMUP_STEPS):
    balls, my_targets, table = obs
    action_dict = basic_agent.decision(balls, my_targets, table)
    result = env.take_shot(action_dict)

    s = agent.encode_state(balls, my_targets, table)
    next_obs = env.get_observation()
    s2 = agent.encode_state(*next_obs)

    reward = agent.compute_sac_reward(result, my_targets)
    reward = np.clip(reward, -10.0, 10.0)

    agent.store_transition(
        s,
        np.zeros(5, dtype=np.float32),
        reward,
        s2,
        result.get("BLACK_BALL_INTO_POCKET", False)
    )

    if result.get("BLACK_BALL_INTO_POCKET", False):
        env.reset(target_ball='solid')

    obs = env.get_observation()

print("Warmup finished.")'''

# =============================
# 训练统计缓存
# =============================
reward_history = deque(maxlen=100)

stat_buffer = defaultdict(list)

# =============================
# Training Loop
# =============================
for episode in range(1, MAX_EPISODES + 1):

    env.reset(target_ball='solid')
    obs = env.get_observation()
    episode_reward = 0.0

    # episode-level 统计
    ep_stats = {
        "hit_target": 0,
        "no_hit": 0,
        "foul": 0,
        "white_pocket": 0,
    }

    for step in range(MAX_STEPS_PER_EPISODE):

        balls, my_targets, table = obs
        state = agent.encode_state(balls, my_targets, table)

        action_norm = agent.select_action(state)
        action = agent.action_postprocess(action_norm)

        result = env.take_shot(action)

        reward = agent.compute_sac_reward(result, my_targets)
        reward = np.clip(reward, -10.0, 10.0)
        episode_reward += reward

        # ===== 行为统计 =====
        if result.get("NO_HIT", False):
            ep_stats["no_hit"] += 1
        elif not result.get("FOUL_FIRST_HIT", False):
            ep_stats["hit_target"] += 1

        if result.get("FOUL_FIRST_HIT", False):
            ep_stats["foul"] += 1

        if result.get("WHITE_BALL_INTO_POCKET", False):
            ep_stats["white_pocket"] += 1

        next_obs = env.get_observation()
        next_state = agent.encode_state(*next_obs)

        done = (
            result.get("NO_HIT", False) or
            result.get("BLACK_BALL_INTO_POCKET", False) or
            result.get("WHITE_BALL_INTO_POCKET", False)
        )

        agent.store_transition(
            state,
            action_norm,
            reward,
            next_state,
            done
        )

        if agent.buffer.size > BATCH_SIZE:
            agent.update()

        obs = next_obs
        if done:
            break

    # =============================
    # episode 统计汇总
    # =============================
    reward_history.append(episode_reward)

    stat_buffer["reward"].append(episode_reward)
    stat_buffer["hit_rate"].append(ep_stats["hit_target"])
    stat_buffer["no_hit_rate"].append(ep_stats["no_hit"])
    stat_buffer["foul_rate"].append(ep_stats["foul"])
    stat_buffer["white_rate"].append(ep_stats["white_pocket"])

    stat_buffer["v0_mean"].append(action_norm[0])
    stat_buffer["phi_mean"].append(action_norm[1])
    stat_buffer["theta_mean"].append(action_norm[2])

    # =============================
    # 日志输出
    # =============================
    if episode % LOG_EVERY == 0:

        def avg(x): return np.mean(x) if x else 0.0

        print(
            f"\n[Episode {episode:5d}]"
            f"\n  Reward(avg100): {avg(reward_history):6.2f}"
            f"\n  Hit target   : {avg(stat_buffer['hit_rate']):.2f}"
            f"\n  NO_HIT       : {avg(stat_buffer['no_hit_rate']):.2f}"
            f"\n  Foul         : {avg(stat_buffer['foul_rate']):.2f}"
            f"\n  White pocket : {avg(stat_buffer['white_rate']):.2f}"
            f"\n  Action(mean) : "
            f"V0={avg(stat_buffer['v0_mean']):+.2f}, "
            f"phi={avg(stat_buffer['phi_mean']):+.2f}, "
            f"theta={avg(stat_buffer['theta_mean']):+.2f}"
            f"\n  SAC: "
            f"actor_loss={agent.last_actor_loss:.4f}, "
            f"critic_loss={agent.last_critic_loss:.4f}, "
            f"alpha={agent.alpha:.4f}"
        )

        stat_buffer.clear()

# =============================
# 保存模型
# =============================
torch.save({
    'actor': agent.actor.state_dict(),
    'critic': agent.critic.state_dict(),
    'critic_target': agent.critic_target.state_dict(),
    'log_alpha': agent.log_alpha,
}, 'sac_model.pth')

print("Model saved to sac_model.pth")
