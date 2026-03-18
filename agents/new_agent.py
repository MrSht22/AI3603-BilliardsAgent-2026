import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent

import copy

# ============ 配置类：方便对照实验 ============
class VariantConfig:
    """变种生成策略配置"""
    BASELINE = {
        'v0_delta_up': 0.5,
        'v0_delta_down': 0.3,
        'phi_delta': 0.3,
        'use_spin_variants': False,  # 是否使用旋转变种
        'n_variants': 5
    }
    
    AGGRESSIVE = {
        'v0_delta_up': 0.8,
        'v0_delta_down': 0.5,
        'phi_delta': 0.5,
        'use_spin_variants': True,
        'n_variants': 7
    }
    
    CONSERVATIVE = {
        'v0_delta_up': 0.3,
        'v0_delta_down': 0.2,
        'phi_delta': 0.2,
        'use_spin_variants': True,
        'n_variants': 7
    }

class ScoringConfig:
    """评分函数配置"""
    BASELINE = {
        'cue_pocketed': -100,
        'eight_illegal': -150,
        'eight_legal': 100,
        'own_ball': 50,
        'enemy_ball': -20,
        'first_hit_foul': -30,
        'legal_no_pot': 5
    }
    
    HIGH_REWARD_LOW_PENALTY = {
        'cue_pocketed': -120,
        'eight_illegal': -400,
        'eight_legal': 250,
        'own_ball': 100,
        'enemy_ball': -50,
        'first_hit_foul': -50,
        'legal_no_pot': -20
    }
    
    LOW_REWARD_HIGH_PENALTY = {
        'cue_pocketed': -150,
        'eight_illegal': -500,
        'eight_legal': 120,
        'own_ball': 70,
        'enemy_ball': -50,
        'first_hit_foul': -40,
        'legal_no_pot': 3
    }

# ============================================

class NewAgent(Agent):
    """物理计算 + 噪声感知 + 多变种评估的混合 Agent"""
    
    def __init__(self, 
                 variant_config='BASELINE',    # 'BASELINE', 'AGGRESSIVE', 'CONSERVATIVE'
                 scoring_config='BASELINE',    # 'BASELINE', 'HIGH_REWARD_LOW_PENALTY', 'LOW_REWARD_HIGH_PENALTY'
                 n_simulations=8):
        super().__init__()
        self.ball_radius = 0.028575
        self.pocket_radius = 0.05
        
        # 加载配置
        self.variant_cfg = getattr(VariantConfig, variant_config)
        self.scoring_cfg = getattr(ScoringConfig, scoring_config)
        
        # 噪声参数（与环境保持一致）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.15,
            'theta': 0.1,
            'a': 0.005,
            'b': 0.005
        }
        
        # 每个候选方案评估次数
        self.n_simulations_per_candidate = n_simulations
        
        print(f"[NewAgent] 配置: 变种={variant_config}, 评分={scoring_config}, 模拟次数={n_simulations}")
        
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法：物理计算 + 噪声鲁棒性评估"""
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        
        # 获取白球位置
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return self._random_action()
        cue_pos = cue_ball.state.rvw[0][:2]
        
        # 确定目标球列表
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_targets) == 0:
            remaining_targets = ['8']
        
        # 保存当前状态（用于模拟）
        saved_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 获取所有袋口位置
        pockets = self._get_pocket_positions(table)
        
        # 获取所有未进袋的球
        all_balls_pos = {}
        for bid, ball in balls.items():
            if ball.state.s != 4:
                all_balls_pos[bid] = ball.state.rvw[0][:2]
        
        # ========== 第一阶段：物理计算生成候选方案 ==========
        candidates = []
        
        for target_id in remaining_targets:
            if balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0][:2]
            
            for pocket_id, pocket_pos in pockets.items():
                score, base_params = self._evaluate_shot(
                    cue_pos, target_pos, pocket_pos,
                    target_id, all_balls_pos, table
                )
                
                if score > 0 and base_params is not None:
                    # 为每个基础方案生成多个变种（应对噪声）
                    variants = self._generate_variants(base_params)
                    for variant in variants:
                        candidates.append({
                            'params': variant,
                            'base_score': score,
                            'target_id': target_id,
                            'pocket_id': pocket_id
                        })
        
        if len(candidates) == 0:
            return self._play_safety(cue_pos, remaining_targets, balls, table, all_balls_pos, pockets)
        
        # ========== 第二阶段：噪声感知评估 ==========
        best_candidate = None
        best_avg_reward = -float('inf')
        
        for candidate in candidates[:15]:  # 只评估前15个候选
            avg_reward = self._evaluate_with_noise(
                candidate['params'], 
                saved_balls, 
                table, 
                my_targets, 
                remaining_targets
            )
            
            # 综合物理得分和模拟得分
            final_score = candidate['base_score'] * 0.3 + avg_reward * 0.7
            
            if final_score > best_avg_reward:
                best_avg_reward = final_score
                best_candidate = candidate
        
        if best_candidate is not None:
            print(f"[NewAgent] 选择动作: V0={best_candidate['params']['V0']:.2f}, "
                  f"phi={best_candidate['params']['phi']:.1f}°, "
                  f"预期得分={best_avg_reward:.1f}")
            return best_candidate['params']
        
        return self._play_safety(cue_pos, remaining_targets, balls, table, all_balls_pos, pockets)
    
    def _generate_variants(self, base_params):
        """为基础参数生成多个变种（应对噪声）- 使用配置参数"""
        cfg = self.variant_cfg
        variants = [base_params]  # 原始参数
        
        # 变种1：力度增加（应对摩擦损耗）
        variants.append({
            **base_params,
            'V0': min(base_params['V0'] + cfg['v0_delta_up'], 7.0)
        })
        
        # 变种2：力度减少（更保守）
        variants.append({
            **base_params,
            'V0': max(base_params['V0'] - cfg['v0_delta_down'], 1.0)
        })
        
        # 变种3：角度正向偏移
        variants.append({
            **base_params,
            'phi': (base_params['phi'] + cfg['phi_delta']) % 360
        })
        
        # 变种4：角度负向偏移
        variants.append({
            **base_params,
            'phi': (base_params['phi'] - cfg['phi_delta']) % 360
        })
        
        # 变种5：组合（力度+角度）
        if cfg['n_variants'] >= 5:
            variants.append({
                **base_params,
                'V0': min(base_params['V0'] + cfg['v0_delta_up'], 7.0),
                'phi': (base_params['phi'] + cfg['phi_delta']) % 360
            })
        
        # ========== 新增变种（仅在 AGGRESSIVE/CONSERVATIVE 模式下启用） ==========
        if cfg['use_spin_variants'] and cfg['n_variants'] >= 7:
            # 变种6：高杆变种（适合切球）
            variants.append({
                **base_params,
                'V0': min(base_params['V0'] + cfg['v0_delta_up'] * 1.2, 7.5),
                'b': 0.15  # 高杆
            })
            
            # 变种7：低杆变种（适合防守）
            variants.append({
                **base_params,
                'V0': max(base_params['V0'] - cfg['v0_delta_down'], 1.0),
                'b': -0.2  # 低杆
            })
        
        return variants
    
    def _evaluate_with_noise(self, action, saved_balls, table, my_targets, remaining_targets):
        """用带噪声的模拟评估动作的期望得分"""
        rewards = []
        
        for _ in range(self.n_simulations_per_candidate):
            # 恢复球状态
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in saved_balls.items()}
            sim_table = copy.deepcopy(table)
            
            try:
                # 创建系统
                cue = pt.Cue(cue_ball_id='cue')
                system = pt.System(cue=cue, table=sim_table, balls=sim_balls)
                
                # 注入噪声
                noisy_action = self._add_noise(action)
                
                system.cue.set_state(
                    V0=noisy_action['V0'],
                    phi=noisy_action['phi'],
                    theta=noisy_action['theta'],
                    a=noisy_action['a'],
                    b=noisy_action['b'],
                    cue_ball_id='cue'
                )
                
                # 运行模拟
                pt.simulate(system, inplace=True)
                
                # 评估结果
                reward = self._analyze_shot_result(system, saved_balls, my_targets, remaining_targets)
                rewards.append(reward)
                
            except Exception as e:
                rewards.append(self.scoring_cfg['cue_pocketed'])  # 模拟失败按白球进袋处理
        
        # 返回平均奖励
        avg_reward = np.mean(rewards)
        return avg_reward
    
    def _add_noise(self, action):
        """为动作添加高斯噪声（模拟真实环境）"""
        return {
            'V0': np.clip(
                action['V0'] + np.random.normal(0, self.noise_std['V0']),
                0.5, 8.0
            ),
            'phi': (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360,
            'theta': np.clip(
                action['theta'] + np.random.normal(0, self.noise_std['theta']),
                0, 90
            ),
            'a': np.clip(
                action['a'] + np.random.normal(0, self.noise_std['a']),
                -0.5, 0.5
            ),
            'b': np.clip(
                action['b'] + np.random.normal(0, self.noise_std['b']),
                -0.5, 0.5
            )
        }
    
    def _analyze_shot_result(self, system, last_state, my_targets, remaining_targets):
        """分析击球结果 - 使用配置参数"""
        cfg = self.scoring_cfg
        score = 0
        
        # 检查白球状态
        cue_ball = system.balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:  # 白球进袋
            return cfg['cue_pocketed']
        
        # 检查黑8
        ball_8 = system.balls.get('8')
        eight_pocketed = (ball_8 is not None and ball_8.state.s == 4)
        
        # 统计进球
        new_pocketed = [
            bid for bid, ball in system.balls.items()
            if ball.state.s == 4 and last_state[bid].state.s != 4
        ]
        
        own_pocketed = [bid for bid in new_pocketed if bid in my_targets]
        
        # 对方球
        opponent_targets = (
            ['1','2','3','4','5','6','7'] if '9' in my_targets 
            else ['9','10','11','12','13','14','15']
        )
        enemy_pocketed = [bid for bid in new_pocketed if bid in opponent_targets]
        
        # 如果还有目标球但黑8进了，判负
        remaining_count = sum(
            1 for bid in remaining_targets 
            if bid != '8' and system.balls[bid].state.s != 4
        )
        
        if eight_pocketed and remaining_count > 0:
            return cfg['eight_illegal']
        
        # 合法打进黑8
        if eight_pocketed and remaining_count == 0 and '8' in remaining_targets:
            return cfg['eight_legal']
        
        # 检查首球犯规
        first_hit_legal = self._check_first_hit(system, my_targets, remaining_targets)
        if not first_hit_legal:
            score += cfg['first_hit_foul']
        
        # 统计得分
        score += len(own_pocketed) * cfg['own_ball']
        score += len(enemy_pocketed) * cfg['enemy_ball']  # 注意：enemy_ball 是负数
        
        # 合法无进球
        if score == 0 and first_hit_legal:
            score = cfg['legal_no_pot']
        
        return score
    
    def _check_first_hit(self, system, my_targets, remaining_targets):
        """检查首次碰撞是否合法"""
        try:
            valid_balls = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
            for event in system.events:
                ids = list(event.ids) if hasattr(event, 'ids') else []
                if 'cue' in ids:
                    other_ids = [i for i in ids if i != 'cue' and i in valid_balls]
                    if other_ids:
                        first_contact = other_ids[0]
                        return first_contact in remaining_targets
        except:
            pass
        return True
    
    def _get_pocket_positions(self, table):
        """获取所有袋口的位置"""
        pockets = {}
        for pocket_id, pocket in table.pockets.items():
            pockets[pocket_id] = np.array(pocket.center[:2])
        return pockets
    
    def _evaluate_shot(self, cue_pos, target_pos, pocket_pos, target_id, all_balls_pos, table):
        """评估一个击球方案（保留原逻辑，但调整力度公式）"""
        # 1. 计算目标球到袋口的方向
        target_to_pocket = pocket_pos - target_pos
        dist_to_pocket = np.linalg.norm(target_to_pocket)
        
        if dist_to_pocket < 0.01:
            return -1, None
            
        target_to_pocket_unit = target_to_pocket / dist_to_pocket
        
        # 2. 计算瞄准点
        aim_point = target_pos - target_to_pocket_unit * (2 * self.ball_radius)
        
        # 3. 计算白球到瞄准点的方向
        cue_to_aim = aim_point - cue_pos
        dist_cue_to_aim = np.linalg.norm(cue_to_aim)
        
        if dist_cue_to_aim < 0.01:
            return -1, None
            
        cue_to_aim_unit = cue_to_aim / dist_cue_to_aim
        
        # 4. 计算击球角度
        phi = math.degrees(math.atan2(cue_to_aim_unit[1], cue_to_aim_unit[0]))
        if phi < 0:
            phi += 360
        
        # 5. 检查路径遮挡
        if self._is_path_blocked(cue_pos, target_pos, target_id, all_balls_pos):
            return -100, None
        
        if self._is_path_blocked(target_pos, pocket_pos, target_id, all_balls_pos, exclude_cue=True):
            return -100, None
        
        # 6. 计算切球角度
        dot_product = np.dot(cue_to_aim_unit, target_to_pocket_unit)
        cut_angle = math.degrees(math.acos(np.clip(dot_product, -1, 1)))
        
        if cut_angle > 70:
            return -50, None
        
        # 7. 评分（更保守的评分）
        score = 100
        score -= cut_angle * 1.2  # 增加切角惩罚
        score -= dist_to_pocket * 18
        score -= dist_cue_to_aim * 10
        
        # 8. 计算击球力度（使用 Pro 的公式）
        total_dist = dist_cue_to_aim + dist_to_pocket
        V0 = 1.5 + total_dist * 1.5  # 更保守的力度
        
        # 切角大时增加力度
        angle_factor = 1 + (cut_angle / 90) * 0.3
        V0 = V0 * angle_factor
        V0 = float(np.clip(V0, 1.0, 6.0))
        
        shot_params = {
            'V0': V0,
            'phi': phi,
            'theta': 3,  # 略微俯击
            'a': 0,
            'b': 0.05 if cut_angle > 40 else 0  # 大角度时加高杆
        }
        
        return score, shot_params
    
    def _is_path_blocked(self, start_pos, end_pos, target_id, all_balls_pos, exclude_cue=False):
        """检查路径是否被遮挡"""
        path_vec = end_pos - start_pos
        path_length = np.linalg.norm(path_vec)
        
        if path_length < 0.01:
            return False
            
        path_unit = path_vec / path_length
        
        for bid, ball_pos in all_balls_pos.items():
            if bid == target_id:
                continue
            if bid == 'cue' and not exclude_cue:
                continue
            if exclude_cue and bid == 'cue':
                continue
                
            ball_to_start = ball_pos - start_pos
            proj_length = np.dot(ball_to_start, path_unit)
            
            if proj_length < self.ball_radius or proj_length > path_length - self.ball_radius:
                continue
            
            perp_dist = np.linalg.norm(ball_to_start - proj_length * path_unit)
            
            if perp_dist < 2 * self.ball_radius + 0.005:
                return True
        
        return False
    
    def _play_safety(self, cue_pos, targets, balls, table, all_balls_pos, pockets):
        """改进的安全球策略"""
        for target_id in targets:
            if balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0][:2]
            
            if self._is_path_blocked(cue_pos, target_pos, target_id, all_balls_pos):
                continue
            
            cue_to_target = target_pos - cue_pos
            dist = np.linalg.norm(cue_to_target)
            
            if dist < 0.05:
                continue
            
            cue_to_target_unit = cue_to_target / dist
            phi = math.degrees(math.atan2(cue_to_target_unit[1], cue_to_target_unit[0]))
            if phi < 0:
                phi += 360
            
            return {
                'V0': min(2.5, 1.2 + dist * 1.2),
                'phi': phi,
                'theta': 5,
                'a': 0,
                'b': -0.15  # 低杆，让白球停住
            }
        
        return self._random_action()