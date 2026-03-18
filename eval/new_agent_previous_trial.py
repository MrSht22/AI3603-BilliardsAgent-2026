import math
import copy
import signal
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent

class NewAgentPre(Agent):
    """自定义 Agent 模板（待学生实现）"""
    
    def __init__(self):
        pass
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            observation: (balls, my_targets, table)
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        return self._random_action()

class NewAgent1(Agent):
    """基于物理计算的精确击球 Agent"""
    
    def __init__(self):
        super().__init__()
        # 球的半径（标准台球约为 0.028575 米）
        self.ball_radius = 0.028575
        # 袋口有效半径（略大于球半径）
        self.pocket_radius = 0.05
        
    def decision(self, balls=None, my_targets=None, table=None):
        """基于物理计算的决策方法
        
        核心流程：
        1. 遍历所有目标球和袋口组合
        2. 计算每个组合的进球可行性（是否有遮挡、角度是否合理）
        3. 选择最优组合，计算精确的击球参数
        """
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        
        # 获取白球位置
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return self._random_action()
        cue_pos = cue_ball.state.rvw[0][:2]  # 只取 x, y
        
        # 确定目标球列表（如果目标球都进了，打黑8）
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_targets) == 0:
            remaining_targets = ['8']
        
        # 获取所有袋口位置
        pockets = self._get_pocket_positions(table)
        
        # 获取所有未进袋的球（用于遮挡检测）
        all_balls_pos = {}
        for bid, ball in balls.items():
            if ball.state.s != 4:  # 未进袋
                all_balls_pos[bid] = ball.state.rvw[0][:2]
        
        # 评估所有可能的击球方案
        best_shot = None
        best_score = -float('inf')
        
        for target_id in remaining_targets:
            if balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0][:2]
            
            for pocket_id, pocket_pos in pockets.items():
                # 评估这个球-袋组合
                score, shot_params = self._evaluate_shot(
                    cue_pos, target_pos, pocket_pos, 
                    target_id, all_balls_pos, table
                )
                
                if score > best_score:
                    best_score = score
                    best_shot = shot_params
        
        if best_shot is not None and best_score > 0:
            return best_shot
        else:
            # 没有好的进球机会，尝试安全球（推向远端）
            return self._play_safety(cue_pos, remaining_targets, balls, table)
    
    def _get_pocket_positions(self, table):
        """获取所有袋口的位置"""
        pockets = {}
        for pocket_id, pocket in table.pockets.items():
            pockets[pocket_id] = np.array(pocket.center[:2])
        return pockets
    
    def _evaluate_shot(self, cue_pos, target_pos, pocket_pos, target_id, all_balls_pos, table):
        """评估一个击球方案的可行性和得分
        
        返回：(score, shot_params)
        """
        # 1. 计算目标球到袋口的方向
        target_to_pocket = pocket_pos - target_pos
        dist_to_pocket = np.linalg.norm(target_to_pocket)
        
        if dist_to_pocket < 0.01:  # 球已经在袋口
            return -1, None
            
        target_to_pocket_unit = target_to_pocket / dist_to_pocket
        
        # 2. 计算白球需要击打的瞄准点（目标球后方一个球直径的位置）
        aim_point = target_pos - target_to_pocket_unit * (2 * self.ball_radius)
        
        # 3. 计算白球到瞄准点的方向和距离
        cue_to_aim = aim_point - cue_pos
        dist_cue_to_aim = np.linalg.norm(cue_to_aim)
        
        if dist_cue_to_aim < 0.01:
            return -1, None
            
        cue_to_aim_unit = cue_to_aim / dist_cue_to_aim
        
        # 4. 计算击球角度 phi（从 x 轴正方向逆时针）
        phi = math.degrees(math.atan2(cue_to_aim_unit[1], cue_to_aim_unit[0]))
        if phi < 0:
            phi += 360
        
        # 5. 检查路径上是否有遮挡（白球到目标球）
        if self._is_path_blocked(cue_pos, target_pos, target_id, all_balls_pos):
            return -100, None  # 有遮挡，大幅扣分
        
        # 6. 检查目标球到袋口是否有遮挡
        if self._is_path_blocked(target_pos, pocket_pos, target_id, all_balls_pos, exclude_cue=True):
            return -100, None
        
        # 7. 计算切球角度（白球击球方向与进袋方向的夹角）
        # 角度越小越容易进球
        dot_product = np.dot(cue_to_aim_unit, target_to_pocket_unit)
        cut_angle = math.degrees(math.acos(np.clip(dot_product, -1, 1)))
        
        # 切球角度过大（>70度）很难进球
        if cut_angle > 70:
            return -50, None
        
        # 8. 计算得分
        # 得分考虑因素：距离、切球角度、袋口距离
        score = 100
        score -= cut_angle * 1.0  # 切角惩罚
        score -= dist_to_pocket * 20  # 距离惩罚
        score -= dist_cue_to_aim * 10  # 白球到目标距离惩罚
        
        # 9. 计算击球力度
        # 基于距离估算，需要足够的力量让球进袋
        total_dist = dist_cue_to_aim + dist_to_pocket
        V0 = self._calculate_velocity(total_dist, cut_angle)
        
        # 10. 构建击球参数
        shot_params = {
            'V0': V0,
            'phi': phi,
            'theta': 0,  # 平击
            'a': 0,      # 中心击球
            'b': 0
        }
        
        return score, shot_params
    
    def _is_path_blocked(self, start_pos, end_pos, target_id, all_balls_pos, exclude_cue=False):
        """检查从 start 到 end 的路径上是否有其他球遮挡"""
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
                
            # 计算该球到路径的垂直距离
            ball_to_start = ball_pos - start_pos
            
            # 在路径方向上的投影长度
            proj_length = np.dot(ball_to_start, path_unit)
            
            # 只考虑路径范围内的球
            if proj_length < self.ball_radius or proj_length > path_length - self.ball_radius:
                continue
            
            # 垂直距离
            perp_dist = np.linalg.norm(ball_to_start - proj_length * path_unit)
            
            # 如果垂直距离小于两个球的直径，则被遮挡
            if perp_dist < 2 * self.ball_radius + 0.005:  # 加一点余量
                return True
        
        return False
    
    def _calculate_velocity(self, total_dist, cut_angle):
        """根据距离和切球角度计算合适的击球力度"""
        # 基础速度：根据总距离
        base_v = 1.5 + total_dist * 2.0
        
        # 切角越大需要越大的力量
        angle_factor = 1 + (cut_angle / 90) * 0.5
        
        V0 = base_v * angle_factor
        
        # 限制在合理范围内
        V0 = np.clip(V0, 1.0, 6.0)
        
        return float(V0)
    
    def _play_safety(self, cue_pos, targets, balls, table):
        """没有好的进球机会时，打安全球
        
        策略：尝试将白球打到远离目标球的位置，或者将目标球打到不利位置
        """
        # 简单策略：找一个目标球，轻轻推向库边
        for target_id in targets:
            if balls[target_id].state.s != 4:
                target_pos = balls[target_id].state.rvw[0][:2]
                
                # 计算白球到目标球的方向
                cue_to_target = target_pos - cue_pos
                dist = np.linalg.norm(cue_to_target)
                
                if dist < 0.01:
                    continue
                
                cue_to_target_unit = cue_to_target / dist
                phi = math.degrees(math.atan2(cue_to_target_unit[1], cue_to_target_unit[0]))
                if phi < 0:
                    phi += 360
                
                return {
                    'V0': 2.0,  # 较轻的力度
                    'phi': phi,
                    'theta': 0,
                    'a': 0,
                    'b': 0
                }
        
        # 实在没办法，随机击球
        return self._random_action()

#############################################################
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


class NewAgent2(Agent):
    """物理计算 + 贝叶斯微调的精确击球 Agent"""
    
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        self.pocket_radius = 0.05
        
        # 贝叶斯优化参数
        self.USE_BAYESIAN_REFINEMENT = True
        self.REFINE_ITERATIONS = 15  # 微调迭代次数
        
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法：物理计算 + 贝叶斯微调"""
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
        
        # 获取所有袋口位置
        pockets = self._get_pocket_positions(table)
        
        # 获取所有未进袋的球
        all_balls_pos = {}
        for bid, ball in balls.items():
            if ball.state.s != 4:
                all_balls_pos[bid] = ball.state.rvw[0][:2]
        
        # ========== 第一阶段：物理计算找候选方案 ==========
        candidates = []
        
        for target_id in remaining_targets:
            if balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0][:2]
            
            for pocket_id, pocket_pos in pockets.items():
                score, shot_params = self._evaluate_shot(
                    cue_pos, target_pos, pocket_pos,
                    target_id, all_balls_pos, table
                )
                
                if score > 0 and shot_params is not None:
                    candidates.append({
                        'score': score,
                        'params': shot_params,
                        'target_id': target_id,
                        'pocket_id': pocket_id
                    })
        
        # 按得分排序，取前3个候选
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:3]
        
        # ========== 第二阶段：贝叶斯微调 ==========
        if self.USE_BAYESIAN_REFINEMENT and len(top_candidates) > 0:
            best_shot = self._bayesian_refine(
                top_candidates, balls, my_targets, remaining_targets, table
            )
            if best_shot is not None:
                return best_shot
        
        # 如果有候选但没用贝叶斯，直接返回最佳物理计算结果
        if len(top_candidates) > 0:
            return top_candidates[0]['params']
        
        # 没有好的进球机会，打安全球
        return self._play_safety(cue_pos, remaining_targets, balls, table, all_balls_pos, pockets)
    
    def _bayesian_refine(self, candidates, balls, my_targets, remaining_targets, table):
        """使用贝叶斯优化在候选方案附近微调"""
        try:
            from bayes_opt import BayesianOptimization
            from bayes_opt import SequentialDomainReductionTransformer
        except ImportError:
            return candidates[0]['params'] if candidates else None
        
        # 保存当前球的状态用于模拟
        saved_balls = self._save_balls_state(balls)
        best_result = {'score': -float('inf'), 'params': None}
        
        for candidate in candidates:
            base_params = candidate['params']
            
            # 定义搜索范围（在物理计算结果附近小范围搜索）
            pbounds = {
                'V0': (max(0.5, base_params['V0'] - 1.0), min(8.0, base_params['V0'] + 1.0)),
                'phi': (base_params['phi'] - 8, base_params['phi'] + 8),  # ±8度
                'theta': (0, 15),  # 小角度俯击
                'a': (-0.3, 0.3),
                'b': (-0.3, 0.3),
            }
            
            def objective(V0, phi, theta, a, b):
                # 恢复球状态
                test_balls = self._restore_balls_state(saved_balls)
                
                # 创建模拟系统
                try:
                    system = pt.System(cue=pt.Cue(cue_ball_id='cue'), table=table, balls=test_balls)
                    
                    # 规范化 phi
                    phi_normalized = phi % 360
                    
                    system.cue.set_state(
                        V0=V0,
                        phi=phi_normalized,
                        theta=theta,
                        a=a,
                        b=b,
                        cue_ball_id='cue'
                    )
                    
                    # 运行模拟
                    pt.simulate(system, inplace=True)
                    
                    # 评估结果
                    return self._analyze_shot_result(system, my_targets, remaining_targets)
                    
                except Exception as e:
                    return -100
            
            # 贝叶斯优化
            try:
                optimizer = BayesianOptimization(
                    f=objective,
                    pbounds=pbounds,
                    random_state=42,
                    verbose=0,
                    bounds_transformer=SequentialDomainReductionTransformer()
                )
                
                # 先用物理计算的结果作为初始点
                try:
                    optimizer.probe(
                        params={
                            'V0': base_params['V0'],
                            'phi': base_params['phi'],
                            'theta': base_params['theta'],
                            'a': base_params['a'],
                            'b': base_params['b']
                        },
                        lazy=True
                    )
                except:
                    pass
                
                optimizer.maximize(init_points=5, n_iter=self.REFINE_ITERATIONS)
                
                if optimizer.max['target'] > best_result['score']:
                    best_result['score'] = optimizer.max['target']
                    best_result['params'] = {
                        'V0': optimizer.max['params']['V0'],
                        'phi': optimizer.max['params']['phi'] % 360,
                        'theta': optimizer.max['params']['theta'],
                        'a': optimizer.max['params']['a'],
                        'b': optimizer.max['params']['b']
                    }
            except Exception as e:
                continue
        
        return best_result['params']
    
    def _analyze_shot_result(self, system, my_targets, remaining_targets):
        """分析击球结果，返回得分"""
        score = 0
        
        # 检查白球状态
        cue_ball = system.balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:  # 白球进袋
            return -100
        
        # 检查黑8
        ball_8 = system.balls.get('8')
        eight_pocketed = (ball_8 is not None and ball_8.state.s == 4)
        
        # 如果还有目标球但黑8进了，判负
        remaining_count = sum(1 for bid in remaining_targets if bid != '8' and system.balls[bid].state.s != 4)
        if eight_pocketed and remaining_count > 0:
            return -150
        
        # 白球和黑8同时进袋
        if cue_ball.state.s == 4 and eight_pocketed:
            return -150
        
        # 合法打进黑8（所有目标球已清空）
        if eight_pocketed and remaining_count == 0 and '8' in remaining_targets:
            return 100
        
        # 检查首球犯规（需要分析碰撞历史）
        first_hit_legal = self._check_first_hit(system, my_targets, remaining_targets)
        if not first_hit_legal:
            score -= 30
        
        # 统计进球
        for bid in my_targets:
            ball = system.balls.get(bid)
            if ball is not None and ball.state.s == 4:
                score += 50  # 己方球进袋
        
        # 对方球进袋扣分
        opponent_targets = ['1','2','3','4','5','6','7'] if '9' in my_targets else ['9','10','11','12','13','14','15']
        for bid in opponent_targets:
            ball = system.balls.get(bid)
            if ball is not None and ball.state.s == 4:
                score -= 20
        
        # 如果没有进球，给一个小的基础分
        if score == 0 and first_hit_legal:
            score = 5
        
        return score
    
    def _check_first_hit(self, system, my_targets, remaining_targets):
        """检查首次碰撞是否合法"""
        try:
            for event in system.events:
                if hasattr(event, 'agents') and len(event.agents) >= 2:
                    ids = [agent.id for agent in event.agents]
                    if 'cue' in ids:
                        other_id = ids[1] if ids[0] == 'cue' else ids[0]
                        # 检查是否是己方目标球
                        if other_id in remaining_targets:
                            return True
                        elif other_id == '8' and len([b for b in remaining_targets if b != '8']) == 0:
                            return True  # 目标球清空后可以打8
                        else:
                            return False
        except:
            pass
        return True  # 默认合法
    
    def _save_balls_state(self, balls):
        """保存球状态"""
        return {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
    
    def _restore_balls_state(self, saved_state):
        """恢复球状态"""
        return {bid: copy.deepcopy(ball) for bid, ball in saved_state.items()}
    
    def _get_pocket_positions(self, table):
        """获取所有袋口的位置"""
        pockets = {}
        for pocket_id, pocket in table.pockets.items():
            pockets[pocket_id] = np.array(pocket.center[:2])
        return pockets
    
    def _evaluate_shot(self, cue_pos, target_pos, pocket_pos, target_id, all_balls_pos, table):
        """评估一个击球方案的可行性和得分"""
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
        
        if cut_angle > 75:  # 稍微放宽限制
            return -50, None
        
        # 7. 评分
        score = 100
        score -= cut_angle * 0.8
        score -= dist_to_pocket * 15
        score -= dist_cue_to_aim * 8
        
        # 袋口加分（角袋比边袋容易）
        pocket_bonus = 5 if 'corner' in str(pocket_pos) else 0
        score += pocket_bonus
        
        # 8. 计算击球力度
        total_dist = dist_cue_to_aim + dist_to_pocket
        V0 = self._calculate_velocity(total_dist, cut_angle)
        
        shot_params = {
            'V0': V0,
            'phi': phi,
            'theta': 2,  # 略微俯击，增加控制
            'a': 0,
            'b': 0.1 if cut_angle > 30 else 0  # 大角度时加点高杆
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
    
    def _calculate_velocity(self, total_dist, cut_angle):
        """计算击球力度"""
        base_v = 1.2 + total_dist * 1.8
        angle_factor = 1 + (cut_angle / 90) * 0.4
        V0 = base_v * angle_factor
        return float(np.clip(V0, 1.0, 5.5))
    
    def _play_safety(self, cue_pos, targets, balls, table, all_balls_pos, pockets):
        """改进的安全球策略"""
        # 策略1：尝试打一个防守球，让白球停在安全位置
        best_safety = None
        best_score = -float('inf')
        
        for target_id in targets:
            if balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0][:2]
            
            # 计算到目标球的方向
            cue_to_target = target_pos - cue_pos
            dist = np.linalg.norm(cue_to_target)
            
            if dist < 0.05:
                continue
            
            # 检查路径是否被遮挡
            if self._is_path_blocked(cue_pos, target_pos, target_id, all_balls_pos):
                continue
            
            cue_to_target_unit = cue_to_target / dist
            phi = math.degrees(math.atan2(cue_to_target_unit[1], cue_to_target_unit[0]))
            if phi < 0:
                phi += 360
            
            # 评估这个安全球的质量
            score = 50 - dist * 10  # 近的球更好控制
            
            if score > best_score:
                best_score = score
                best_safety = {
                    'V0': min(2.5, 1.0 + dist * 1.5),  # 轻推
                    'phi': phi,
                    'theta': 5,  # 略微下压
                    'a': 0,
                    'b': -0.2  # 低杆，让白球停住
                }
        
        if best_safety is not None:
            return best_safety
        
        return self._random_action()


#############################################
class NewAgent3(Agent):
    """物理计算 + 噪声感知 + 多变种评估的混合 Agent"""
    
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        self.pocket_radius = 0.05
        
        # 噪声参数（与环境保持一致）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.15,
            'theta': 0.1,
            'a': 0.005,
            'b': 0.005
        }
        
        # 每个候选方案评估次数
        self.n_simulations_per_candidate = 8
        
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
            if (best_avg_reward < 5.0):
                return self._play_safety(cue_pos, remaining_targets, balls, table, all_balls_pos, pockets)
            
            return best_candidate['params']
        
        return self._play_safety(cue_pos, remaining_targets, balls, table, all_balls_pos, pockets)
    
    def _generate_variants(self, base_params):
        """为基础参数生成多个变种（应对噪声）"""
        variants = [base_params]  # 原始参数
        
        # 变种1：力度 +0.5（应对摩擦损耗）
        variants.append({
            **base_params,
            'V0': min(base_params['V0'] + 0.5, 7.0)
        })
        
        # 变种2：力度 -0.3（更保守）
        variants.append({
            **base_params,
            'V0': max(base_params['V0'] - 0.3, 1.0)
        })
        
        # 变种3：角度 +0.3°
        variants.append({
            **base_params,
            'phi': (base_params['phi'] + 0.3) % 360
        })
        
        # 变种4：角度 -0.3°
        variants.append({
            **base_params,
            'phi': (base_params['phi'] - 0.3) % 360
        })
        
        # 变种5：组合（力度+0.5, 角度+0.3）
        variants.append({
            **base_params,
            'V0': min(base_params['V0'] + 0.5, 7.0),
            'phi': (base_params['phi'] + 0.3) % 360
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
                rewards.append(-100)  # 模拟失败
        
        # 返回平均奖励（归一化到 [0, 100]）
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
        """分析击球结果（借鉴 BasicAgentPro 的评分函数）"""
        score = 0
        
        # 检查白球状态
        cue_ball = system.balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:  # 白球进袋
            return -150
        
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
            return -500
        
        # 合法打进黑8
        if eight_pocketed and remaining_count == 0 and '8' in remaining_targets:
            return 300
        
        # 检查首球犯规
        first_hit_legal = self._check_first_hit(system, my_targets, remaining_targets)
        if not first_hit_legal:
            score -= 80
        
        # 统计得分
        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20
        
        # 合法无进球
        if score == 0 and first_hit_legal:
            score = -30
        
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
        """
        安全球策略（快）：从多个“轻推/低杆”为主的方案中选一个最让对手难受的。
        核心目标：
        1) 尽可能制造遮挡（斯诺克/挡线）
        2) 做不到就拉远距离 + 尽量贴库
        """
        print(f"[NewAgent] 采用了安全球策略")
        cue_pos = np.array(cue_pos, dtype=float)

        # 推断对手目标（用于“让对手难打”评估）
        # 注意：这里用 balls 中的状态判断能否推断；推断失败就退化为“让对手难打任何球”
        try:
            # 复用你 decision 里的推断逻辑：my_targets 在这里拿不到，所以退化：
            opponent_targets = ['1','2','3','4','5','6','7','9','10','11','12','13','14','15']
        except Exception:
            opponent_targets = ['1','2','3','4','5','6','7','9','10','11','12','13','14','15']

        opp_remaining = [bid for bid in opponent_targets if balls.get(bid) is not None and balls[bid].state.s != 4]
        if len(opp_remaining) == 0:
            # 对手只剩 8 的情况很少见；兜底
            opp_remaining = ['8'] if balls.get('8') is not None and balls['8'].state.s != 4 else opponent_targets

        # 桌面“坏点”（贴库/角落附近），用来粗略鼓励把白球滚向难受区域
        bad_points = self._get_bad_points(table)

        best_action = None
        best_score = -float('inf')

        # 只对“能直线碰到的目标球”尝试安全球
        for target_id in targets:
            if balls.get(target_id) is None or balls[target_id].state.s == 4:
                continue

            target_pos = np.array(balls[target_id].state.rvw[0][:2], dtype=float)

            # 路径遮挡：cue -> target 必须通
            if self._is_path_blocked(cue_pos, target_pos, target_id, all_balls_pos):
                continue

            cue_to_target = target_pos - cue_pos
            dist = float(np.linalg.norm(cue_to_target))
            if dist < 0.05:
                continue

            phi = math.degrees(math.atan2(float(cue_to_target[1]), float(cue_to_target[0]))) % 360

            # 生成少量“低成本安全球候选”（不仿真）
            # V0 控制轻碰/中碰，b 用低杆优先，a 先不塞，避免不可控
            v0_grid = [
                float(np.clip(0.9 + dist * 0.7, 0.8, 2.2)),
                float(np.clip(1.2 + dist * 0.9, 1.0, 2.8)),
            ]
            b_grid = [-0.20, -0.12, -0.05]
            phi_grid = [phi, (phi + 0.6) % 360, (phi - 0.6) % 360]

            for V0 in v0_grid:
                for b in b_grid:
                    for ph in phi_grid:
                        candidate = {
                            'V0': V0,
                            'phi': ph,
                            'theta': 5,   # 略下压提高可控
                            'a': 0.0,
                            'b': b
                        }

                        # 快速“好坏评估”：不仿真，只用几何估计“对手难不难”
                        s = 0.0

                        # 1) 斯诺克/挡线倾向：让我白球“尽量停在目标球背后”
                        # 用一个几何近似：鼓励白球方向接近“目标球 -> 最近坏点”的反方向（就是滚去坏点附近）
                        # 这里不算真实停球位置，只给个方向偏好
                        s += self._safety_direction_score(cue_pos, target_pos, ph, bad_points) * 1.0

                        # 2) 远离对手可打球（距离越远越好）
                        # 用“当前白球位置到对手球”的距离做下界（不仿真），只能弱引导，但至少不反着来
                        s += 0.15 * self._min_distance_to_set(cue_pos, opp_remaining, balls)

                        # 3) 贴库倾向（不仿真）：如果目标球本身靠近库边，轻碰后更可能形成贴库/难受球型
                        s += 0.25 * self._near_rail_bonus(target_pos, table)

                        # 4) 过大力量惩罚：安全球不应乱飞
                        s -= 0.35 * max(0.0, V0 - 2.2)

                        if s > best_score:
                            best_score = s
                            best_action = candidate

        if best_action is not None:
            print(f"[NewAgent] 安全策略: V0={best_action['V0']:.2f}, phi={best_action['phi']:.1f}, b={best_action['b']:.2f}, 预期得分={best_score:.2f}")
            return best_action

        # 实在找不到能碰到的安全球，随机
        return self._random_action()

    def _get_bad_points(self, table):
        """生成一些“对手难受”的目标区域点：六个袋口附近 + 四个长库中点（粗略）"""
        pts = []

        # 袋口中心
        for _, pocket in table.pockets.items():
            pts.append(np.array(pocket.center[:2], dtype=float))

        # 额外加一些“库边点”（桌面规格不一定好取，这里用 pocket 坐标做插值近似）
        # 用左上/右上/左下/右下四角袋推断长库方向
        pocket_centers = [np.array(p.center[:2], dtype=float) for p in table.pockets.values()]
        if len(pocket_centers) >= 4:
            # 取 x 最小/最大, y 最小/最大 做粗估
            xs = [p[0] for p in pocket_centers]
            ys = [p[1] for p in pocket_centers]
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))

            pts.append(np.array([(xmin + xmax) / 2, ymin], dtype=float))  # 下长库中点
            pts.append(np.array([(xmin + xmax) / 2, ymax], dtype=float))  # 上长库中点
            pts.append(np.array([xmin, (ymin + ymax) / 2], dtype=float))  # 左短库中点
            pts.append(np.array([xmax, (ymin + ymax) / 2], dtype=float))  # 右短库中点

        return pts

    def _safety_direction_score(self, cue_pos, target_pos, phi_deg, bad_points):
        """
        几何近似：安全球更倾向让白球朝“坏点”方向运行（贴库/靠袋口/角落等）。
        这里不仿真，只评估击球方向与“目标球->坏点方向”一致性。
        """
        # 击球方向单位向量（从 cue 出发）
        rad = math.radians(phi_deg)
        shot_dir = np.array([math.cos(rad), math.sin(rad)], dtype=float)

        # 选一个最接近的坏点（目标球离哪个坏点近，就鼓励往那边推）
        best = -1e9
        for bp in bad_points:
            v = bp - target_pos
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                continue
            u = v / n
            # dot 越大，说明 shot_dir 越指向 “把目标球推向坏点” 的方向
            # 虽然真实走球复杂，但这是很便宜的方向性启发
            best = max(best, float(np.dot(shot_dir, u)) - 0.15 * n)
        return best if best > -1e8 else 0.0

    def _min_distance_to_set(self, cue_pos, ids, balls):
        """当前白球到对手球的最小距离（不仿真，下界指标）"""
        md = 1e9
        for bid in ids:
            b = balls.get(bid)
            if b is None or b.state.s == 4:
                continue
            p = np.array(b.state.rvw[0][:2], dtype=float)
            md = min(md, float(np.linalg.norm(p - cue_pos)))
        if md == 1e9:
            return 0.0
        return md

    def _near_rail_bonus(self, pos, table):
        """球离库边越近越加分（用袋口坐标推断桌子边界的粗近似）"""
        pocket_centers = [np.array(p.center[:2], dtype=float) for p in table.pockets.values()]
        if len(pocket_centers) < 4:
            return 0.0
        xs = [p[0] for p in pocket_centers]
        ys = [p[1] for p in pocket_centers]
        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))

        x, y = float(pos[0]), float(pos[1])
        d = min(abs(x - xmin), abs(x - xmax), abs(y - ymin), abs(y - ymax))
        # d 越小越贴库，给正奖励
        return float(np.clip(0.25 - d, 0.0, 0.25)) * 100.0
    
######################################
class NewAgent(Agent):
    """物理计算 + 噪声感知 + 多变种评估的混合 Agent"""
    
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        self.pocket_radius = 0.05
        
        # 噪声参数（与环境保持一致）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.15,
            'theta': 0.1,
            'a': 0.005,
            'b': 0.005
        }
        
        # 每个候选方案评估次数
        self.n_simulations_per_candidate = 8
        
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
            if (best_avg_reward < 5.0):
                return self._play_safety(cue_pos, remaining_targets, balls, table, all_balls_pos, pockets)
            
            return best_candidate['params']
        
        return self._play_safety(cue_pos, remaining_targets, balls, table, all_balls_pos, pockets)
    
    def _generate_variants(self, base_params):
        """为基础参数生成多个变种（应对噪声）"""
        variants = [base_params]  # 原始参数
        
        # 变种1：力度 +0.5（应对摩擦损耗）
        variants.append({
            **base_params,
            'V0': min(base_params['V0'] + 0.5, 7.0)
        })
        
        # 变种2：力度 -0.3（更保守）
        variants.append({
            **base_params,
            'V0': max(base_params['V0'] - 0.3, 1.0)
        })
        
        # 变种3：角度 +0.3°
        variants.append({
            **base_params,
            'phi': (base_params['phi'] + 0.3) % 360
        })
        
        # 变种4：角度 -0.3°
        variants.append({
            **base_params,
            'phi': (base_params['phi'] - 0.3) % 360
        })
        
        # 变种5：组合（力度+0.5, 角度+0.3）
        variants.append({
            **base_params,
            'V0': min(base_params['V0'] + 0.5, 7.0),
            'phi': (base_params['phi'] + 0.3) % 360
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
                rewards.append(-100)  # 模拟失败
        
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
        """分析击球结果（借鉴 BasicAgentPro 的评分函数）"""
        score = 0
        
        # 检查白球状态
        cue_ball = system.balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:  # 白球进袋
            return -150
        
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
            return -500
        
        # 合法打进黑8
        if eight_pocketed and remaining_count == 0 and '8' in remaining_targets:
            return 300
        
        # 检查首球犯规
        first_hit_legal = self._check_first_hit(system, my_targets, remaining_targets)
        if not first_hit_legal:
            score -= 80
        
        # 统计得分
        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20
        
        # 合法无进球
        if score == 0 and first_hit_legal:
            score = -30
        
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
        """评估一个击球方案"""
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
        
        # 7. 评分
        score = 100
        score -= cut_angle * 1.2
        score -= dist_to_pocket * 18
        score -= dist_cue_to_aim * 10
        
        # 8. 计算击球力度
        total_dist = dist_cue_to_aim + dist_to_pocket
        V0 = 1.5 + total_dist * 1.5
        
        # 切角大时增加力度
        angle_factor = 1 + (cut_angle / 90) * 0.3
        V0 = V0 * angle_factor
        V0 = float(np.clip(V0, 1.0, 6.0))
        
        shot_params = {
            'V0': V0,
            'phi': phi,
            'theta': 3,
            'a': 0,
            'b': 0.05 if cut_angle > 40 else 0
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
    
    # ========== 改进的安全球策略 ==========
    def _play_safety(self, cue_pos, targets, balls, table, all_balls_pos, pockets):
        """
        改进的安全球策略：使用简化物理模型预测白球停球位置
        """
        print(f"[NewAgent] 采用安全球策略")
        cue_pos = np.array(cue_pos, dtype=float)

        # 推断对手目标球
        opponent_targets = ['1','2','3','4','5','6','7','9','10','11','12','13','14','15']
        opp_remaining = [bid for bid in opponent_targets if balls.get(bid) is not None and balls[bid].state.s != 4]
        if len(opp_remaining) == 0:
            opp_remaining = ['8'] if balls.get('8') is not None and balls['8'].state.s != 4 else opponent_targets

        # 获取对手球的位置
        opp_positions = []
        for bid in opp_remaining:
            if balls.get(bid) is not None and balls[bid].state.s != 4:
                opp_positions.append(np.array(balls[bid].state.rvw[0][:2], dtype=float))

        # 获取袋口位置（用于判断是否挡线）
        pocket_positions = list(pockets.values())

        best_action = None
        best_score = -float('inf')

        # 对每个可打的目标球尝试安全球
        for target_id in targets:
            if balls.get(target_id) is None or balls[target_id].state.s == 4:
                continue

            target_pos = np.array(balls[target_id].state.rvw[0][:2], dtype=float)

            # 检查路径遮挡
            if self._is_path_blocked(cue_pos, target_pos, target_id, all_balls_pos):
                continue

            cue_to_target = target_pos - cue_pos
            dist = float(np.linalg.norm(cue_to_target))
            if dist < 0.05:
                continue

            phi = math.degrees(math.atan2(float(cue_to_target[1]), float(cue_to_target[0]))) % 360

            # 生成安全球候选动作（低速 + 低杆）
            v0_candidates = [
                float(np.clip(0.8 + dist * 0.5, 0.7, 2.0)),  # 轻碰
                float(np.clip(1.2 + dist * 0.8, 1.0, 2.5)),  # 中等
            ]
            
            b_candidates = [-0.25, -0.15, -0.08]  # 低杆为主
            phi_offsets = [0, 0.8, -0.8, 1.5, -1.5]  # 角度微调

            for V0 in v0_candidates:
                for b in b_candidates:
                    for offset in phi_offsets:
                        ph = (phi + offset) % 360
                        
                        candidate = {
                            'V0': V0,
                            'phi': ph,
                            'theta': 5,
                            'a': 0.0,
                            'b': b
                        }

                        # 预测白球停球位置（简化物理模型）
                        predicted_cue_pos = self._predict_cue_stop_position(
                            cue_pos, target_pos, V0, ph, b
                        )

                        # 评估这个停球位置的安全性
                        score = self._evaluate_safety_position(
                            predicted_cue_pos,
                            target_pos,
                            opp_positions,
                            pocket_positions,
                            table
                        )

                        # 力度惩罚（安全球不宜过猛）
                        score -= 0.4 * max(0.0, V0 - 2.0)

                        if score > best_score:
                            best_score = score
                            best_action = candidate

        if best_action is not None:
            print(f"[NewAgent] 安全策略: V0={best_action['V0']:.2f}, phi={best_action['phi']:.1f}, "
                  f"b={best_action['b']:.2f}, 预期得分={best_score:.2f}")
            return best_action

        return self._random_action()

    def _predict_cue_stop_position(self, cue_pos, target_pos, V0, phi_deg, b):
        """
        简化物理模型预测白球停球位置
        
        参数：
            cue_pos: 白球当前位置
            target_pos: 目标球位置
            V0: 击球速度
            phi_deg: 击球角度（度）
            b: 垂直击球参数（负值=低杆）
        
        返回：
            预测的白球停球位置
        """
        cue_pos = np.array(cue_pos, dtype=float)
        target_pos = np.array(target_pos, dtype=float)
        
        # 计算碰撞点（假设完美击中目标球中心）
        cue_to_target = target_pos - cue_pos
        dist_to_collision = float(np.linalg.norm(cue_to_target) - 2 * self.ball_radius)
        
        if dist_to_collision <= 0:
            return cue_pos  # 距离太近，无法预测
        
        # 击球方向
        rad = math.radians(phi_deg)
        shot_dir = np.array([math.cos(rad), math.sin(rad)], dtype=float)
        
        # 碰撞点位置
        collision_point = cue_pos + shot_dir * dist_to_collision
        
        # 简化的速度衰减模型（考虑摩擦和距离）
        # 假设速度衰减系数 k ≈ 0.15 m^-1
        k_friction = 0.15
        v_at_collision = V0 * math.exp(-k_friction * dist_to_collision)
        
        # 低杆效果：负b值会让白球在碰撞后获得回旋（向后运动）
        # 简化：b < 0 时，碰撞后白球会反向运动一段距离
        if b < -0.1:
            # 低杆效果：白球碰撞后反弹
            bounce_factor = abs(b) * 1.5  # 低杆越强，反弹越远
            v_after_collision = v_at_collision * 0.3 * bounce_factor  # 碰撞后保留部分速度
            
            # 反向运动距离（考虑摩擦）
            bounce_distance = (v_after_collision ** 2) / (2 * 9.8 * 0.2)  # 假设摩擦系数 0.2
            
            # 最终位置：碰撞点向后反弹
            final_pos = collision_point - shot_dir * min(bounce_distance, dist_to_collision * 0.5)
        else:
            # 正常击球：白球继续前进（但速度降低）
            v_after_collision = v_at_collision * 0.5
            forward_distance = (v_after_collision ** 2) / (2 * 9.8 * 0.2)
            
            final_pos = collision_point + shot_dir * min(forward_distance, 0.3)
        
        return final_pos

    def _evaluate_safety_position(self, cue_pos, target_pos, opp_positions, pocket_positions, table):
        """
        评估白球停球位置的安全性
        
        核心指标：
        1. 是否挡住了对手到袋口的线路（斯诺克）
        2. 距离对手球的最小距离（越远越好）
        3. 是否贴库（贴库难打）
        """
        cue_pos = np.array(cue_pos, dtype=float)
        target_pos = np.array(target_pos, dtype=float)
        
        score = 0.0
        
        # 1. 斯诺克效果：检查是否挡住对手球到袋口的线路
        blocked_lines = 0
        for opp_pos in opp_positions:
            for pocket_pos in pocket_positions:
                if self._is_blocking_line(cue_pos, opp_pos, pocket_pos):
                    blocked_lines += 1
        
        score += blocked_lines * 25.0  # 每条被挡的线路加25分
        
        # 2. 距离对手球的距离（越远越安全）
        min_dist_to_opp = float('inf')
        for opp_pos in opp_positions:
            dist = float(np.linalg.norm(cue_pos - opp_pos))
            min_dist_to_opp = min(min_dist_to_opp, dist)
        
        # 距离奖励：超过0.5m开始加分
        if min_dist_to_opp < float('inf'):
            score += max(0, (min_dist_to_opp - 0.5) * 30.0)
        
        # 3. 贴库效果（距离库边越近越好）
        rail_distance = self._distance_to_nearest_rail(cue_pos, table)
        if rail_distance < 0.15:  # 距离库边小于15cm
            score += (0.15 - rail_distance) * 80.0
        
        # 4. 惩罚：如果白球停在桌子中心区域（对手容易打）
        table_center = self._get_table_center(table)
        dist_to_center = float(np.linalg.norm(cue_pos - table_center))
        max_dist = float(np.linalg.norm(self._get_table_dimensions(table) / 2))
        
        center_penalty = max(0, 1.0 - dist_to_center / max_dist) * 15.0
        score -= center_penalty
        
        return score

    def _is_blocking_line(self, blocker_pos, start_pos, end_pos):
        """
        检查 blocker_pos 是否挡在 start_pos 到 end_pos 的线路上
        """
        blocker_pos = np.array(blocker_pos, dtype=float)
        start_pos = np.array(start_pos, dtype=float)
        end_pos = np.array(end_pos, dtype=float)
        
        line_vec = end_pos - start_pos
        line_length = float(np.linalg.norm(line_vec))
        
        if line_length < 0.01:
            return False
        
        line_unit = line_vec / line_length
        
        # 计算 blocker 到线段的垂直距离
        start_to_blocker = blocker_pos - start_pos
        proj_length = float(np.dot(start_to_blocker, line_unit))
        
        # blocker 必须在线段之间（不在两端）
        if proj_length < 0.05 or proj_length > line_length - 0.05:
            return False
        
        perp_dist = float(np.linalg.norm(start_to_blocker - proj_length * line_unit))
        
        # 如果垂直距离小于球半径的2倍，则认为挡住了
        return perp_dist < 2 * self.ball_radius + 0.01

    def _distance_to_nearest_rail(self, pos, table):
        """计算点到最近库边的距离"""
        pos = np.array(pos, dtype=float)
        
        # 从袋口坐标推断桌子边界
        pocket_centers = [np.array(p.center[:2], dtype=float) for p in table.pockets.values()]
        if len(pocket_centers) < 4:
            return float('inf')
        
        xs = [float(p[0]) for p in pocket_centers]
        ys = [float(p[1]) for p in pocket_centers]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        x, y = float(pos[0]), float(pos[1])
        
        # 计算到四条边的距离
        distances = [
            abs(x - xmin),  # 左边
            abs(x - xmax),  # 右边
            abs(y - ymin),  # 下边
            abs(y - ymax),  # 上边
        ]
        
        return min(distances)

    def _get_table_center(self, table):
        """获取桌子中心点"""
        pocket_centers = [np.array(p.center[:2], dtype=float) for p in table.pockets.values()]
        xs = [float(p[0]) for p in pocket_centers]
        ys = [float(p[1]) for p in pocket_centers]
        
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        
        return np.array([center_x, center_y], dtype=float)

    def _get_table_dimensions(self, table):
        """获取桌子尺寸（宽度, 长度）"""
        pocket_centers = [np.array(p.center[:2], dtype=float) for p in table.pockets.values()]
        xs = [float(p[0]) for p in pocket_centers]
        ys = [float(p[1]) for p in pocket_centers]
        
        width = max(xs) - min(xs)
        length = max(ys) - min(ys)
        
        return np.array([width, length], dtype=float)