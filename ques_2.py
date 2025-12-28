import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Set, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# ===========================
# 1. 基础类库 (保留你提供的原始内容)
# ===========================
@dataclass
class SimConfig:
    """仿真参数配置类"""
    dt: float = 0.01              # 仿真步长 (s)
    max_time: float = 60.0        # 超时时间 (s)
    
    # 机器人物理参数
    n_robots: int = 8             # 机器人数量
    # 注意：题目图示直径0.5m => 半径0.25m。
    # 用户Prompt提及半径0.5m，此处以图示直径0.5m为准，若需修改请调整此处。
    robot_radius: float = 0.25    
    
    max_v: float = 1.0            # 最大速度 1m/s
    max_a: float = 5.0            # 最大加速度 5m/s^2 (仅二阶系统有效)
    comm_range: float = 2.5       # 通讯距离 2.5m
    
    # 任务模式
    dynamics_model: str = 'second_order'  # 'first_order' 或 'second_order'
    comm_delay: float = 0.0               # 通讯延时 (s)
    
    # 成功判定
    target_x: float = 25.0        # 全部机器人超过此X坐标视为通过

@dataclass
class RobotState:
    """机器人状态向量"""
    pos: np.ndarray  # [x, y]
    vel: np.ndarray  # [vx, vy]
    acc: np.ndarray  # [ax, ay]

    @staticmethod
    def create_zero():
        return RobotState(np.zeros(2), np.zeros(2), np.zeros(2))

class RobotAgent:
    """封装机器人的动力学和状态更新"""
    def __init__(self, agent_id: int, initial_pos: np.ndarray, config: SimConfig):
        self.id = agent_id
        self.config = config
        self.state = RobotState(
            pos=np.array(initial_pos, dtype=float),
            vel=np.zeros(2),
            acc=np.zeros(2)
        )
        self.history = deque()
        self.history.append((0.0, self.copy_state()))

    def copy_state(self):
        return RobotState(self.state.pos.copy(), self.state.vel.copy(), self.state.acc.copy())

    def update(self, control_input: np.ndarray, dt: float, current_time: float):
        """根据控制输入更新状态 (一阶/二阶)"""
        if self.config.dynamics_model == 'first_order':
            target_vel = control_input
            speed = np.linalg.norm(target_vel)
            if speed > self.config.max_v:
                target_vel = target_vel / speed * self.config.max_v
            self.state.vel = target_vel
            self.state.pos += self.state.vel * dt
            self.state.acc = np.zeros(2)

        elif self.config.dynamics_model == 'second_order':
            target_acc = control_input
            acc_mag = np.linalg.norm(target_acc)
            if acc_mag > self.config.max_a:
                target_acc = target_acc / acc_mag * self.config.max_a
            self.state.acc = target_acc
            new_vel = self.state.vel + self.state.acc * dt
            speed = np.linalg.norm(new_vel)
            if speed > self.config.max_v:
                new_vel = new_vel / speed * self.config.max_v
            self.state.vel = new_vel
            self.state.pos += self.state.vel * dt

        self.history.append((current_time, self.copy_state()))
        # 仅保留最近2秒的历史用于通讯延时查询
        while len(self.history) > 1 and self.history[0][0] < current_time - self.config.comm_delay - 0.5:
            self.history.popleft()

    def get_delayed_state(self, current_time: float, delay: float) -> RobotState:
        if delay <= 1e-4: return self.state
        query_time = current_time - delay
        if query_time <= 0: return self.history[0][1]
        for t, state in reversed(self.history):
            if t <= query_time: return state
        return self.history[0][1]

class Environment:
    """定义环境、障碍物及碰撞检测接口"""
    def __init__(self):
        self.obstacles = []
        # 通道定义：长20m，宽2m (y: -1 to 1)，入口 x=5
        self.obstacles.append({'rect': np.array([5, 1.0, 20, 0.5]), 'type': 'wall'})   # 上墙
        self.obstacles.append({'rect': np.array([5, -1.5, 20, 0.5]), 'type': 'wall'}) # 下墙
        # 中间障碍物
        self.obstacles.append({'rect': np.array([7, 0.2, 2.0, 0.8]), 'type': 'obstacle'})

    def check_static_collision(self, pos: np.ndarray, radius: float) -> bool:
        """检测机器人与静态环境(墙、障碍物)的碰撞"""
        x, y = pos
        for obs in self.obstacles:
            ox, oy, w, h = obs['rect']
            # AABB 碰撞检测 (考虑机器人半径)
            if (x + radius > ox) and (x - radius < ox + w) and \
               (y + radius > oy) and (y - radius < oy + h):
                return True
        return False

    def check_inter_robot_collision(self, robots: List[RobotAgent], radius: float) -> Set[int]:
        """
        检测机器人之间的碰撞
        返回发生碰撞的机器人ID集合
        """
        collided_ids = set()
        n = len(robots)
        positions = np.array([r.state.pos for r in robots])
        
        # 利用矩阵广播计算两两距离矩阵
        # diff[i, j] = pos[i] - pos[j]
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # 碰撞条件: dist < 2 * radius
        # 对角线设为无穷大(自己不撞自己)
        np.fill_diagonal(dist_sq, np.inf)
        
        threshold_sq = (2 * radius) ** 2
        # 找到小于阈值的索引对
        collisions = np.argwhere(dist_sq < threshold_sq)
        
        for i, j in collisions:
            collided_ids.add(robots[i].id)
            collided_ids.add(robots[j].id)
            
        return collided_ids

    def plot(self, ax):
        for obs in self.obstacles:
            color = 'black' if obs['type'] == 'wall' else 'red'
            rect = patches.Rectangle((obs['rect'][0], obs['rect'][1]), 
                                     obs['rect'][2], obs['rect'][3], 
                                     linewidth=1, edgecolor='none', facecolor=color, alpha=0.6)
            ax.add_patch(rect)
        # 绘制终点线提示
        ax.plot([25, 25], [-3, 3], 'g--', alpha=0.5, label='Finish Line')

class CommChannel:
    def __init__(self, robots: List[RobotAgent], config: SimConfig):
        self.robots = robots
        self.config = config

    def get_neighbors(self, agent_id: int, current_time: float):
        neighbors = []
        me = self.robots[agent_id]
        my_real_pos = me.state.pos
        
        for other in self.robots:
            if other.id == agent_id: continue
            dist = np.linalg.norm(my_real_pos - other.state.pos)
            if dist <= self.config.comm_range:
                observed_state = other.get_delayed_state(current_time, self.config.comm_delay)
                neighbors.append({'id': other.id, 'state': observed_state, 'dist': dist})
        return neighbors

class BaseController:
    def __init__(self, config: SimConfig):
        self.config = config

    def calculate_control(self, agent_id: int, current_state: RobotState, neighbors: List[dict], env: Environment) -> np.ndarray:
        raise NotImplementedError

# ===========================
# 2. 第二问核心解答: 队形控制器
# ===========================
class SolutionQ2Controller(BaseController):
    """二阶系统的通道通行控制器，聚焦管道队形与碰撞抑制。"""

    def __init__(self, config: SimConfig):
        super().__init__(config)
        # 长itudinal跟随与PD调节参数
        self.kp = 3.2
        self.kv = 2.8
        self.k_follow_p = 4.6
        self.k_follow_d = 2.0
        self.desired_gap = 1.15
        self.lookahead_lead = 1.8
        self.max_forward_look = 4.5
        self.max_backtrack = 0.9
        self.cruise_speed = 1.0
        self.lateral_gain = 1.5
        self.forward_push = 0.32
        self.catchup_boost = 0.45
        self.final_kp = 3.4
        self.final_kv = 3.2
        self.formation_tolerance = 0.2

        # 人工势场参数
        self.k_rep_wall = 14.0
        self.k_rep_agent = 20.0
        self.k_rep_obstacle = 18.0
        self.rep_dist_agent = 1.2
        self.wall_buffer = 0.32
        self.final_rep_scale = 0.45

        # 通道几何与编队信息
        self.reform_trigger = 25.5
        self.reform_x = 27.0
        self.finish_center = np.array([30.0, 0.0])
        self.finish_radius = 2.5
        self.final_min_x = self.finish_center[0] - self.finish_radius
        self.offset_step = 0.12
        self.max_offset = 0.24

        # 状态缓存用于串行车辆队形
        self.latest_states = [
            {'pos': None, 'vel': None} for _ in range(self.config.n_robots)
        ]
        # 终点线后的编队阶段标记
        self.final_stage_flags = [False] * self.config.n_robots
        # 终态圆形阵位
        self.slot_positions = self._build_slot_positions()
        self.slot_assignments = [None] * self.config.n_robots
        self.slot_occupied = [False] * self.config.n_robots
        self.slot_sequence = self._build_slot_sequence()
        self.next_slot_pointer = 0
        self.ready_to_reform = False

    def calculate_control(
        self,
        agent_id: int,
        current_state: RobotState,
        neighbors: List[dict],
        env: Environment
    ) -> np.ndarray:
        p = current_state.pos
        v = current_state.vel

        # 缓存当前状态供后续车辆查询
        self.latest_states[agent_id]['pos'] = p.copy()
        self.latest_states[agent_id]['vel'] = v.copy()

        if not self.final_stage_flags[agent_id] and p[0] >= self.config.target_x:
            self.final_stage_flags[agent_id] = True
            self._assign_final_slot(agent_id, p)
            if all(self.final_stage_flags):
                self.ready_to_reform = True

        if self.final_stage_flags[agent_id] and self.slot_assignments[agent_id] is None:
            self._assign_final_slot(agent_id, p)

        lane_offset = self._lane_offset(agent_id)
        final_stage = self.final_stage_flags[agent_id]

        if final_stage:
            target_pos, desired_vel = self._final_formation_target(agent_id, p)
            u_pd = self.final_kp * (target_pos - p) + self.final_kv * (desired_vel - v)
        else:
            leader_final = self.final_stage_flags[agent_id - 1] if agent_id > 0 else False
            target_y = self._target_lateral_position(p[0], lane_offset)
            target_x, desired_speed_x, leader_state = self._target_longitudinal(agent_id, p, leader_final)

            target_pos = np.array([target_x, target_y])
            desired_vel = np.array([
                desired_speed_x,
                self.lateral_gain * (target_y - p[1])
            ])
            desired_vel = self._limit_norm(desired_vel, self.config.max_v)
            u_pd = self.kp * (target_pos - p) + self.kv * (desired_vel - v)

            if agent_id > 0 and leader_state is not None and not self.final_stage_flags[agent_id - 1]:
                leader_pos, leader_vel = leader_state
                dx = leader_pos[0] - p[0]
                dv = leader_vel[0] - v[0]
                follow_term = self.k_follow_p * (dx - self.desired_gap) + self.k_follow_d * dv
                follow_term = float(np.clip(follow_term, -self.config.max_a, self.config.max_a))
                u_pd[0] += follow_term

            u_pd[0] += self._forward_bias(p[0], desired_speed_x, v[0])

        u_rep = self._agent_repulsion(p, neighbors, final_stage)
        u_rep += self._wall_repulsion(p)

        total_acc = u_pd + u_rep
        return self._limit_norm(total_acc, self.config.max_a)

    def _limit_norm(self, vec: np.ndarray, limit: float) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > limit and norm > 1e-6:
            return vec / norm * limit
        return vec

    def _lane_offset(self, agent_id: int) -> float:
        level = agent_id // 2
        sign = 1 if agent_id % 2 == 0 else -1
        return sign * min(level * self.offset_step, self.max_offset)

    def _target_lateral_position(self, x: float, lane_offset: float) -> float:
        centerline = self._centerline_y(x)
        weight = self._lane_weight(x)
        target_y = centerline + weight * lane_offset
        return float(np.clip(target_y, -0.75, 0.75))

    def _build_slot_positions(self) -> List[np.ndarray]:
        slots = []
        for i in range(self.config.n_robots):
            angle = 2.0 * np.pi * i / self.config.n_robots
            pos = self.finish_center + np.array([
                np.cos(angle),
                np.sin(angle)
            ]) * self.finish_radius
            if pos[0] < self.final_min_x:
                pos[0] = self.final_min_x
            slots.append(pos.copy())
        return slots

    def _build_slot_sequence(self) -> List[int]:
        x_values = [pos[0] for pos in self.slot_positions]
        far_index = int(np.argmax(x_values))
        angles = []
        for idx, pos in enumerate(self.slot_positions):
            rel = pos - self.finish_center
            angle = float(np.arctan2(rel[1], rel[0]))
            angles.append(angle)

        pos_indices = [idx for idx in range(self.config.n_robots) if idx != far_index and angles[idx] > 1e-6]
        neg_indices = [idx for idx in range(self.config.n_robots) if idx != far_index and angles[idx] < -1e-6]
        zero_indices = [idx for idx in range(self.config.n_robots) if idx != far_index and abs(angles[idx]) <= 1e-6]

        pos_indices.sort(key=lambda idx: angles[idx])
        neg_indices.sort(key=lambda idx: abs(angles[idx]))

        sequence = [far_index]
        pos_ptr = 0
        neg_ptr = 0
        while pos_ptr < len(pos_indices) or neg_ptr < len(neg_indices):
            if pos_ptr < len(pos_indices):
                sequence.append(pos_indices[pos_ptr])
                pos_ptr += 1
            if neg_ptr < len(neg_indices):
                sequence.append(neg_indices[neg_ptr])
                neg_ptr += 1

        for idx in zero_indices:
            sequence.append(idx)

        remaining = [idx for idx in range(self.config.n_robots) if idx not in sequence]
        sequence.extend(remaining)
        return sequence

    def _target_longitudinal(self, agent_id: int, p: np.ndarray, leader_final: bool):
        leader_state = None
        if agent_id == 0 or leader_final:
            target_x = p[0] + self.lookahead_lead
            desired_speed_x = self.cruise_speed
        else:
            leader_state = self._get_leader_state(agent_id - 1)
            if leader_state is not None:
                leader_pos, leader_vel = leader_state
                target_x = leader_pos[0] - self.desired_gap
                dx = leader_pos[0] - p[0]
                gap_error = dx - self.desired_gap
                speed_ref = leader_vel[0] + 0.55 * gap_error
                desired_speed_x = float(np.clip(speed_ref, 0.4, self.cruise_speed))
            else:
                target_x = p[0] - 0.3
                desired_speed_x = self.cruise_speed * 0.8

        min_x = p[0] - self.max_backtrack
        dynamic_forward = self.max_forward_look
        if agent_id > 0 and not leader_final:
            leader_state = leader_state if leader_state is not None else self._get_leader_state(agent_id - 1)
            if leader_state is not None:
                leader_pos, _ = leader_state
                dx = leader_pos[0] - p[0]
                gap_error = max(0.0, dx - self.desired_gap)
                dynamic_forward += min(6.0 - self.max_forward_look, 0.6 * gap_error)
        max_x = p[0] + dynamic_forward
        target_x = float(np.clip(target_x, min_x, max_x))
        return target_x, desired_speed_x, leader_state if not leader_final else None

    def _assign_final_slot(self, agent_id: int, position: np.ndarray):
        if self.slot_assignments[agent_id] is not None:
            return
        slot_idx = self._next_sequence_slot()
        if slot_idx is None:
            slot_idx = self._find_closest_available_slot(position)
            if slot_idx is None:
                return
        self.slot_assignments[agent_id] = slot_idx
        self.slot_occupied[slot_idx] = True

    def _next_sequence_slot(self) -> Optional[int]:
        while self.next_slot_pointer < len(self.slot_sequence):
            candidate = self.slot_sequence[self.next_slot_pointer]
            self.next_slot_pointer += 1
            if not self.slot_occupied[candidate]:
                return candidate
        return None

    def _find_closest_available_slot(self, position: np.ndarray) -> Optional[int]:
        available = [idx for idx, taken in enumerate(self.slot_occupied) if not taken]
        if not available:
            return None
        dists = [np.linalg.norm(position - self.slot_positions[idx]) for idx in available]
        return available[int(np.argmin(dists))]

    def _get_leader_state(self, leader_id: int):
        state = self.latest_states[leader_id]
        if state['pos'] is None or state['vel'] is None:
            return None
        return state['pos'], state['vel']

    def _centerline_y(self, x: float) -> float:
        if x <= 5.0:
            return 0.0
        if x <= 6.5:
            ratio = (x - 5.0) / 1.5
            return -0.5 * ratio
        if x <= 12.0:
            return -0.5
        if x <= 16.0:
            ratio = (x - 12.0) / 4.0
            return -0.5 * (1.0 - ratio)
        return 0.0

    def _lane_weight(self, x: float) -> float:
        if x <= 5.0:
            return 0.0
        if x >= 7.5:
            return 1.0
        return (x - 5.0) / 2.5

    def _agent_repulsion(self, p: np.ndarray, neighbors: List[dict], final_stage: bool) -> np.ndarray:
        rep = np.zeros(2)
        scale = self.final_rep_scale if final_stage else 1.0
        for nb in neighbors:
            diff = p - nb['state'].pos
            dist = np.linalg.norm(diff)
            if dist < 1e-4:
                continue
            if dist < self.rep_dist_agent:
                factor = self.k_rep_agent * (1.0 / dist - 1.0 / self.rep_dist_agent) / (dist ** 2)
                factor *= scale
                rep += factor * (diff / dist)
        return rep

    def _wall_repulsion(self, p: np.ndarray) -> np.ndarray:
        rep = np.zeros(2)
        # 通道墙体
        if 5.0 <= p[0] <= 25.0:
            top_clear = 1.0 - (p[1] + self.config.robot_radius)
            if top_clear < self.wall_buffer:
                top_clear = max(top_clear, 0.02)
                rep[1] -= self.k_rep_wall / (top_clear + 0.05)
            bottom_clear = (p[1] - self.config.robot_radius) - (-1.0)
            if bottom_clear < self.wall_buffer:
                bottom_clear = max(bottom_clear, 0.02)
                rep[1] += self.k_rep_wall / (bottom_clear + 0.05)

        # 中央障碍物 (推向下侧通道)
        if 6.5 <= p[0] <= 9.5 and p[1] > -0.15:
            rep[1] -= self.k_rep_obstacle * (0.15 + p[1])

        return rep

    def _final_formation_target(self, agent_id: int, p: np.ndarray):
        slot_idx = self.slot_assignments[agent_id]
        if slot_idx is None:
            slot_idx = agent_id
        target_pos = self.slot_positions[slot_idx].copy()
        desired_vel = np.zeros(2)
        return target_pos, desired_vel

    def _forward_bias(self, x: float, desired_speed_x: float, current_vx: float) -> float:
        if x >= self.reform_x:
            return 0.0
        bias = self.forward_push
        speed_error = max(0.0, desired_speed_x - current_vx)
        bias += self.catchup_boost * speed_error
        return float(np.clip(bias, 0.0, self.config.max_a))

    def formation_error(self, robots: List[RobotAgent]) -> float:
        max_err = 0.0
        for robot in robots:
            target_pos, _ = self._final_formation_target(robot.id, robot.state.pos)
            err = float(np.linalg.norm(robot.state.pos - target_pos))
            if err > max_err:
                max_err = err
        return max_err

# ===========================
# 3. 仿真与主程序
# ===========================
class Simulation:
    """
    统一管理仿真流程、状态记录与可视化
    """
    def __init__(self, config: SimConfig, controller: BaseController, env: Environment = None):
        self.config = config
        self.controller = controller
        self.env = env if env else Environment()
        self.current_time = 0.0
        
        # 初始化机器人
        self.robots = []
        self._init_robots()
        
        self.comm = CommChannel(self.robots, config)
        
        # 数据记录 (用于回放和分析)
        # history[t] = [state_robot_0, state_robot_1, ...]
        self.data_history = [] 
        self.collision_log = [] # (time, list_of_collided_ids)
        self.finish_times = [None] * self.config.n_robots
        self.formation_complete_time = None

    def _init_robots(self):
        """初始化为左侧圆形分布"""
        center = np.array([0.0, 0.0])
        radius = 2.5
        for i in range(self.config.n_robots):
            angle = 2 * np.pi * i / self.config.n_robots
            pos = center + np.array([np.cos(angle), np.sin(angle)]) * radius
            self.robots.append(RobotAgent(i, pos, self.config))

    def step(self) -> bool:
        """执行单步仿真。返回 False 表示需要停止(超时)。"""
        if self.current_time > self.config.max_time:
            return False

        # 1. 计算控制
        controls = []
        for r in self.robots:
            neighbors = self.comm.get_neighbors(r.id, self.current_time)
            u = self.controller.calculate_control(r.id, r.state, neighbors, self.env)
            controls.append(u)
        
        # 2. 更新物理状态
        step_data = []
        time_stamp = self.current_time + self.config.dt
        for i, r in enumerate(self.robots):
            r.update(controls[i], self.config.dt, self.current_time)
            step_data.append(r.state.pos.copy())
            if self.finish_times[i] is None and r.state.pos[0] >= self.config.target_x:
                self.finish_times[i] = time_stamp
        
        # 3. 记录数据
        self.data_history.append(step_data)
        
        # 4. 碰撞检测 (记录但不打断仿真，作为惩罚项)
        # A. 机器人-机器人碰撞
        inter_collisions = self.env.check_inter_robot_collision(self.robots, self.config.robot_radius)
        # B. 机器人-环境碰撞
        static_collisions = set()
        for r in self.robots:
            if self.env.check_static_collision(r.state.pos, self.config.robot_radius):
                static_collisions.add(r.id)
                
        all_collisions = inter_collisions.union(static_collisions)
        if all_collisions:
            self.collision_log.append((self.current_time, list(all_collisions)))

        assigned_ready = all(slot is not None for slot in self.controller.slot_assignments)
        if self.formation_complete_time is None and all(t is not None for t in self.finish_times) and assigned_ready:
            formation_err = self.controller.formation_error(self.robots)
            avg_speed = float(np.mean([np.linalg.norm(r.state.vel) for r in self.robots]))
            if formation_err < self.controller.formation_tolerance and avg_speed < 0.15:
                self.formation_complete_time = time_stamp

        self.current_time += self.config.dt
        return True

    def check_success(self) -> bool:
        """检查是否所有机器人到达目标区域"""
        # 假设目标是全员越过 x = target_x
        xs = [r.state.pos[0] for r in self.robots]
        return min(xs) > self.config.target_x

    def run(self) -> float:
        """运行仿真直到成功或超时，返回耗时"""
        print(f"Simulation Start: {self.config.dynamics_model}, Delay={self.config.comm_delay}s")
        while self.step():
            if self.check_success():
                print(f"Task Completed! Time: {self.current_time:.2f}s")
                return self.current_time
            
        print(f"Simulation Timeout after {self.config.max_time}s")
        return self.current_time

    def animate(self, filename: str = None):
        """生成并展示动画"""
        print("Generating animation...")
        n_steps = len(self.data_history)
        # 为了动画流畅且不生成过大文件，每隔k帧抽样一次
        skip = max(1, int(0.05 / self.config.dt)) 
        
        fig, ax = plt.subplots(figsize=(12, 5))
        self.env.plot(ax)
        
        colors = plt.cm.get_cmap('tab10', self.config.n_robots)
        
        # 初始化图形对象
        robot_circles = []
        robot_trails = []
        for i in range(self.config.n_robots):
            c = patches.Circle((0,0), self.config.robot_radius, color=colors(i), zorder=10)
            ax.add_patch(c)
            robot_circles.append(c)
            line, = ax.plot([], [], color=colors(i), alpha=0.3, linewidth=1)
            robot_trails.append(line)
            
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        # 坐标轴范围
        ax.set_xlim(-4, 35)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True)

        def update(frame):
            idx = frame * skip
            if idx >= n_steps: idx = n_steps - 1
            
            current_positions = self.data_history[idx]
            timestamp = idx * self.config.dt
            
            for i, pos in enumerate(current_positions):
                robot_circles[i].set_center(pos)
                # 绘制尾迹 (最近100个点)
                start_trail = max(0, idx - 100)
                trail_data = np.array([step[i] for step in self.data_history[start_trail:idx+1]])
                if len(trail_data) > 0:
                    robot_trails[i].set_data(trail_data[:,0], trail_data[:,1])
            
            time_text.set_text(f'Time: {timestamp:.2f}s')
            return robot_circles + robot_trails + [time_text]

        ani = FuncAnimation(fig, update, frames=n_steps//skip, interval=30, blit=True)
        
        if filename:
            ani.save(filename, writer='pillow', fps=30)
            print(f"Animation saved to {filename}")
        
        plt.show()

if __name__ == "__main__":
    # 配置：二阶系统，最大加速度5，无延迟
    cfg = SimConfig(
        dynamics_model='second_order',
        max_a=5.0,
        max_v=1.0,
        comm_delay=0.0,
        comm_range=2.5,
        target_x=25.0
    )
    
    # 实例化控制器
    controller = SolutionQ2Controller(cfg)
    
    # 运行仿真
    sim = Simulation(cfg, controller)
    max_steps = int(cfg.max_time / cfg.dt)
    success_time = None
    timed_out = False

    for _ in range(max_steps):
        if not sim.step():
            timed_out = True
            break

        if sim.formation_complete_time is not None:
            success_time = sim.formation_complete_time
            formation_err = controller.formation_error(sim.robots)
            print(f"Task Completed! Time: {success_time:.2f}s, formation error {formation_err:.3f}m")
            break

    if success_time is None:
        if timed_out:
            print(f"Simulation Timeout after {cfg.max_time}s")
        else:
            print("Simulation ended without achieving the desired formation.")

    print("Finish line crossing times:")
    for idx, t in enumerate(sim.finish_times):
        if t is None:
            print(f"  Robot {idx}: not reached")
        else:
            print(f"  Robot {idx}: {t:.2f}s")

    print("Final slot assignments:")
    for idx, slot in enumerate(controller.slot_assignments):
        if slot is None:
            print(f"  Robot {idx}: unassigned")
        else:
            print(f"  Robot {idx}: slot {slot}")

    if sim.formation_complete_time is not None:
        print(f"Formation recovery time: {sim.formation_complete_time:.2f}s")
    else:
        print("Formation recovery time: not achieved within simulation.")

    # 检查是否有碰撞
    if not sim.collision_log:
        print("Result: No collisions detected.")
    else:
        print(f"Result: {len(sim.collision_log)} collision events occurred.")
        
    sim.animate("simulation_ques2_v2.gif")