import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# ===========================
# 1. 配置与常量定义
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

# ===========================
# 2. 基础数据结构
# ===========================
@dataclass
class RobotState:
    """机器人状态向量"""
    pos: np.ndarray  # [x, y]
    vel: np.ndarray  # [vx, vy]
    acc: np.ndarray  # [ax, ay]

    @staticmethod
    def create_zero():
        return RobotState(np.zeros(2), np.zeros(2), np.zeros(2))

# ===========================
# 3. 机器人智能体 (RobotAgent)
# ===========================
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

# ===========================
# 4. 环境类 (Environment) - 增强碰撞检测
# ===========================
class Environment:
    """定义环境、障碍物及碰撞检测接口"""
    def __init__(self):
        self.obstacles = []
        # 通道定义：长20m，宽2m (y: -1 to 1)，入口 x=5
        self.obstacles.append({'rect': np.array([5, 1.0, 20, 0.5]), 'type': 'wall'})   # 上墙
        self.obstacles.append({'rect': np.array([5, -1.5, 20, 0.5]), 'type': 'wall'}) # 下墙
        # 中间障碍物
        self.obstacles.append({'rect': np.array([14, -0.2, 2.0, 0.4]), 'type': 'obstacle'})

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

# ===========================
# 5. 通讯管理器 (CommChannel)
# ===========================
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

# ===========================
# 6. 控制器基类
# ===========================
class BaseController:
    def __init__(self, config: SimConfig):
        self.config = config

    def calculate_control(self, agent_id: int, current_state: RobotState, neighbors: List[dict], env: Environment) -> np.ndarray:
        raise NotImplementedError

# ===========================
# 7. 仿真引擎 (Simulation Class)
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
        for i, r in enumerate(self.robots):
            r.update(controls[i], self.config.dt, self.current_time)
            step_data.append(r.state.pos.copy())
        
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
        ax.set_xlim(-4, 30)
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