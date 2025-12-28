import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
import csv


# ==================== 1. Simulation Parameters ====================
class SimConfig:
    """仿真基础参数配置"""

    def __init__(self):
        self.dt = 0.02
        self.max_time = 100.0  # 适当延长最大时间以确保队形恢复
        self.n_robots = 8
        self.robot_radius = 0.25
        self.max_v = 1.0
        self.safe_dist = 0.7
        self.formation_center = np.array([33.0, 0.0])  # 最终集合中心
        self.recovery_threshold = 0.3  # 队形恢复判定阈值


class Environment:
    """环境障碍物定义"""

    def __init__(self):
        # 墙壁配置
        self.walls = [
            {'rect': np.array([5.0, 1.0, 20.0, 0.5])},  # 上墙
            {'rect': np.array([5.0, -1.5, 20.0, 0.5])}  # 下墙
        ]
        # 静态障碍物 (统一逻辑与视觉坐标)
        self.obs = {'rect': np.array([7.0, 0.2, 2.0, 0.8])}


# ==================== 2. Controller ====================
class RobotController:
    """机器人控制算法：包含引导、流控与避障"""

    def __init__(self, config):
        self.cfg = config

    def get_velocity(self, agent_id, positions, env):
        pos = positions[agent_id]
        px = pos[0]

        # --- 策略 1: 分段目标引导 ---
        if px < 5.0:
            target = np.array([5.5, -0.75])  # 入口引导点
        elif px < 25.0:
            target = np.array([26.0, -0.75])  # 通道引导点
        else:
            # 最终圆形编队目标点
            angle = 2 * np.pi * agent_id / self.cfg.n_robots
            target = self.cfg.formation_center + np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle)])

        v_des = (target - pos) * 1.8

        # --- 策略 2: 机器人间流控与防撞 ---
        for i in range(self.cfg.n_robots):
            if i == agent_id: continue
            dist_vec = positions[i] - pos
            dist = np.linalg.norm(dist_vec)

            # 相互排斥（人工势场）
            if dist < 0.75:
                v_des -= (dist_vec / (dist + 1e-2)) * (0.75 - dist) * 20.0

            # 速度协调（防止追尾）
            if positions[i][0] > px and dist < 0.8:
                v_des[0] = min(v_des[0], 0.3)

        # --- 策略 3: 环境避障 ---
        for element in env.walls + [env.obs]:
            r = element['rect']
            closest = np.clip(pos, [r[0], r[1]], [r[0] + r[2], r[1] + r[3]])
            diff = pos - closest
            d = np.linalg.norm(diff)
            if d < 0.55:
                # 产生指向障碍物外部的斥力
                v_des += (diff / (d + 1e-3)) * (1.2 / max(d - 0.2, 0.01))

        # 速度幅值限幅
        v_mag = np.linalg.norm(v_des)
        if v_mag > self.cfg.max_v:
            v_des = (v_des / v_mag) * self.cfg.max_v

        return v_des


# ==================== 3. Simulation & Export ====================
def main():
    cfg = SimConfig()
    env = Environment()
    ctrl = RobotController(cfg)

    # 初始布局：起始圆形编队
    positions = np.array([[2.5 * np.cos(2 * np.pi * i / cfg.n_robots),
                           2.5 * np.sin(2 * np.pi * i / cfg.n_robots)] for i in range(cfg.n_robots)])

    history, log_data = [], []
    collision_count = 0
    success_time = None
    recovery_time = None  # 初始化队形恢复时间变量

    print("Simulation running...")
    for step in range(int(cfg.max_time / cfg.dt)):
        t = step * cfg.dt
        history.append(positions.copy())

        # 碰撞检测
        min_sep = 10.0
        for i in range(cfg.n_robots):
            for j in range(i + 1, cfg.n_robots):
                d = np.linalg.norm(positions[i] - positions[j])
                min_sep = min(min_sep, d)
                if d < 0.499: collision_count += 1

        log_data.append({'time': t, 'min_sep': min_sep, 'pos': positions.copy()})

        # --- 任务状态检查 ---
        # 1. 通道穿越检查
        if np.all(positions[:, 0] > 25.0) and success_time is None:
            success_time = t
            print(f"Pass channel success: {t:.2f}s")

        # 2. 队形恢复检查
        if success_time is not None and recovery_time is None:
            errors = []
            for i in range(cfg.n_robots):
                angle = 2 * np.pi * i / cfg.n_robots
                target = cfg.formation_center + np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle)])
                errors.append(np.linalg.norm(positions[i] - target))

            if np.all(np.array(errors) < cfg.recovery_threshold):
                recovery_time = t
                print(f"Formation fully recovered: {t:.2f}s")

        # 动力学更新
        new_pos = np.zeros_like(positions)
        for i in range(cfg.n_robots):
            new_pos[i] = positions[i] + ctrl.get_velocity(i, positions, env) * cfg.dt
        positions = new_pos

        # 终止条件：队形恢复后再运行5秒，或者达到最大时长
        if recovery_time and (t > recovery_time + 5.0): break

    # --- 数据导出 ---
    path = "./output_results"  # 建议使用相对路径提高兼容性
    if not os.path.exists(path): os.makedirs(path)

    # 结果 CSV
    with open(os.path.join(path, "result_data.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'min_sep'] + [f'r{i}_{c}' for i in range(cfg.n_robots) for c in ['x', 'y']])
        for d in log_data:
            writer.writerow([d['time'], d['min_sep']] + d['pos'].flatten().tolist())

    # 报告 TXT
    with open(os.path.join(path, "simulation_report.txt"), 'w') as f:
        f.write(f"Pass Channel Time: {success_time if success_time else 'FAIL'} s\n")
        f.write(f"Formation Recovery Time: {recovery_time if recovery_time else 'N/A'} s\n")
        f.write(f"Total Collision Counts: {collision_count}\n")
        f.write(f"Global Minimum Distance: {min(d['min_sep'] for d in log_data):.4f} m\n")

    # --- 动画制作 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-5, 40);
    ax.set_ylim(-4, 4);
    ax.set_aspect('equal')
    ax.set_title("Multi-Robot Formation Control Simulation (1x Speed)", fontsize=13, fontweight='bold')

    # 环境绘制
    ax.add_patch(patches.Rectangle((5, 1), 20, 0.5, color='gray', alpha=0.6, label='Wall'))
    ax.add_patch(patches.Rectangle((5, -1.5), 20, 0.5, color='gray', alpha=0.6))
    ax.add_patch(patches.Rectangle((7, 0.2), 2, 0.8, color='red', alpha=0.7, label='Obstacle'))

    robots = [patches.Circle((0, 0), cfg.robot_radius, fc=plt.cm.tab10(i / cfg.n_robots), ec='k', zorder=10) for i in
              range(cfg.n_robots)]
    for r in robots: ax.add_patch(r)

    time_label = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontweight='bold')

    # 配置 1倍速演示 (1x Real-time)
    frame_skip = 5
    real_time_interval = frame_skip * cfg.dt * 1000  # 这里计算出 100ms
    actual_fps = 1 / (frame_skip * cfg.dt)  # 这里计算出 10 FPS 以保证 1x 速

    def update(frame_idx):
        frame = frame_idx * frame_skip
        if frame >= len(history): frame = len(history) - 1
        for i, r in enumerate(robots):
            r.center = history[frame][i]
        time_label.set_text(f"Real-time: {frame * cfg.dt:.2f} s")
        return robots + [time_label]

    ani = FuncAnimation(fig, update, frames=range(len(history) // frame_skip),
                        interval=real_time_interval, blit=True)

    # 保存动画
    save_path = os.path.join(path, "simulation_1x.gif")
    ani.save(save_path, writer='pillow', fps=actual_fps)
    print(f"Animation saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()