#! /usr/bin/env python3
# encoding: utf-8
"""
三路穿越：
- 中间3个机器人：沿垂直平分线直线穿过
- 最左边1个：绕左边走
- 最右边1个：绕右边走
"""

import rospy
import numpy as np
from gazebo_swarm_robot_control import SwarmRobot

# 参数
KP = 0.3
SPACING = 0.5
SAFE_DIST = 0.6
REPULSION = 0.6

# 障碍物参数
OBS_SAFE_DIST = 0.4
OBS_REPULSION = 0.4

# 绕行参数
DETOUR_OFFSET = 0.4  # 左右绕行的偏移距离


def get_pass_direction(obs_robot, swarm_robot):
    """计算穿越方向（垂直平分线）"""
    obs_poses = obs_robot.get_robot_poses()
    swarm_poses = swarm_robot.get_robot_poses()
    
    obs1 = np.array([obs_poses[0][0], obs_poses[0][1]])
    obs2 = np.array([obs_poses[1][0], obs_poses[1][1]])
    
    rospy.loginfo(f"障碍物1位置: ({obs1[0]:.2f}, {obs1[1]:.2f})")
    rospy.loginfo(f"障碍物2位置: ({obs2[0]:.2f}, {obs2[1]:.2f})")
    
    center = (obs1 + obs2) / 2.0
    rospy.loginfo(f"缝隙中心: ({center[0]:.2f}, {center[1]:.2f})")
    
    direction = obs2 - obs1
    perp = np.array([-direction[1], direction[0]])
    perp = perp / np.linalg.norm(perp)
    
    swarm_center = sum(np.array([p[0], p[1]]) for p in swarm_poses) / len(swarm_poses)
    rospy.loginfo(f"移动机器人质心: ({swarm_center[0]:.2f}, {swarm_center[1]:.2f})")
    
    to_obstacle = center - swarm_center
    dot_product = np.dot(perp, to_obstacle)
    
    if dot_product < 0:
        perp = -perp
        rospy.loginfo("方向反转！")
    
    rospy.loginfo(f"穿越方向: ({perp[0]:.2f}, {perp[1]:.2f})")
    
    # 返回垂直平分线方向和垂直于它的方向（左右方向）
    lateral_dir = np.array([-perp[1], perp[0]])  # 左右方向
    
    return center, perp, lateral_dir


def assign_groups(swarm_poses, obs_center, lateral_dir):
    """
    分配机器人到三组
    
    返回：
    - left_robot: 最左边机器人的索引
    - middle_robots: 中间3个机器人的索引列表
    - right_robot: 最右边机器人的索引
    """
    n = len(swarm_poses)
    
    # 计算每个机器人在左右方向上的投影
    lateral_projections = []
    for i, pose in enumerate(swarm_poses):
        pos = np.array([pose[0], pose[1]])
        proj = np.dot(pos - obs_center, lateral_dir)
        lateral_projections.append((proj, i))
    
    # 按左右位置排序
    lateral_projections.sort()
    
    # 分配：最左、中间3个、最右
    left_robot = lateral_projections[0][1]
    middle_robots = [lateral_projections[i][1] for i in range(1, 4)]
    right_robot = lateral_projections[4][1]
    
    rospy.loginfo("\n=== 机器人分组 ===")
    rospy.loginfo(f"左路（绕左边）: Robot {left_robot+1}")
    rospy.loginfo(f"中路（直线穿）: Robot {[r+1 for r in middle_robots]}")
    rospy.loginfo(f"右路（绕右边）: Robot {right_robot+1}")
    rospy.loginfo("")
    
    return left_robot, middle_robots, right_robot


def three_path_control(poses, obs_center, pass_dir, lateral_dir, 
                       left_robot, middle_robots, right_robot,
                       target_ahead, adjacency, obs_poses):
    """
    三路穿越控制
    
    参数：
    - target_ahead: 目标点距离（沿穿越方向）
    """
    n = len(poses)
    vel_x = []
    vel_y = []
    
    # 计算中路机器人的质心
    middle_center = sum(np.array([poses[i][0], poses[i][1]]) for i in middle_robots) / len(middle_robots)
    
    # 垂直于前进方向（用于对齐）
    perpendicular = lateral_dir
    
    for i in range(n):
        pos_i = np.array([poses[i][0], poses[i][1]])
        
        if i == left_robot:
            # 左路：绕左边走
            # 前后位置与中间机器人的中间那个对齐（middle_robots[1]）
            middle_center_robot_idx = middle_robots[1]  # 中间那3个的第2个
            middle_center_pos = np.array([poses[middle_center_robot_idx][0], 
                                         poses[middle_center_robot_idx][1]])
            
            # 目标：与中间机器人对齐，但横向偏移到左边
            target_pos = middle_center_pos - DETOUR_OFFSET * lateral_dir  # 往左偏移
            
            force = KP * (target_pos - pos_i)
            
        elif i == right_robot:
            # 右路：绕右边走
            # 前后位置与中间机器人的中间那个对齐
            middle_center_robot_idx = middle_robots[1]  # 中间那3个的第2个
            middle_center_pos = np.array([poses[middle_center_robot_idx][0], 
                                         poses[middle_center_robot_idx][1]])
            
            # 目标：与中间机器人对齐，但横向偏移到右边
            target_pos = middle_center_pos + DETOUR_OFFSET * lateral_dir  # 往右偏移
            
            force = KP * (target_pos - pos_i)
            
        else:  # i in middle_robots
            # 中路：直线穿越，排成纵队
            
            # 1. 前进力
            target_point = obs_center + target_ahead * pass_dir
            forward = KP * (target_point - middle_center)
            
            # 2. 排队力（在中路的3个机器人中的位置）
            middle_rank = middle_robots.index(i)  # 0, 1, 或 2
            offset_along = (1 - middle_rank) * SPACING  # 1, 0, -1 对应前中后
            
            ideal_pos = middle_center + offset_along * pass_dir
            queue_force = KP * 1.2 * (ideal_pos - pos_i)
            
            # 3. 对齐力：保持在垂直平分线上
            lateral_offset = np.dot(pos_i - middle_center, perpendicular)
            alignment_force = -KP * 10 * lateral_offset * perpendicular
            
            force = forward + queue_force + alignment_force
        
        # 一致性力（所有机器人）
        consensus = np.zeros(2)
        neighbor_count = 0
        
        for j in range(n):
            if adjacency[i, j] > 0 and i != j:
                pos_j = np.array([poses[j][0], poses[j][1]])
                
                # 期望距离：同组的要保持队形，不同组的适当分开
                if (i in middle_robots and j in middle_robots):
                    # 中路机器人之间
                    rank_i = middle_robots.index(i)
                    rank_j = middle_robots.index(j)
                    desired_dist = (rank_i - rank_j) * SPACING
                    actual_diff = pos_i - pos_j
                    actual_dist = np.dot(actual_diff, pass_dir)
                    error = desired_dist - actual_dist
                    consensus += 0.3 * error * pass_dir
                else:
                    # 不同组之间：轻微协调
                    diff = pos_i - pos_j
                    consensus += 0.1 * diff
                
                neighbor_count += 1
        
        if neighbor_count > 0:
            consensus /= neighbor_count
        
        force += consensus
        
        # 避碰力
        repulsion = np.zeros(2)
        
        # 与其他机器人避碰
        for j in range(n):
            if i == j:
                continue
            
            pos_j = np.array([poses[j][0], poses[j][1]])
            diff = pos_i - pos_j
            dist = np.linalg.norm(diff)
            
            if dist < SAFE_DIST and dist > 0.01:
                force_mag = REPULSION * (SAFE_DIST - dist) / dist
                repulsion += force_mag * diff
        
        # 与障碍物避碰
        for obs_pose in obs_poses:
            obs_pos = np.array([obs_pose[0], obs_pose[1]])
            diff = pos_i - obs_pos
            dist = np.linalg.norm(diff)
            
            if dist < OBS_SAFE_DIST and dist > 0.01:
                force_mag = OBS_REPULSION * (OBS_SAFE_DIST - dist) / (dist ** 2)
                repulsion += force_mag * diff
        
        total = force + repulsion
        
        vel_x.append(total[0])
        vel_y.append(total[1])
    
    return vel_x, vel_y


def create_adjacency(n):
    A = np.ones((n, n)) - np.eye(n)
    return A


def stop_robots(robot):
    robot.move_robots_by_u([0.0]*robot.robot_num, [0.0]*robot.robot_num)


def main():
    rospy.init_node("three_path_pass")
    
    swarm_robot = SwarmRobot([1, 2, 3, 4, 5])
    obs_robot = SwarmRobot([6, 7])
    
    n = swarm_robot.robot_num
    A = create_adjacency(n)
    
    rospy.sleep(1.0)
    
    rospy.loginfo("=== 三路穿越任务 ===\n")
    
    # 计算穿越方向和左右方向
    swarm_poses = swarm_robot.get_robot_poses()
    obs_center, pass_dir, lateral_dir = get_pass_direction(obs_robot, swarm_robot)
    
    rospy.loginfo(f"左右方向: ({lateral_dir[0]:.2f}, {lateral_dir[1]:.2f})")
    
    # 分配机器人到三组
    left_robot, middle_robots, right_robot = assign_groups(swarm_poses, obs_center, lateral_dir)
    
    # 计算穿越距离
    current_center = sum(np.array([p[0], p[1]]) for p in swarm_poses) / n
    distance_to_obs = np.linalg.norm(current_center - obs_center)
    target_ahead = distance_to_obs * 1.5  # 目标：穿过障碍物后
    
    rospy.loginfo(f"起点质心: ({current_center[0]:.2f}, {current_center[1]:.2f})")
    rospy.loginfo(f"目标距离: {target_ahead:.2f}m")
    rospy.loginfo("\n开始移动...\n")
    
    rate = rospy.Rate(20)
    
    while not rospy.is_shutdown():
        stop_robots(obs_robot)
        
        poses = swarm_robot.get_robot_poses()
        obs_poses = obs_robot.get_robot_poses()
        
        vel_x, vel_y = three_path_control(
            poses, obs_center, pass_dir, lateral_dir,
            left_robot, middle_robots, right_robot,
            target_ahead, A, obs_poses
        )
        
        swarm_robot.move_robots_by_u(vel_x, vel_y)
        
        # 检查中路机器人是否到达
        middle_center = sum(np.array([poses[i][0], poses[i][1]]) for i in middle_robots) / 3
        dist_to_target = np.linalg.norm(middle_center - (obs_center + target_ahead * pass_dir))
        
        if dist_to_target < 0.5:
            rospy.loginfo("\n✓ 穿越完成！")
            break
        
        rate.sleep()
    
    stop_robots(swarm_robot)
    
    # 显示最终结果
    poses = swarm_robot.get_robot_poses()
    
    rospy.loginfo("\n=== 任务完成 ===")
    rospy.loginfo(f"左路 Robot {left_robot+1}: ({poses[left_robot][0]:.2f}, {poses[left_robot][1]:.2f})")
    for i in middle_robots:
        rospy.loginfo(f"中路 Robot {i+1}: ({poses[i][0]:.2f}, {poses[i][1]:.2f})")
    rospy.loginfo(f"右路 Robot {right_robot+1}: ({poses[right_robot][0]:.2f}, {poses[right_robot][1]:.2f})")


if __name__ == "__main__":
    main()