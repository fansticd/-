#! /usr/bin/env python3
# encoding: utf-8
"""
完整版本：包含对障碍物的排斥力
"""

import rospy
import numpy as np
from gazebo_swarm_robot_control import SwarmRobot

# 参数
KP = 0.3
SPACING = 0.5
SAFE_DIST = 0.6          # 机器人间安全距离
REPULSION = 0.6          # 机器人间排斥力

# 障碍物参数
OBS_SAFE_DIST = 0.4      # 障碍物安全距离（更大！）
OBS_REPULSION = 0.4      # 障碍物排斥力（更强！）

def get_pass_direction(obs_robot, swarm_robot):
    """
    计算穿越方向（垂直平分线）
    同时判断应该往哪边穿
    """
    obs_poses = obs_robot.get_robot_poses()
    swarm_poses = swarm_robot.get_robot_poses()
    
    # 障碍物位置
    obs1 = np.array([obs_poses[0][0], obs_poses[0][1]])
    obs2 = np.array([obs_poses[1][0], obs_poses[1][1]])
    
    rospy.loginfo(f"障碍物1位置: ({obs1[0]:.2f}, {obs1[1]:.2f})")
    rospy.loginfo(f"障碍物2位置: ({obs2[0]:.2f}, {obs2[1]:.2f})")
    
    # 中点（缝隙中心）
    center = (obs1 + obs2) / 2.0
    rospy.loginfo(f"缝隙中心: ({center[0]:.2f}, {center[1]:.2f})")
    
    # 障碍物连线方向
    direction = obs2 - obs1
    rospy.loginfo(f"障碍物连线方向: ({direction[0]:.2f}, {direction[1]:.2f})")
    
    # 垂直方向（逆时针旋转90度）
    perp = np.array([-direction[1], direction[0]])
    perp = perp / np.linalg.norm(perp)
    rospy.loginfo(f"垂直方向1: ({perp[0]:.2f}, {perp[1]:.2f})")
    
    # 计算移动机器人的质心
    swarm_center = sum(np.array([p[0], p[1]]) for p in swarm_poses) / len(swarm_poses)
    rospy.loginfo(f"移动机器人质心: ({swarm_center[0]:.2f}, {swarm_center[1]:.2f})")
    
    # 判断机器人在障碍物的哪一边
    to_obstacle = center - swarm_center
    dot_product = np.dot(perp, to_obstacle)
    
    if dot_product < 0:
        perp = -perp
        rospy.loginfo("方向反转！")
    
    rospy.loginfo(f"最终穿越方向: ({perp[0]:.2f}, {perp[1]:.2f})")
    
    return center, perp


def simple_control(poses, target_point, target_direction, adjacency, obs_poses, fixed_rank):
    """
    简单控制：
    1. 往目标点前进
    2. 排成纵队（沿前进方向，垂直方向对齐成一条直线）
    3. 避碰（机器人之间 + 障碍物）
    
    参数：
    - fixed_rank: 固定的排名列表，在初始时刻根据到障碍物距离计算一次后不再改变
    """
    n = len(poses)
    vel_x = []
    vel_y = []
    
    # 当前质心
    center = sum(np.array([p[0], p[1]]) for p in poses) / n
    
    # 垂直于前进方向的向量（用于对齐）
    perpendicular = np.array([-target_direction[1], target_direction[0]])
    
    # 使用固定排名（初始时刻计算，后续不变）
    rank = fixed_rank
    
    for i in range(n):
        pos_i = np.array([poses[i][0], poses[i][1]])
        
        # 1. 前进力：整体往目标点移动
        to_target = target_point - center
        forward = KP * to_target
        
        # 2. 纵队排列力
        # rank=0 应该在最前面（沿前进方向，正offset）
        # rank=n-1 应该在最后面（沿前进方向，负offset）
        # 所以要反转公式
        offset_along = ((n-1)/2.0 - rank[i]) * SPACING
        offset_perp = 0.0
        
        ideal_pos = center + offset_along * target_direction + offset_perp * perpendicular
        queue_force = KP * 1.2 * (ideal_pos - pos_i)
        
        # 3. 对齐力：强制所有机器人在垂直方向上对齐
        relative_to_center = pos_i - center
        lateral_offset = np.dot(relative_to_center, perpendicular)
        alignment_force = -KP * 10 * lateral_offset * perpendicular
        
        # 4. 一致性：和邻居保持距离（仅在前进方向上）
        consensus = np.zeros(2)
        neighbor_count = 0
        
        for j in range(n):
            if adjacency[i, j] > 0:
                pos_j = np.array([poses[j][0], poses[j][1]])
                desired_distance = (rank[i] - rank[j]) * SPACING
                
                actual_diff = pos_i - pos_j
                actual_distance = np.dot(actual_diff, target_direction)
                
                error = desired_distance - actual_distance
                consensus += 0.3 * error * target_direction
                neighbor_count += 1
        
        if neighbor_count > 0:
            consensus /= neighbor_count
        
        # 5. 避碰力
        repulsion = np.zeros(2)
        
        # 5a. 与其他机器人的排斥
        for j in range(n):
            if i == j:
                continue
            
            pos_j = np.array([poses[j][0], poses[j][1]])
            diff = pos_i - pos_j
            dist = np.linalg.norm(diff)
            
            if dist < SAFE_DIST and dist > 0.01:
                force = REPULSION * (SAFE_DIST - dist) / dist
                repulsion += force * diff
        
        # 5b. 与障碍物的排斥（重要！）
        for obs_pose in obs_poses:
            obs_pos = np.array([obs_pose[0], obs_pose[1]])
            diff = pos_i - obs_pos
            dist = np.linalg.norm(diff)
            
            if dist < OBS_SAFE_DIST and dist > 0.01:
                # 距离障碍物越近，推力越强（平方反比）
                force = OBS_REPULSION * (OBS_SAFE_DIST - dist) / (dist ** 2)
                repulsion += force * diff
                
                # 调试：打印障碍物排斥信息
                if dist < 0.7:
                    rospy.loginfo_throttle(1.0, 
                        f"Robot {i+1} 距障碍物 {dist:.2f}m, 排斥力: {force:.2f}")
        
        # 合成：前进 + 排队 + 对齐 + 一致性 + 避碰
        total = forward + queue_force + alignment_force + consensus + repulsion
        
        vel_x.append(total[0])
        vel_y.append(total[1])
    
    return vel_x, vel_y


def create_adjacency(n):
    A = np.ones((n, n)) - np.eye(n)
    return A


def stop_robots(robot):
    robot.move_robots_by_u([0.0]*robot.robot_num, [0.0]*robot.robot_num)


def main():
    rospy.init_node("pass_with_obs_avoid")
    
    swarm_robot = SwarmRobot([1, 2, 3, 4, 5])
    obs_robot = SwarmRobot([6, 7])
    
    n = swarm_robot.robot_num
    A = create_adjacency(n)
    
    rospy.sleep(1.0)
    
    rospy.loginfo("=== 开始穿越任务（带障碍物避障）===\n")
    rospy.loginfo(f"障碍物安全距离: {OBS_SAFE_DIST}m")
    rospy.loginfo(f"障碍物排斥力强度: {OBS_REPULSION}\n")
    
    # 计算穿越方向
    obs_center, pass_dir = get_pass_direction(obs_robot, swarm_robot)
    
    # ===== 初始排名计算（固定不变）=====
    swarm_poses = swarm_robot.get_robot_poses()
    
    # 计算每个机器人在垂直平分线上的投影
    initial_projections = []
    for i, pose in enumerate(swarm_poses):
        pos = np.array([pose[0], pose[1]])
        # 投影到穿越方向：点积越大说明越靠近目标方向
        proj = np.dot(pos - obs_center, pass_dir)
        initial_projections.append((proj, i))
    
    # 按投影排序：投影大的（靠近目标）排前面
    initial_projections.sort(reverse=True)
    
    # 生成固定排名映射
    fixed_rank = [0] * n
    for rank_position, (_, robot_idx) in enumerate(initial_projections):
        fixed_rank[robot_idx] = rank_position
    
    rospy.loginfo("初始队列排序（固定不变）:")
    for rank_pos, (proj, robot_idx) in enumerate(initial_projections):
        rospy.loginfo(f"  位置{rank_pos}: Robot {robot_idx+1} (投影: {proj:.2f}m)")
    rospy.loginfo("")
    
    # 目标：障碍物另一边
    current_center = sum(np.array([p[0], p[1]]) for p in swarm_poses) / n
    
    # 计算穿越距离
    distance_to_obs = np.linalg.norm(current_center - obs_center)
    target_point = obs_center + distance_to_obs * pass_dir
    
    rospy.loginfo(f"起点: ({current_center[0]:.2f}, {current_center[1]:.2f})")
    rospy.loginfo(f"障碍物中心: ({obs_center[0]:.2f}, {obs_center[1]:.2f})")
    rospy.loginfo(f"目标点: ({target_point[0]:.2f}, {target_point[1]:.2f})")
    rospy.loginfo("\n开始移动...\n")
    
    rate = rospy.Rate(20)
    
    while not rospy.is_shutdown():
        stop_robots(obs_robot)
        
        poses = swarm_robot.get_robot_poses()
        obs_poses = obs_robot.get_robot_poses()  # 获取障碍物位置
        
        vel_x, vel_y = simple_control(poses, target_point, pass_dir, A, obs_poses, fixed_rank)
        swarm_robot.move_robots_by_u(vel_x, vel_y)
        
        # 检查是否到达
        center = sum(np.array([p[0], p[1]]) for p in poses) / n
        dist_to_target = np.linalg.norm(center - target_point)
        
        if dist_to_target < 0.3:
            rospy.loginfo("\n✓ 穿越完成！")
            break
        
        rate.sleep()
    
    stop_robots(swarm_robot)
    
    # 显示最终结果
    poses = swarm_robot.get_robot_poses()
    final_center = sum(np.array([p[0], p[1]]) for p in poses) / n
    
    rospy.loginfo("\n=== 任务完成 ===")
    rospy.loginfo(f"最终质心: ({final_center[0]:.2f}, {final_center[1]:.2f})")
    rospy.loginfo("\n各机器人位置:")
    for i in range(n):
        rospy.loginfo(f"  Robot {i+1}: ({poses[i][0]:.2f}, {poses[i][1]:.2f})")


if __name__ == "__main__":
    main()