import numpy as np
import rospy
from swarm_robot_control import SwarmRobot


def main():
    # 初始化节点
    rospy.init_node("swarm_robot_straight_line")
    # 机器人的id
    index = [1, 5, 6, 7, 8, 9]
    # 建立对象
    swarm_robot = SwarmRobot(index)


    conv_th = 0.1  # 收敛阈值
    MAX_W = 1.0  # 最大角速度 (rad/s)
    MIN_W = 0.05  # 最小角速度 (rad/s)
    MAX_V = 0.2  # 最大线速度 (m/s)
    MIN_V = 0.01  # 最小线速度 (m/s)
    k_w = 0.5  # 角度控制增益
    k_p = 0.3  # 位置控制增益
    d = 0.5  # 期望间距 (m)



    # 定义图结构，链式拓扑或完全图
    lap_heading = np.array([
        [1, -1, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0],
        [0, -1, 2, -1, 0, 0],
        [0, 0, -1, 2, -1, 0],
        [0, 0, 0, -1, 2,-1],
        [0, 0, 0, 0, -1, 1]
    ])
    # lap_heading = np.array([
    #     [5, -1, -1, -1, -1, -1],
    #     [-1, 5, -1, -1, -1, -1],
    #     [-1, -1, 5, -1, -1, -1],
    #     [-1, -1, -1, 5, -1, -1],
    #     [-1, -1, -1, -1, 5, -1],
    #     [-1, -1, -1, -1, -1, 5]
    # ])

    is_conv = False
    while not is_conv:
        current_pose = swarm_robot.get_robot_poses()
        thetas = np.array([p[2] for p in current_pose])

        # 计算角度误差
        del_theta = -np.dot(lap_heading, thetas)
        is_conv = np.all(np.abs(del_theta) <= conv_th)

        # 控制角速度
        for i in range(swarm_robot.robot_num):
            w = k_w * del_theta[i]
            w = swarm_robot.check_vel(w, MAX_W, MIN_W)
            swarm_robot.move_robot(i, 0, w)

        rospy.sleep(0.05)

    # 停止旋转
    swarm_robot.stop_robots()
    rospy.loginfo("Heading consensus achieved!")



if __name__ == "__main__":
        main()