// v2 main.cpp：仅负责初始化并运行 MotorNode
#include <memory>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include "../include/motor_drive_v2/motor_node.hpp"
#include "../include/motor_drive_v2/motor_drive.hpp"

int main(int argc, char** argv) {
    // 初始化ROS 2
    rclcpp::init(argc, argv);
    
    auto motor = std::make_shared<MotorDrive>();
    auto node = std::make_shared<MotorNode>(motor);
    node->Run();
    
    // 用ROS2定时器驱动里程计计算
    
    auto odom_timer = node->create_wall_timer(
        std::chrono::milliseconds(16), // 60Hz
        [node]() { node->odomTimerCallback(); }
    );
    
    std::thread io_thread([node]() { node->ioThreadFunc(); });

    // 启动ROS2节点
    rclcpp::spin(node);

    // 发送零速度指令，确保退出前底盘停止
    {
        MotorDrive::RobotVelocity zero_vel{};
        zero_vel.linear_vel_x = 0.0f;
        zero_vel.linear_vel_y = 0.0f;
        zero_vel.angular_vel_z = 0.0f;
        motor->setRobotVelocity(zero_vel);
    }

    // 退出时回收IO线程
    if (io_thread.joinable()) {
        io_thread.join();
    }
    // 定时器由rclcpp管理，无需手动join
    
    // 关闭ROS 2
    rclcpp::shutdown();
    
    return 0;
}

