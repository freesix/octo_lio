// motor_node.hpp
// 声明 MotorNode 类（从 v1 的 main.cpp 中拆分）

#ifndef MOTOR_DRIVE_V2_MOTOR_NODE_HPP
#define MOTOR_DRIVE_V2_MOTOR_NODE_HPP

// 仅在头文件中做最小依赖，避免对外暴露 ROS 头
#include <memory>
#include <map>
#include <mutex>
#include <chrono>
#include <atomic>
#include <deque>
#include <boost/lockfree/spsc_queue.hpp>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <std_srvs/srv/empty.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
class MotorNode : public rclcpp::Node {
public:
    explicit MotorNode(std::shared_ptr<class MotorDrive> motor);

    void Run();

    // 将控制命令写入队列的接口
    void enqueueVelocityCmd(const geometry_msgs::msg::Twist& cmd);

    // 读取编码器并计算里程计（由 ROS2 定时器回调驱动）
    void odomTimerCallback();

    // 供外部线程弹出一条待发送速度指令（若有）
    bool tryPopCmd(geometry_msgs::msg::Twist& out_cmd);

    // 供外部线程写入一条编码器历史，时间戳单位：纳秒
    void recordEncoderSample(int64_t stamp_ns, int left_encoder, int right_encoder);

    // 串口读写线程函数：用于在外部通过 std::thread 启动
    void ioThreadFunc();

private:
    // 依赖的电机驱动智能指针
    std::shared_ptr<class MotorDrive> motor_;

    // 队列元素：写/读区分；写时携带 Twist
    struct CommandItem {
        bool is_write; // true: 写速度命令；false: 读请求（里程计）
        geometry_msgs::msg::Twist cmd; // 仅当 is_write=true 有效
    };

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_vel;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_pub;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_odom_srv_;

    // 速度历史：key 为时间戳(纳秒)，value 为速度命令；仅保留最近15条
    std::map<int64_t, geometry_msgs::msg::Twist> velocity_history_;
    // 编码器历史：使用 SPSC 无锁环形缓冲（固定容量）
    using EncoderSample = std::pair<int64_t, std::pair<int, int>>;
    boost::lockfree::spsc_queue<EncoderSample, boost::lockfree::capacity<4096>> encoder_history_;
    // 写入到底盘的待发送命令队列（按时间顺序，仅存新命令）
    std::deque<geometry_msgs::msg::Twist> cmd_queue_;
    std::mutex cmd_queue_mutex_;

    // 记录上一帧速度命令（用于去重发送）
    bool last_cmd_valid_ = false;
    geometry_msgs::msg::Twist last_velocity_cmd_{};
    
    // 记录上一帧使用的编码器值（用于速度积分）
    bool prev_encoder_valid_ = false;
    int prev_left_encoder_ = 0;
    int prev_right_encoder_ = 0;
    
};

#endif // MOTOR_DRIVE_V2_MOTOR_NODE_HPP


