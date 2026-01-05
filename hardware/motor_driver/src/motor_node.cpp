// motor_node.cpp
// 直接让 MotorNode 继承自 rclcpp::Node

#include "../include/motor_drive_v2/motor_node.hpp"
#include "motor_drive_v2/motor_drive.hpp"

MotorNode::MotorNode(std::shared_ptr<MotorDrive> motor) : rclcpp::Node("robot_motor"), motor_(std::move(motor)) {
    this->declare_parameter<std::string>("serial_port_name", "/dev/ttyUSB0");
    this->declare_parameter<int>("serial_baudrate", 115200);
    this->declare_parameter<double>("right_wheel_radius", 0.0845);
    this->declare_parameter<double>("left_wheel_radius", 0.0845);
    this->declare_parameter<double>("wheel_distance", 0.489);
    this->declare_parameter<int>("encoder_resolution", 5600);
    this->declare_parameter<bool>("begin_lost_pose", false);
    this->declare_parameter<std::string>("odom_frame_id", "odom");
    this->declare_parameter<std::string>("base_frame_id", "base_link");

    this->declare_parameter<std::string>("velocity_cmd_topic", "velocity_cmd");
    sub_vel = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
            if(!msg) return;
            enqueueVelocityCmd(*msg);
        }
    );
    velocity_pub = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
        "/vel_2", 10);
    this->declare_parameter<std::string>("odom_publisher", "odom_diff");
    odom_pub = this->create_publisher<nav_msgs::msg::Odometry>(
        this->get_parameter("odom_publisher").as_string(), 10);
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    reset_odom_srv_ = this->create_service<std_srvs::srv::Empty>(
        "reset_odometry",
        [this](const std_srvs::srv::Empty::Request::SharedPtr /*req*/,
               std_srvs::srv::Empty::Response::SharedPtr /*res*/) {
            if(motor_) motor_->resetOdometry();
            RCLCPP_INFO(this->get_logger(), "Odometry reset to zero");
        }
    );
}

void MotorNode::odomTimerCallback() {
    if (!motor_) return;

    nav_msgs::msg::Odometry odom_msg;
    const int64_t t_now_ns = this->now().nanoseconds();

    bool used_encoder = false;
    int left_encoder = 0, right_encoder = 0;
    {
        MotorNode::EncoderSample sample;
        if (encoder_history_.pop(sample)) {
            left_encoder = sample.second.first;
            right_encoder = sample.second.second;
            used_encoder = true;
        }
    }

    if (used_encoder) {
        // RCLCPP_INFO(this->get_logger(), "里程计使用编码器解码");
        odom_msg = motor_->decoder(left_encoder, right_encoder);
        
        // 记录这一帧使用的编码器值
        {
            std::lock_guard<std::mutex> lock(cmd_queue_mutex_);
            prev_left_encoder_ = left_encoder;
            prev_right_encoder_ = right_encoder;
            prev_encoder_valid_ = true;
        }
    } else {
        // 回退到速度积分，使用上一帧的编码器数值
        // RCLCPP_INFO(this->get_logger(), "里程计使用速度积分");
        {
            std::lock_guard<std::mutex> lock(cmd_queue_mutex_);
            if (prev_encoder_valid_) {
                // 使用上一帧记录的编码器值
                left_encoder = prev_left_encoder_;
                right_encoder = prev_right_encoder_;
            } else {
                return;
            }
        }
        odom_msg = motor_->decoder(left_encoder, right_encoder);
    }

    if (odom_pub) {
        odom_msg.header.stamp = this->now();
        odom_msg.header.frame_id = this->get_parameter("odom_frame_id").as_string();
        odom_msg.child_frame_id = this->get_parameter("base_frame_id").as_string();
        odom_pub->publish(odom_msg);
    }
    // 发布速度消息
    if (velocity_pub && motor_) {
        geometry_msgs::msg::TwistWithCovarianceStamped velocity_msg;
        velocity_msg.header.stamp = this->now();
        velocity_msg.header.frame_id = this->get_parameter("base_frame_id").as_string();
        
        geometry_msgs::msg::Twist twist_msg;
        if (motor_->getCurrentVelocity(twist_msg)) {
            velocity_msg.twist.twist = twist_msg;
            // 设置协方差矩阵（6x6，对应linear.x, linear.y, linear.z, angular.x, angular.y, angular.z）
            velocity_msg.twist.covariance[0] = 0.015;  // linear.x
            velocity_msg.twist.covariance[7] = 0.025;  // linear.y
            velocity_msg.twist.covariance[14] = 0.00; // linear.z
            velocity_msg.twist.covariance[21] = 0.025; // angular.x
            velocity_msg.twist.covariance[28] = 0.025; // angular.y
            velocity_msg.twist.covariance[35] = 0.025; // angular.z
            //velocity_pub->publish(velocity_msg);
        }
    }
    if (tf_broadcaster) {
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = this->now();
        transform_stamped.header.frame_id = this->get_parameter("odom_frame_id").as_string();
        transform_stamped.child_frame_id = this->get_parameter("base_frame_id").as_string();
        transform_stamped.transform.translation.x = odom_msg.pose.pose.position.x;
        transform_stamped.transform.translation.y = odom_msg.pose.pose.position.y;
        transform_stamped.transform.translation.z = odom_msg.pose.pose.position.z;
        transform_stamped.transform.rotation = odom_msg.pose.pose.orientation;
        //tf_broadcaster->sendTransform(transform_stamped);
    }
}

void MotorNode::Run() {
    if(motor_) {
        // 先读取参数到本地变量，打印确认
        const std::string port = this->get_parameter("serial_port_name").as_string();
        const int baud = 115200;
        RCLCPP_INFO(this->get_logger(), "Serial params: port=%s, baud=%d", port.c_str(), baud);

        // 串口初始化重试逻辑（可被 Ctrl+C 中断）
        bool serial_initialized = false;
        int retry_count = 0;
        const int max_retries = 10; // 最多重试10次
        const int retry_delay_ms = 2000; // 每次重试间隔2秒

        while(rclcpp::ok() && !serial_initialized && retry_count < max_retries) {
            RCLCPP_INFO(this->get_logger(), "Attempting to initialize serial port (attempt %d/%d)...", 
                        retry_count + 1, max_retries);

            serial_initialized = motor_->serial_init(port, baud);

            if(!serial_initialized) {
                retry_count++;
                if(retry_count < max_retries && rclcpp::ok()) {
                    RCLCPP_WARN(this->get_logger(), "Serial init failed, retrying in %d seconds...", 
                                retry_delay_ms / 1000);
                    // 可被打断的短周期睡眠
                    const int step_ms = 100;
                    int waited = 0;
                    while(rclcpp::ok() && waited < retry_delay_ms) {
                        rclcpp::sleep_for(std::chrono::milliseconds(step_ms));
                        waited += step_ms;
                    }
                }
            }
        }

        // 若被 Ctrl+C 中断，直接返回
        if(!rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "Shutdown requested during serial init; aborting Run()");
            return;
        }

        if(!serial_initialized) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize serial port after %d attempts, aborting!", max_retries);
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Serial port initialized successfully, proceeding with motor setup...");

        // 串口初始化成功后，进行电机初始化
        const double wheel_distance_ = this->get_parameter("wheel_distance").as_double();
        RCLCPP_INFO(this->get_logger(), "wheel_distance=%f", wheel_distance_);
        motor_->motor_init(this->get_parameter("right_wheel_radius").as_double(),
                           this->get_parameter("left_wheel_radius").as_double(),
                           wheel_distance_,
                           this->get_parameter("encoder_resolution").as_int());

        // 启用电机
        if(motor_->enable_motor()) {
            RCLCPP_INFO(this->get_logger(), "Motor enabled successfully");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to enable motor");
        }
        
        // 发送零速度指令
        MotorDrive::RobotVelocity ZeroVel;
        ZeroVel.linear_vel_x = 0.0;
        ZeroVel.angular_vel_z = 0.0;
        motor_->setRobotVelocity(ZeroVel);
        
        RCLCPP_INFO(this->get_logger(), "Motor setup completed successfully");
    }
}

void MotorNode::enqueueVelocityCmd(const geometry_msgs::msg::Twist& cmd) {
    {
        std::lock_guard<std::mutex> lock(cmd_queue_mutex_);
        last_velocity_cmd_ = cmd;
        last_cmd_valid_ = true;
    }
}

bool MotorNode::tryPopCmd(geometry_msgs::msg::Twist& out_cmd) {
    std::lock_guard<std::mutex> lock(cmd_queue_mutex_);
    if (!last_cmd_valid_) {
        return false;
    }
    out_cmd = last_velocity_cmd_;
    last_cmd_valid_ = false; // 取走一次，等待新的最新值
    return true;
}

void MotorNode::recordEncoderSample(int64_t stamp_ns, int left_encoder, int right_encoder) {
    MotorNode::EncoderSample sample{stamp_ns, {left_encoder, right_encoder}};
    (void)encoder_history_.push(sample); // 若满则丢弃该样本
}

void MotorNode::ioThreadFunc() {
    if (!motor_) return;
    while (rclcpp::ok()) {
        geometry_msgs::msg::Twist cmd;
        bool has_cmd = tryPopCmd(cmd);
        if (has_cmd) {
            auto msg_ptr = std::make_shared<geometry_msgs::msg::Twist>(cmd);
            motor_->cmdCallback(msg_ptr);//写入
        }
        
        int left = 0, right = 0;
        int ret = motor_->getEncoderInformation(left, right);//读
        if (ret >= 0) {
            int64_t now_ns = this->now().nanoseconds();
            recordEncoderSample(now_ns, left, right);
        }
        rclcpp::sleep_for(std::chrono::milliseconds(2));
    }
}
