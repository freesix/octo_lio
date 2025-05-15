#pragma once 

#include <message_filters/subscriber.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

typedef tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2> MessageFilter;

class point2scan : public rclcpp::Node {

public:
    explicit point2scan(const rclcpp::NodeOptions &options);
    ~point2scan() override;

    
private:
    
    std::unique_ptr<tf2_ros::Buffer> tf2_;
    std::unique_ptr<tf2_ros::TransformListener> tf2_listener_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
    std::unique_ptr<MessageFilter> message_filter_;

    std::thread sub_listener_thread_;
    std::atomic_bool alive_{true};

    int input_queue_size_;
    std::string target_frame_;
    std::string sub_topic_;
    std::string scan_topic_;
    double min_height_, max_height_, angle_min_, angle_max_, angle_increment_, scan_time_, range_min_, range_max_;
    bool use_inf_;
    double inf_epsilon_;
    double tolerance_;

    void cloudCallBack(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg);
    void subscriptionListenerThreadLoop();
    void getparams(){
        target_frame_ = this->declare_parameter("target_frame", "");
        sub_topic_ = this->declare_parameter("sub_topic", "");
        scan_topic_ = this->declare_parameter("scan_topic", "");
        tolerance_ = this->declare_parameter("transform_tolerance", 0.01);
        input_queue_size_ = this->declare_parameter("queue_size", static_cast<int>(std::thread::hardware_concurrency())); // 获取硬件核心
        min_height_ = this->declare_parameter("min_height", std::numeric_limits<double>::min());
        max_height_ = this->declare_parameter("max_height", std::numeric_limits<double>::max());
        angle_min_ = this->declare_parameter("angle_min", -M_PI_2);
        angle_max_ = this->declare_parameter("angle_max", M_PI_2);
        angle_increment_ = this->declare_parameter("angle_increment", M_PI/180.0);
        scan_time_ = this->declare_parameter("scan_time", 1.0/10.0);
        range_min_ = this->declare_parameter("range_min", 0.0);
        range_max_ = this->declare_parameter("range_max", std::numeric_limits<double>::max());
        inf_epsilon_ = this->declare_parameter("inf_epsilon", 1.0);
        use_inf_ = this->declare_parameter("use_inf", true);
    }

};