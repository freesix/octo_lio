#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav2_costmap_2d/layer.hpp>
#include <nav2_costmap_2d/layered_costmap.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.hpp>

namespace nav2_costmap_2d{
    

class CollisionLayer : public Layer{
public:
    CollisionLayer();
    virtual ~CollisionLayer();

    void onInitialize() override;

    void updateBounds(
        double robot_x, double robot_y, double robot_yaw, double * min_x,
        double * min_y,
        double * max_x,
        double * max_y) override;
    void updateCosts(
        nav2_costmap_2d::Costmap2D & master_grid,
        int min_i, int min_j, int max_i, int max_j) override;

    virtual void reset()
    {
        return;
    }

    void onFootprintChanged() override;

    virtual bool isClearable() {return true;}

    struct Obstacle{
        double x;
        double y;
        double yaw;
    };

    typedef std::recursive_mutex mutex_t;
    
    mutex_t* getMutex()
    {
        return access_;
    }
    
private:

    void collisionCallback(const std_msgs::msg::Bool::SharedPtr msg);
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_;

    std::vector<Obstacle> obstacle_; // 存储的障碍
    int pending_triggers_;           // 上升沿触发信号
    bool last_collision_state_;      // 检测上升沿

    std::string topic_;
    double front_offest_;            // 在前方多少距离生成障碍物
    double size_x_;                  // 障碍物长
    double size_y_;                  // 障碍物宽
    double leave_distance_;          // 机器人离开距离
    std::string frame_id_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::string global_frame_;

    mutex_t * access_;

};


}