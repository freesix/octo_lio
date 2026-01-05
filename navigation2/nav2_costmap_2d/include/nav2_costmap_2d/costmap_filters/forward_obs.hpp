#pragma once

#include <nav2_costmap_2d/costmap_filters/costmap_filter.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>


namespace nav2_costmap_2d{


class ForwardObs : public CostmapFilter{
public:

ForwardObs();

void initializeFilter(const std::string &filter_info_topic);

void process(nav2_costmap_2d::Costmap2D & master_grid,
             int min_i, int min_j, int max_i, int max_j,
             const geometry_msgs::msg::Pose2D & pose);

void resetFilter();

bool isActive();


private:
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_sub_;
    double rect_length_ = 0.6;
    double rect_width_ = 0.3;
    double forward_offset_ = 0.35;
    double clear_distance_ = 2.0;

    std::atomic<bool> pending_trigger_{false};
    std::atomic<bool> active_{false};
    double anchor_x_{0.0};
    double anchor_y_{0.0};
    double anchor_yaw_{0.0};

    std::mutex state_mtx_;
    
    void collisionCallback(const std_msgs::msg::Bool::SharedPtr msg);

    void computerAnchorFromPose(const geometry_msgs::msg::Pose2D& pose);

      // 将世界坐标点旋转到以矩形中心为原点、x 沿航向的局部坐标
    inline void worldToRectLocal(double wx, double wy, double cx, double cy, double yaw,
                               double & lx, double & ly)
    {
        const double dx = wx - cx;
        const double dy = wy - cy;
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);
        // 旋转 -yaw：将世界系点转到“矩形局部系”(x 前向, y 左向)
        lx =  c * dx + s * dy;
        ly = -s * dx + c * dy;
    }

};

};