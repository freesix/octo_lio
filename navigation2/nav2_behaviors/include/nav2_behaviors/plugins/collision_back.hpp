#pragma once

#include <chrono>
#include "nav2_behaviors/timed_behavior.hpp"
#include "nav2_msgs/action/back_up.hpp"
#include "nav2_util/node_utils.hpp"


namespace nav2_behaviors
{

class CollisionBack : public TimedBehavior<nav2_msgs::action::BackUp>
{
public:

    CollisionBack();

    ~CollisionBack() = default;

    Status onRun(const std::shared_ptr<const nav2_msgs::action::BackUp::Goal> command) override;

    Status onCycleUpdate() override;

    void onCleanup() override;

    void onActionCompletion() override;

private:

    void onConfigure() override;

    nav2_msgs::action::BackUp::Feedback::SharedPtr feedback_;

    geometry_msgs::msg::PoseStamped initial_pose_;
    double command_x_;
    double command_speed_;
    rclcpp::Duration command_time_allowance_{0, 0};
    rclcpp::Time end_time_;
    double simulate_ahead_time_;
    double acceleration_limit_;
    double deceleration_limit_;
    double minimum_speed_;
    double last_vel_ = std::numeric_limits<double>::max();
};

}