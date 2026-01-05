#include "nav2_behaviors/plugins/collision_back.hpp"
#include <limits>
#include <memory>
#include <utility>
#include <algorithm>

namespace nav2_behaviors
{

CollisionBack::CollisionBack() :
    TimedBehavior<nav2_msgs::action::BackUp>(),
    feedback_(std::make_shared<nav2_msgs::action::BackUp::Feedback>()),
    command_x_(0.0),
    command_speed_(0.0),
    simulate_ahead_time_(0.0)
{
}

Status CollisionBack::onRun(const std::shared_ptr<const nav2_msgs::action::BackUp::Goal> command){
    if(command->target.y != 0.0 || command->target.z != 0.0){
        RCLCPP_INFO(this->logger_, "DrivingOnHeading in Y and Z not supported, will only move in X.");
        return Status::FAILED;
    }   

    if (!((command->target.x > 0.0) == (command->speed > 0.0)) ) {
      RCLCPP_ERROR(this->logger_, "Speed and command sign did not match");
      return Status::FAILED;
    }

    command_x_ = -std::fabs(command->target.x);
    command_speed_ = -std::fabs(command->speed);
    command_time_allowance_ = command->time_allowance;

    end_time_ = this->clock_->now() + command_time_allowance_;

    if (!nav2_util::getCurrentPose(
        initial_pose_, *this->tf_, this->global_frame_, this->robot_base_frame_,
        this->transform_tolerance_))
    {
      RCLCPP_ERROR(this->logger_, "Initial robot pose is not available.");
      return Status::FAILED;
    }

    return Status::SUCCEEDED;
}


Status CollisionBack::onCycleUpdate(){
    rclcpp::Duration time_remaining = end_time_ - this->clock_->now();
    if(time_remaining.seconds() < 0.0 && command_time_allowance_.seconds() > 0.0){
        this->stopRobot();
        RCLCPP_WARN(this->logger_,
            "Exceeded time allowance before reaching the DriveOnHeading goal - Exiting DriveOnHeading");
        return Status::FAILED;
    }

    geometry_msgs::msg::PoseStamped current_pose;
    if(!nav2_util::getCurrentPose(
        current_pose, *this->tf_, this->global_frame_, this->robot_base_frame_,
        this->transform_tolerance_))
    {
        RCLCPP_ERROR(this->logger_, "Current robot pose is not available.");
        return Status::FAILED;
    }

    double diff_x = initial_pose_.pose.position.x - current_pose.pose.position.x;
    double diff_y = initial_pose_.pose.position.y - current_pose.pose.position.y;
    double distance = hypot(diff_x, diff_y);

    feedback_->distance_traveled = distance;
    this->action_server_->publish_feedback(feedback_);

    if(distance >= std::fabs(command_x_)){
        this->stopRobot();
        return Status::SUCCEEDED;
    }

    auto cmd_vel = std::make_unique<geometry_msgs::msg::Twist>();
    cmd_vel->linear.y = 0.0;
    cmd_vel->angular.z = 0.0;

    bool forward = command_speed_ > 0.0;
    if(acceleration_limit_ == 0.0 || deceleration_limit_ == 0.0) {
        RCLCPP_INFO_ONCE(this->logger_, "DriveOnHeading: no acceleration or deceleration limits set");
        cmd_vel->linear.x = command_speed_;
    }
    else{
        double current_speed = last_vel_ == std::numeric_limits<double>::max() ? 0.0 : last_vel_;
        double min_feasible_speed, max_feasible_speed;
        if(forward){
            min_feasible_speed = current_speed + deceleration_limit_ / this->cycle_frequency_;
            max_feasible_speed = current_speed + acceleration_limit_ / this->cycle_frequency_;
        }
        else{
            min_feasible_speed = current_speed - acceleration_limit_ / this->cycle_frequency_;
            max_feasible_speed = current_speed - deceleration_limit_ / this->cycle_frequency_;
        }
        cmd_vel->linear.x = std::clamp(command_speed_, min_feasible_speed, max_feasible_speed);

        // Check if we need to slow down to avoid overshooting
        auto remaining_distance = std::fabs(command_x_) - distance;
        double max_vel_to_stop = std::sqrt(-2.0 * deceleration_limit_ * remaining_distance);
        if(max_vel_to_stop < std::abs(cmd_vel->linear.x)){
            cmd_vel->linear.x = forward ? max_vel_to_stop : -max_vel_to_stop;
        }
    }

    // Ensure we don't go below minimum speed
    if(std::fabs(cmd_vel->linear.x) < minimum_speed_){
        cmd_vel->linear.x = forward ? minimum_speed_ : -minimum_speed_;
    }

    
    last_vel_ = cmd_vel->linear.x;
    this->vel_pub_->publish(std::move(cmd_vel));

    return Status::RUNNING;
}

void CollisionBack::onCleanup(){
    last_vel_ = std::numeric_limits<double>::max();
}


void CollisionBack::onActionCompletion()
{
    last_vel_ = std::numeric_limits<double>::max();
}


void CollisionBack::onConfigure()
{
    auto node = this->node_.lock();
    if(!node){
        throw std::runtime_error{"Failed to lock node"};
    }

    nav2_util::declare_parameter_if_not_declared(
        node, "simulate_ahead_time", rclcpp::ParameterValue(2.0));
    node->get_parameter("simulate_ahead_time", simulate_ahead_time_);

    nav2_util::declare_parameter_if_not_declared(
        node, this->behavior_name_ + ".acceleration_limit", rclcpp::ParameterValue(0.0));
    nav2_util::declare_parameter_if_not_declared(
        node, this->behavior_name_ + ".deceleration_limit", rclcpp::ParameterValue(0.0));
    nav2_util::declare_parameter_if_not_declared(
        node, this->behavior_name_ + ".minimum_speed", rclcpp::ParameterValue(0.0));
    node->get_parameter(this->behavior_name_ + ".acceleration_limit", acceleration_limit_);
    node->get_parameter(this->behavior_name_ + ".deceleration_limit", deceleration_limit_);
    node->get_parameter(this->behavior_name_ + ".minimum_speed", minimum_speed_);
    if (acceleration_limit_ < 0.0 || deceleration_limit_ > 0.0) {
        RCLCPP_ERROR(this->logger_,
            "DriveOnHeading: acceleration_limit and deceleration_limit must be "
            "positive and negative respectively");
        acceleration_limit_ = std::abs(acceleration_limit_);
        deceleration_limit_ = -std::abs(deceleration_limit_);
    }
}


}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_behaviors::CollisionBack, nav2_core::Behavior)
