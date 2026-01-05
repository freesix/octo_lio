#pragma once

#include <string>
#include "behaviortree_cpp_v3/control_node.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

namespace nav2_behavior_tree
{

class CollisionRecovery : public BT::ControlNode
{
public:
    CollisionRecovery(
        const std::string & name,
        const BT::NodeConfiguration & conf);

    ~CollisionRecovery() override = default;

    static BT::PortsList providedPorts()
    {
        return{
            BT::InputPort<std::string>(
                "collision_topic", std::string("/collision_trigger"), "Collision signal."),
            BT::InputPort<int>("number_of_retries", 1, "Number of retries")
        };
    }

private:
    unsigned int current_child_idx_;
    unsigned int number_of_retries_;
    unsigned int retry_count_;


    void collisionCallback(std_msgs::msg::Bool::SharedPtr msg);

    rclcpp::Node::SharedPtr node_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    rclcpp::executors::SingleThreadedExecutor callback_group_exe_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_;
    std::string collision_topic_;

    bool is_collision_;
    bool last_state_; 
    BT::NodeStatus tick() override;

    void halt() override;
};

}  // namespace nav2_behavior_tree

