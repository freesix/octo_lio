#pragma once

#include <rclcpp/rclcpp.hpp>
#include <behaviortree_cpp_v3/condition_node.h>
#include <std_msgs/msg/bool.hpp>


namespace nav2_behavior_tree
{

class IsCollision : public BT::ConditionNode{
public:
    IsCollision(const std::string& condition_name,
                const BT::NodeConfiguration& conf);
    IsCollision() = delete;

    BT::NodeStatus tick() override;

    static BT::PortsList providedPorts(){
        return {
            BT::InputPort<std::string>(
                "collision_topic", std::string("/collision_trigger"), "Collision signal.")
        };
    }

private:
    void collisionCallback(std_msgs::msg::Bool::SharedPtr msg);

    rclcpp::Node::SharedPtr node_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    rclcpp::executors::SingleThreadedExecutor callback_group_exe_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_;
    std::string collision_topic_;
    bool is_collision_;
    bool last_state_; 

};

}