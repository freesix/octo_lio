#pragma once

#include <string>
#include <behaviortree_cpp_v3/action_node.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
namespace nav2_behavior_tree{

class PubCostmapTrigger : public BT::SyncActionNode{
public:

    PubCostmapTrigger(
        const std::string& xml_tag_name,
        const BT::NodeConfiguration& conf
    );

    static BT::PortsList providedPorts();

    BT::NodeStatus tick() override;

private:

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_;
    std::string topic_name_;
};


}