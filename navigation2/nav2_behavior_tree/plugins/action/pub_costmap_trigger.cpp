#include "nav2_behavior_tree/plugins/action/pub_costmap_trigger.hpp"
#include <thread>

namespace nav2_behavior_tree{

PubCostmapTrigger::PubCostmapTrigger(
    const std::string& xml_tag_name,
    const BT::NodeConfiguration& conf):
    BT::SyncActionNode(xml_tag_name, conf)
{
    node_ = config().blackboard->get<rclcpp::Node::SharedPtr>("node");
    
    getInput("topic_name", topic_name_);
    if(topic_name_.empty()){
        topic_name_ = "/costmap_trigger";
    }

    pub_ = node_->create_publisher<std_msgs::msg::Bool>(topic_name_, 10);

}

BT::PortsList PubCostmapTrigger::providedPorts(){
    return{
        BT::InputPort<std::string>("topic_name", "/costmap_trigger", "Topic to pub") 
    }; 
}

BT::NodeStatus PubCostmapTrigger::tick(){
    std_msgs::msg::Bool msg;
    msg.data = true;

    pub_->publish(msg);

    std::thread([this](){
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std_msgs::msg::Bool msg;
        msg.data = false;
        pub_->publish(msg);
    }).detach();

    RCLCPP_INFO(node_->get_logger(), "Publish costmap trigger: on topic '%s'", topic_name_.c_str());

    return BT::NodeStatus::SUCCESS;
}

}

#include "behaviortree_cpp_v3/bt_factory.h"
BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<nav2_behavior_tree::PubCostmapTrigger>("PubCostmapTrigger");
}