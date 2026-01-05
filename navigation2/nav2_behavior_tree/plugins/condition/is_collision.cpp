#include "nav2_behavior_tree/plugins/condition/is_collision.hpp"

namespace nav2_behavior_tree{

IsCollision::IsCollision(const std::string& condition_name, 
                         const BT::NodeConfiguration& conf)
    :BT::ConditionNode(condition_name, conf),
    collision_topic_("/collision_trigger"),
    is_collision_(false),
    last_state_(false){

    getInput("collision_topic", collision_topic_);

    node_ = config().blackboard->get<rclcpp::Node::SharedPtr>("node");
    callback_group_ = node_->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive, false);
    callback_group_exe_.add_callback_group(callback_group_, node_->get_node_base_interface());

    rclcpp::SubscriptionOptions sub_option;
    sub_option.callback_group = callback_group_;
    sub_ = node_->create_subscription<std_msgs::msg::Bool>(
        collision_topic_, rclcpp::SystemDefaultsQoS(),
        std::bind(&IsCollision::collisionCallback, this, std::placeholders::_1),
        sub_option); 
}

BT::NodeStatus IsCollision::tick(){
    callback_group_exe_.spin_some();
    if(is_collision_){
        return BT::NodeStatus::SUCCESS;
    }
    return BT::NodeStatus::RUNNING;
}

void IsCollision::collisionCallback(std_msgs::msg::Bool::SharedPtr msg){
    if(msg->data && !last_state_){
        is_collision_ = true; 
    }
    else{
        is_collision_ = false;
    }
    last_state_ = msg->data;
}

} // namesapce nav2_behavior_tree

#include "behaviortree_cpp_v3/bt_factory.h"
BT_REGISTER_NODES(factory){
    factory.registerNodeType<nav2_behavior_tree::IsCollision>("IsCollision");
}