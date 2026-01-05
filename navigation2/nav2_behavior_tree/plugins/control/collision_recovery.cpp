#include "nav2_behavior_tree/plugins/control/collison_recovery.hpp"

namespace nav2_behavior_tree
{

CollisionRecovery::CollisionRecovery(
    const std::string& name,
    const BT::NodeConfiguration& conf)
    : BT::ControlNode::ControlNode(name, conf),
      current_child_idx_(0),
      number_of_retries_(1),
      retry_count_(0),
      collision_topic_("/collision_trigger"),
      is_collision_(false),
      last_state_(false)
{   

            
    getInput("collision_topic", collision_topic_);
    getInput("number_of_retries", number_of_retries_);

    node_ = config().blackboard->get<rclcpp::Node::SharedPtr>("node");
    callback_group_ = node_->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive, false);
    callback_group_exe_.add_callback_group(callback_group_, node_->get_node_base_interface());

    rclcpp::SubscriptionOptions sub_option;
    sub_option.callback_group = callback_group_;
    sub_ = node_->create_subscription<std_msgs::msg::Bool>(
        collision_topic_, rclcpp::SystemDefaultsQoS(),
        std::bind(&CollisionRecovery::collisionCallback, this, std::placeholders::_1),
        sub_option);  
}

BT::NodeStatus CollisionRecovery::tick(){
    callback_group_exe_.spin_some();
    const unsigned children_count = children_nodes_.size();
    if(children_count != 2){
        throw BT::BehaviorTreeException("CollisionRecovery: '" + name() + "' must only have 2 children.");
    }
    setStatus(BT::NodeStatus::RUNNING);
    while(current_child_idx_ < children_count && retry_count_ <= number_of_retries_){
        TreeNode* child_node = children_nodes_[current_child_idx_];
        const BT::NodeStatus child_status = child_node->executeTick();
/*         std::cout << "\033[32mcurrent_child_idx=" << current_child_idx_ 
        << "\033[0m, "
        << "\033[33mchild_status=" << child_status
        << "\033[0m" 
        << std::endl; */
        if(current_child_idx_ == 0){
            switch (child_status)
            {
                case BT::NodeStatus::SUCCESS:
                {
                    halt();
                    return BT::NodeStatus::SUCCESS;
                }
                case BT::NodeStatus::FAILURE:
                {
                    if(retry_count_ < number_of_retries_){
                        ControlNode::haltChild(0);
                        current_child_idx_++;
                        break;
                    }
                    else{
                        halt();
                        return BT::NodeStatus::FAILURE;
                    }
                }
                case BT::NodeStatus::RUNNING:
                {
                    if(is_collision_){
                        is_collision_ = false;
                        ControlNode::haltChild(0);
                        current_child_idx_++;
                        break;
                    }
                    return BT::NodeStatus::RUNNING;
                }
                default:
                {
                    throw BT::LogicError("node must never return IDLE.");
                }
            } // end switch
        }
        else if(current_child_idx_ == 1){
       /*   std::cout << "\033[32mcurrent_child_idx=" << current_child_idx_ 
        << "\033[0m, "
        << "\033[33mchild_status=" << child_status
        << "\033[0m" 
        << std::endl; */
           
            switch(child_status)
            {
                case BT::NodeStatus::SUCCESS:
                {
                    ControlNode::haltChild(1);
                    retry_count_++;
                    current_child_idx_--;
                }
                break;
                case BT::NodeStatus::FAILURE:
                {
                    halt();
                    return BT::NodeStatus::FAILURE;
                }
                case BT::NodeStatus::RUNNING:
                {
                    return BT::NodeStatus::RUNNING;
                }
                default:
                {
                    throw BT::LogicError("node must never return IDLE.");
                }
            } // end switch
        }
    }

    halt();
    return BT::NodeStatus::FAILURE;
}

void CollisionRecovery::halt(){
    ControlNode::halt();
    retry_count_ = 0;
    current_child_idx_ = 0;
    last_state_ = false;
    is_collision_ = false;
}


void CollisionRecovery::collisionCallback(std_msgs::msg::Bool::SharedPtr msg){
    // RCLCPP_INFO_STREAM(rclcpp::get_logger("CollisionRecovery"), "is_collision: "<<is_collision_);
    // std::cout<<"CollisionRecovery is_collision: "<<is_collision_<<std::endl; 
    if(msg->data && !last_state_){
        is_collision_ = true;
        // RCLCPP_INFO_STREAM(rclcpp::get_logger("CollisionRecovery"), "is_collision: "<<is_collision_); 
    }
    // else{
        // is_collision_ = false;
    // }
    last_state_ = msg->data;
}

    
} // namesapce

#include "behaviortree_cpp_v3/bt_factory.h"
BT_REGISTER_NODES(factory){
    factory.registerNodeType<nav2_behavior_tree::CollisionRecovery>("CollisionRecovery");
}