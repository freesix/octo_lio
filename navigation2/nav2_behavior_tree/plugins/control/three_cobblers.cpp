#include "nav2_behavior_tree/plugins/control/three_cobblers.hpp"


namespace nav2_behavior_tree
{

ThreeCobblers::ThreeCobblers(
    const std::string& name,
    const BT::NodeConfiguration& conf)
    : BT::ControlNode::ControlNode(name, conf)
    // number_of_retries_(1),
    // retry_count_(0)
{
    // getInput("num_of_retires", number_of_retries_);
}
    

BT::NodeStatus ThreeCobblers::tick(){
    const unsigned children_count = children_nodes_.size();
    if(children_count != 3){
        throw BT::BehaviorTreeException("ThreeCobbles Node '" + name() + "' must only have 3 children.");
    }

    setStatus(BT::NodeStatus::RUNNING);

    auto& node1 = children_nodes_[0];
    auto& node2 = children_nodes_[1];
    auto& node3 = children_nodes_[2];

    switch(phase_)
    {
        case Phase::child_node1:
        {
            BT::NodeStatus s1 = node1->executeTick();

            BT::NodeStatus s2 = node2->executeTick();

            switch(s1)
            {
                case BT::NodeStatus::SUCCESS:
                {
                    halt();
                    return BT::NodeStatus::SUCCESS;
                }
                case BT::NodeStatus::FAILURE:
                {
                    halt();
                    return BT::NodeStatus::FAILURE;
                }
                case BT::NodeStatus::RUNNING:
                {
                    if(s2 == BT::NodeStatus::SUCCESS){
                        std::cout << "\033[32ms1=" << static_cast<int>(s1) 
                            << "\033[0m, "
                            << "\033[33ms2=" << static_cast<int>(s2) 
                            << "\033[0m" 
                            << std::endl;
                        ControlNode::haltChild(0);
                        
                        phase_ = Phase::child_node3;
                        return BT::NodeStatus::RUNNING;
                    }
                    else{
                        return BT::NodeStatus::RUNNING;
                    }
                }
                default:
                {
                    throw BT::LogicError("A child node must never return IDLE");
                } 
            }
        }

        case Phase::child_node3:
        {
            BT::NodeStatus s3 = node3->executeTick();
            if(s3 == BT::NodeStatus::SUCCESS){
                phase_ = Phase::child_node1;
                return BT::NodeStatus::RUNNING;
            }
            else if(s3 == BT::NodeStatus::FAILURE){
                halt();
                return BT::NodeStatus::FAILURE;
            }
            return BT::NodeStatus::RUNNING;
        }
        default:
        {
            throw BT::LogicError("must never return IDLE");
        }
    }
    return BT::NodeStatus::RUNNING;
}


void ThreeCobblers::halt()
{
    // for (size_t i=0; i<children_nodes_.size(); i++)
    // {
        // haltChild(i);
    // }

    ControlNode::halt();

    phase_ = Phase::child_node1;
}

} // namespace

#include "behaviortree_cpp_v3/bt_factory.h"
BT_REGISTER_NODES(factory)
{
    factory.registerNodeType<nav2_behavior_tree::ThreeCobblers>("ThreeCobblers");
}

