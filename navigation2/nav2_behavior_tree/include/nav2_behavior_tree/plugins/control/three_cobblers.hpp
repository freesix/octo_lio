#ifndef THREE_COBBLERS_HPP_
#define THREE_COBBLERS_HPP

#include <behaviortree_cpp_v3/control_node.h>

namespace nav2_behavior_tree
{

class ThreeCobblers : public BT::ControlNode{
public:

    ThreeCobblers(const std::string &name, const BT::NodeConfiguration& conf);

    ~ThreeCobblers() override = default;

    static BT::PortsList providedPorts(){
        return{
            // BT::InputPort<int>("num_of_retries", 1, "Num of retries")
        };
    }

private:

    // int number_of_retries_;
    // int retry_count_;
    enum class Phase{
        child_node1,
        child_node3
    };

    Phase phase_ = Phase::child_node1;

    BT::NodeStatus tick() override;

    void halt() override;
};

}


#endif