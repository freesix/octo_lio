#pragma once


#include <string>

#include "nav2_behavior_tree/bt_action_node.hpp"
#include "nav2_msgs/action/back_up.hpp"


namespace nav2_behavior_tree
{

class CollisionBackAction : public BtActionNode<nav2_msgs::action::BackUp>
{
public:
    CollisionBackAction(
        const std::string& xml_tag_name,
        const std::string& action_name,
        const BT::NodeConfiguration& conf
    );


    void on_tick() override;

    static BT::PortsList providedPorts()
    {
    return providedBasicPorts(
        {
            BT::InputPort<double>("backup_dist", 0.15, "Distance to backup"),
            BT::InputPort<double>("backup_speed", 0.025, "Speed at which to backup"),
            BT::InputPort<double>("time_allowance", 10.0, "Allowed time for reversing")
        });
    }

};

}