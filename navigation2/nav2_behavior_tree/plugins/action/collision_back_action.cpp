#include <string>
#include <memory>

#include "nav2_behavior_tree/plugins/action/collision_back_action.hpp"


namespace nav2_behavior_tree
{

CollisionBackAction::CollisionBackAction(
    const std::string& xml_tag_name,
    const std::string& action_name,
    const BT::NodeConfiguration& conf
): BtActionNode<nav2_msgs::action::BackUp>(xml_tag_name, action_name, conf)
{
    double dist;
    getInput("backup_dist", dist);
    double speed;
    getInput("backup_speed", speed);
    double time_allowance;
    getInput("time_allowance",time_allowance);


    goal_.target.x = dist;
    goal_.target.y = 0.0;
    goal_.target.z = 0.0;
    goal_.speed = speed;
    goal_.time_allowance = rclcpp::Duration::from_seconds(time_allowance);
}

void CollisionBackAction::on_tick(){
    increment_recovery_count();
}

} // namespace


#include "behaviortree_cpp_v3/bt_factory.h"
BT_REGISTER_NODES(factory)
{
  BT::NodeBuilder builder =
    [](const std::string & name, const BT::NodeConfiguration & config)
    {
      return std::make_unique<nav2_behavior_tree::CollisionBackAction>(
        name, "collision_back", config);
    };

  factory.registerBuilder<nav2_behavior_tree::CollisionBackAction>("CollisionBack", builder);
}
