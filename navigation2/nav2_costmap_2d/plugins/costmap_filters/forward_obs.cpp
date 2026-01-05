#include "nav2_costmap_2d/costmap_filters/forward_obs.hpp"

namespace nav2_costmap_2d{

ForwardObs::ForwardObs()
{
}

void ForwardObs::initializeFilter(const std::string &filter_info_topic){
    (void) filter_info_topic;
    std::lock_guard<CostmapFilter::mutex_t> guard(*getMutex());

    rclcpp_lifecycle::LifecycleNode::SharedPtr node = node_.lock();
    if(!node){
        throw std::runtime_error{"Failed to lock node"};
    }

    std::string forward_obs_topic;
    declareParameter("forward_obs_topic", rclcpp::ParameterValue("forward_obs"));
    node->get_parameter(name_ + "." + "forward_obs_topic", forward_obs_topic);

    declareParameter("rect_length", rclcpp::ParameterValue("rect_length"));
    declareParameter("rect_width", rclcpp::ParameterValue("rect_width"));
    declareParameter("forward_offset", rclcpp::ParameterValue("forward_offset"));
    declareParameter("clear_distance", rclcpp::ParameterValue("clear_distance"));

    node->get_parameter(name_ + "." + "rect_length", rect_length_);
    node->get_parameter(name_ + "." + "rect_width", rect_width_);
    node->get_parameter(name_ + "." + "forward_offset", forward_offset_);
    node->get_parameter(name_ + "." + "clear_distance", clear_distance_);
    
    collision_sub_ = node->create_subscription<std_msgs::msg::Bool>(
        forward_obs_topic, 10,
        std::bind(&ForwardObs::collisionCallback, this, std::placeholders::_1));
}

void ForwardObs::collisionCallback(const std_msgs::msg::Bool::SharedPtr msg){
    if(msg->data){
        if(!active_.load()){
            pending_trigger_.store(true);
        }
    }
}

void ForwardObs::computerAnchorFromPose(const geometry_msgs::msg::Pose2D& pose){
    anchor_yaw_ = pose.theta;
    anchor_x_ = pose.x + forward_offset_ * std::cos(anchor_yaw_);
    anchor_y_ = pose.y + forward_offset_ * std::sin(anchor_yaw_);
}

void ForwardObs::process(nav2_costmap_2d::Costmap2D & master_grid,
                         int min_i, int min_j, int max_i, int max_j,
                         const geometry_msgs::msg::Pose2D & pose)
{
    if(pending_trigger_.load()){
        std::lock_guard<std::mutex> lk(state_mtx_);
        if(!active_){
            computerAnchorFromPose(pose);
            active_ = true;
            RCLCPP_DEBUG(logger_, "ForwardObs: trigger accepted @ (%.3f, %.3f, yaw=%.2f)",
                   anchor_x_, anchor_y_, anchor_yaw_);
        }
        pending_trigger_.store(false);
    }
    
    if(!active_.load()){
        return;
    }

    const double dx = pose.x - anchor_x_;
    const double dy = pose.y - anchor_y_;
    const double dist = std::hypot(dx, dy);
    if(dist > clear_distance_){
        std::lock_guard<std::mutex> lk(state_mtx_);
        active_ = false;
        RCLCPP_DEBUG(logger_, "ForwardObs: cleared (dist %.3f > %.3f)", dist, clear_distance_);
        return;
    }

    // const double res = master_grid.getResolution();
    unsigned int ui, uj;
    double wx, wy;

    const double hx = 0.5 * rect_length_;
    const double hy = 0.5 * rect_width_;

    // 为减少计算量，先构造一个世界对齐的包围盒，和窗口求交
    // 包围盒的最保守做法是：以中心为 (anchor_x_, anchor_y_)，扩展 max(hx, hy) 到四周
    const double r = std::hypot(hx, hy); // 旋转后外接圆半径
    // 将世界包围盒映射到 map 索引，再与[min_i,max_i)×[min_j,max_j)取交集
    unsigned int bb_min_i, bb_min_j, bb_max_i, bb_max_j;
    {
        double bb_min_x = anchor_x_ - r;
        double bb_min_y = anchor_y_ - r;
        double bb_max_x = anchor_x_ + r;
        double bb_max_y = anchor_y_ + r;

        // clamp 到地图范围
        // int li, lj, ui_, uj_;
        master_grid.worldToMapEnforceBounds(bb_min_x, bb_min_y, (int&)bb_min_i, (int&)bb_min_j);
        master_grid.worldToMapEnforceBounds(bb_max_x, bb_max_y, (int&)bb_max_i, (int&)bb_max_j);

        // 与传入窗口求交（注意 max_i/max_j 本身是“上界”，非含）
        if ((int)bb_min_i < min_i) bb_min_i = (unsigned int)min_i;
        if ((int)bb_min_j < min_j) bb_min_j = (unsigned int)min_j;
        if ((int)bb_max_i > max_i) bb_max_i = (unsigned int)max_i;
        if ((int)bb_max_j > max_j) bb_max_j = (unsigned int)max_j;

        if (bb_min_i >= bb_max_i || bb_min_j >= bb_max_j) {
            return; // 与窗口无交集
        }
    }

    for (ui = bb_min_i; ui < bb_max_i; ++ui) {
        for (uj = bb_min_j; uj < bb_max_j; ++uj) {
            master_grid.mapToWorld(ui, uj, wx, wy);

            double lx, ly;  // 矩形局部坐标
            worldToRectLocal(wx, wy, anchor_x_, anchor_y_, anchor_yaw_, lx, ly);

            // 点是否落在局部轴对齐矩形内部
            if (std::fabs(lx) <= hx && std::fabs(ly) <= hy) {
            master_grid.setCost(ui, uj, LETHAL_OBSTACLE);
        }
    }
  }
}


void ForwardObs::resetFilter(){
    std::lock_guard<std::mutex> lk(state_mtx_);
    active_ = false;
    pending_trigger_ = false;
}

bool ForwardObs::isActive()
{
  return active_.load();
}
};



#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::ForwardObs, nav2_costmap_2d::Layer)