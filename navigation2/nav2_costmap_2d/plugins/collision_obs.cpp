#include "nav2_costmap_2d/collision_obs.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>

PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::CollisionLayer, nav2_costmap_2d::Layer)
#define LOG_INFO_COLOR(logger, color, fmt, ...) \
  RCLCPP_INFO(logger, "%s" fmt "%s", color, ##__VA_ARGS__, COLOR_RESET)
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_RESET   "\033[0m"
namespace nav2_costmap_2d{


CollisionLayer::CollisionLayer()
    : pending_triggers_(0),
      last_collision_state_(false),
      topic_("/collision_topic"),
      front_offest_(0.40),
      size_x_(0.6),
      size_y_(0.3),
      leave_distance_(1.5),
      frame_id_("base_link")
{
    access_ = new mutex_t();
}
CollisionLayer::~CollisionLayer(){
    delete access_;
}

void CollisionLayer::onInitialize(){
    auto node = node_.lock();
    if(!node){
        throw std::runtime_error{"Failed to lock node"};
    }
    current_ = true;

    declareParameter("enabled", rclcpp::ParameterValue(true));
    declareParameter("topic", rclcpp::ParameterValue(std::string("collision_topic")));
    declareParameter("front_offset", rclcpp::ParameterValue(0.9));
    declareParameter("size_x", rclcpp::ParameterValue(0.3));
    declareParameter("size_y", rclcpp::ParameterValue(0.6));
    declareParameter("leave_distance", rclcpp::ParameterValue(1.5));
    declareParameter("frame_id", rclcpp::ParameterValue(std::string("")));
    
    node->get_parameter(name_ + "." + "enabled", enabled_);
    node->get_parameter(name_ + "." + "topic", topic_);
    node->get_parameter(name_ + "." + "front_offset", front_offest_);
    node->get_parameter(name_ + "." + "size_x", size_x_);
    node->get_parameter(name_ + "." + "size_y", size_y_);
    node->get_parameter(name_ + "." + "leave_distance", leave_distance_);
    node->get_parameter(name_ + "." + "frame_id", frame_id_);

    sub_ = node->create_subscription<std_msgs::msg::Bool>(
        topic_, 10,
        std::bind(&CollisionLayer::collisionCallback, this, std::placeholders::_1));
        
    enabled_ = true;
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

}

void CollisionLayer::collisionCallback(const std_msgs::msg::Bool::SharedPtr msg){
    std::lock_guard<Costmap2D::mutex_t> lock(*getMutex());
    bool cur = msg->data;
    if(cur && !last_collision_state_){
        pending_triggers_ += 1;
        LOG_INFO_COLOR(logger_, COLOR_RED, ": pending_triggers %d", pending_triggers_);
    }

    last_collision_state_ = cur;
}

void CollisionLayer::updateBounds(
    double robot_x, double robot_y, double /*robot_yaw*/,
    double* min_x, double* min_y, double* max_x, double* max_y)
{   
    std::lock_guard<Costmap2D::mutex_t> lock(*getMutex());
    if(!enabled_){
        return;
    }
    // 消费者
    // LOG_INFO_COLOR(logger_, COLOR_RED, "updataBounds: %i", pending_triggers_);
    while(pending_triggers_ > 0){
        global_frame_ = layered_costmap_->getGlobalFrameID();
        try{     
            geometry_msgs::msg::TransformStamped tf = 
                tf_buffer_->lookupTransform(global_frame_, frame_id_, tf2::TimePointZero);
            tf2::Quaternion q(tf.transform.rotation.x, tf.transform.rotation.y, 
                tf.transform.rotation.z, tf.transform.rotation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
        

            double obs_x = tf.transform.translation.x + std::cos(yaw)*front_offest_;
            double obs_y = tf.transform.translation.y + std::sin(yaw)*front_offest_;
            obstacle_.push_back(Obstacle{obs_x, obs_y, yaw});
            LOG_INFO_COLOR(logger_, COLOR_RED, "updataBounds: %i", pending_triggers_);
            LOG_INFO_COLOR(logger_, COLOR_RED, "robot_x: %f,   robot_y: %f", robot_x, robot_y);
            // LOG_INFO_COLOR(logger_, COLOR_RED, "front_offest: %f,   robot_yaw: %f", front_offest_, robot_yaw);
            // LOG_INFO_COLOR(logger_, COLOR_RED, "obs_x: %f,   obs_y: %f", obs_x, obs_y);
        }catch(tf2::TransformException &e){
            RCLCPP_WARN(logger_, "Transorm failed: %s", e.what());
        }
        pending_triggers_ -= 1;
    }
    // LOG_INFO_COLOR(logger_, COLOR_GREEN, "updataBounds: %i", pending_triggers_);

    double half_x = size_x_ / 2.0;
    double half_y = size_y_ / 2.0;
    std::vector<size_t> to_remove;
    for(size_t i = 0; i < obstacle_.size(); ++i){
        const Obstacle &obs = obstacle_[i];

          // 四角旋转
        std::vector<std::pair<double,double>> corners = {
            { half_x,  half_y},
            { half_x, -half_y},
            {-half_x, -half_y},
            {-half_x,  half_y}
        };
        for(auto &c : corners){
            double wx = obs.x + std::cos(obs.yaw)*c.first - std::sin(obs.yaw)*c.second;
            double wy = obs.y + std::sin(obs.yaw)*c.first + std::cos(obs.yaw)*c.second;
            *min_x = std::min(*min_x, wx);
            *min_y = std::min(*min_y, wy);
            *max_x = std::max(*max_x, wx);
            *max_y = std::max(*max_y, wy);
        }


/*         *min_x = std::min(*min_x, obs.x - half_x);
        *min_y = std::min(*min_y, obs.y - half_y);
        *max_x = std::max(*max_x, obs.x + half_x);
        *max_y = std::max(*max_y, obs.y + half_y);
 */
        double dx = robot_x - obs.x;
        double dy = robot_y - obs.y;
        double dist = std::sqrt(dx*dx + dy*dy);

        if(dist > leave_distance_){
            to_remove.push_back(i);
        }
    }

    // 按索引删除（从后往前）
    for(auto it = to_remove.rbegin(); it != to_remove.rend(); ++it){
        obstacle_.erase(obstacle_.begin() + *it);
    }
}


void CollisionLayer::updateCosts(
    nav2_costmap_2d::Costmap2D& master_grid,
    int /*min_i*/, int /*min_j*/, int /*max_i*/, int /*max_j*/)
{
    std::lock_guard<Costmap2D::mutex_t> lock(*getMutex());
    if(!enabled_ || obstacle_.empty()){
        return;
    }

    unsigned int mx, my;
    double resolution = master_grid.getResolution();

    // 对每个障碍，在 costmap 上写入 LETHAL_OBSTACLE
    for(const auto &obs : obstacle_){
        double half_x = size_x_ / 2.0;
        double half_y = size_y_ / 2.0;

        std::vector<std::pair<double, double>> local_corners = {
            { half_x,  half_y},
            { half_x, -half_y},
            {-half_x, -half_y},
            {-half_x,  half_y}
        };

        // 旋转 + 平移到 global_frame
        std::vector<std::pair<double, double>> world_corners;
        for (auto &p : local_corners) {
            double wx = obs.x + cos(obs.yaw) * p.first - sin(obs.yaw) * p.second;
            double wy = obs.y + sin(obs.yaw) * p.first + cos(obs.yaw) * p.second;
            world_corners.push_back({wx, wy});
        }

        // 将矩形网格化到 costmap
        double min_cx = world_corners[0].first, max_cx = world_corners[0].first;
        double min_cy = world_corners[0].second, max_cy = world_corners[0].second;
        for (auto &c : world_corners) {
            min_cx = std::min(min_cx, c.first);
            max_cx = std::max(max_cx, c.first);
            min_cy = std::min(min_cy, c.second);
            max_cy = std::max(max_cy, c.second);
        }

        for (double x = min_cx; x <= max_cx; x += resolution) {
            for (double y = min_cy; y <= max_cy; y += resolution) {
                if (master_grid.worldToMap(x, y, mx, my)) {
                    master_grid.setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
                }
            }
        }


        /* for(double x = obs.x - half_x; x <= obs.x + half_x; x += resolution){
            for(double y = obs.y - half_y; y <= obs.y + half_y; y += resolution){
                if(master_grid.worldToMap(x, y, mx, my)){
                    master_grid.setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
                }
            }
        } */
    }

}


void CollisionLayer::onFootprintChanged()
{

    RCLCPP_DEBUG(rclcpp::get_logger(
        "nav2_costmap_2d"), "GradientLayer::onFootprintChanged(): num footprint points: %lu",
        layered_costmap_->getFootprint().size());
}

}