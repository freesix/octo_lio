#include "point_to_map/point2scan.hpp"

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "tf2_ros/create_timer_ros.h"

point2scan::point2scan(const rclcpp::NodeOptions &options) :
    rclcpp::Node("point2scan", options){
    
    getparams();

    pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(scan_topic_, rclcpp::SensorDataQoS());

    if(!target_frame_.empty()){
        tf2_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
            this->get_node_base_interface(), this->get_node_timers_interface());
        tf2_->setCreateTimerInterface(timer_interface);
        tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_);
        message_filter_ = std::make_unique<MessageFilter>(
            sub_, *tf2_, target_frame_, input_queue_size_,
            this->get_node_logging_interface(),
            this->get_node_clock_interface());
        message_filter_->registerCallback(
            std::bind(&point2scan::cloudCallBack, this, std::placeholders::_1)); 
    }
    else{
        sub_.registerCallback(std::bind(&point2scan::cloudCallBack, this, std::placeholders::_1)); 
    }

    sub_listener_thread_ = std::thread(
        std::bind(&point2scan::subscriptionListenerThreadLoop, this));   
};
/**
 * @brief 转换为单线LaserScan
 */
void point2scan::cloudCallBack(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg){
    auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
    scan_msg->header = cloud_msg->header;
    if(!target_frame_.empty()){
        scan_msg->header.frame_id = target_frame_; 
    }

    scan_msg->angle_min = angle_min_;
    scan_msg->angle_max = angle_max_;
    scan_msg->angle_increment = angle_increment_;
    scan_msg->time_increment = 0.0;
    scan_msg->scan_time = scan_time_;
    scan_msg->range_min = range_min_;
    scan_msg->range_max = range_max_;

    uint32_t ranges_size = std::ceil((scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment);

    if(use_inf_){
        scan_msg->ranges.assign(ranges_size, std::numeric_limits<double>::infinity()); 
    }
    else{
        scan_msg->ranges.assign(ranges_size, scan_msg->range_max + inf_epsilon_); 
    }

    if(scan_msg->header.frame_id != cloud_msg->header.frame_id){
        try{
            auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
            tf2_->transform(*cloud_msg, *cloud, target_frame_, tf2::durationFromSec(tolerance_));
            cloud_msg = cloud; 
        }catch(tf2::TransformException &ex){
            RCLCPP_ERROR_STREAM(this->get_logger(), "Transform failure: " << ex.what());
            return; 
        } 
    }

    for(sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x"), iter_y(*cloud_msg, "y"),
        iter_z(*cloud_msg, "z"); iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z){

        if(std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z)){
            RCLCPP_DEBUG(this->get_logger(), "rejected for nan in point(%f, %f, %f)\n",
                *iter_x, *iter_y, *iter_z);
            continue; 
        }

        if(*iter_z > max_height_ || *iter_z < min_height_){
            RCLCPP_DEBUG(this->get_logger(), "rejected for height %f not in range(%f, %f)\n",
                *iter_z, *iter_x, *iter_y);
            continue; 
        }

        double range = hypot(*iter_x, *iter_y);
        if(range < range_min_){
            RCLCPP_DEBUG(this->get_logger(), "rejected for range %f above minimum value %f. Point: (%f, %f, %f)\n", 
                range, range_min_, *iter_x, *iter_y, *iter_z);
            continue;
        }
        
        if(range > range_max_){
            RCLCPP_DEBUG(this->get_logger(), "rejected for range %f above maximum value %f. Point: (%f, %f, %f)\n", 
                range, range_max_, *iter_x, *iter_y, *iter_z);
            continue;
        }
        
        double angle = atan2(*iter_y, *iter_x);
        if(angle < scan_msg->angle_min || angle > scan_msg->angle_max){
            RCLCPP_DEBUG(this->get_logger(),"rejected for angle %f not in range (%f, %f)\n",
                angle, scan_msg->angle_min, scan_msg->angle_max);
            continue;
        }

        int index = (angle - scan_msg->angle_min) / scan_msg->angle_increment;
        if(range < scan_msg->ranges[index]){
            scan_msg->ranges[index] = range; 
        }
    }
    pub_->publish(std::move(scan_msg));
}
/**
 * @brief 订阅3d点云
 */
void point2scan::subscriptionListenerThreadLoop(){
    rclcpp::Context::SharedPtr context = this->get_node_base_interface()->get_context();

    const std::chrono::milliseconds timeout(100);
    while(rclcpp::ok(context) && alive_.load()){
        int subscription_count = pub_->get_subscription_count() + pub_->get_intra_process_subscription_count();
        if(subscription_count > 0){
            if(!sub_.getSubscriber()){
                RCLCPP_INFO(this->get_logger(), "Got a subscriber to laserscan, starting pointcloud subscriber");
                rclcpp::SensorDataQoS qos;
                qos.keep_last(input_queue_size_);
                sub_.subscribe(this, sub_topic_, qos.get_rmw_qos_profile()); 
            } 
        }
        else if(sub_.getSubscriber()){
            RCLCPP_INFO(this->get_logger(), "No subscriber to laserscan!");
            sub_.unsubscribe(); 
        }
        rclcpp::Event::SharedPtr event = this->get_graph_event();
        this->wait_for_graph_change(event, timeout); 
    }
    sub_.unsubscribe();
}

point2scan::~point2scan()
{
  alive_.store(false);
  sub_listener_thread_.join();
}

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(point2scan)