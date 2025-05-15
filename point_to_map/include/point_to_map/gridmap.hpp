#pragma once

#include <queue>
#include <map>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include "value_conversion_tables.hpp"
#include "probability_values.hpp"
#include "map_limits.hpp"
#include "xy_index.hpp"
#include "grid_2d.hpp"
#include "probability_grid.hpp"
#include "probability_grid_range_data_inserter.hpp"

template<typename T>
inline double stamp2Sec(const T& stamp)
{
    return rclcpp::Time(stamp).seconds();
}

class gridmap : public rclcpp::Node{

public:
    gridmap(const rclcpp::NodeOptions &options);
    ~gridmap();

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);


    
private:

    std::unique_ptr<ProbabilityGrid> CreateGrid(const Eigen::Vector2f &origin);
    void InsertRangeData(RangeData &range_data, 
            const ProbabilityGridRangeDataInserter2D* range_data_inserter);
    void RosToRangeData(nav_msgs::msg::Odometry odom, sensor_msgs::msg::LaserScan scan,
        RangeData &rangeData); 
    void pubOccMap(std_msgs::msg::Header header, std::unique_ptr<Grid2D> &grid);

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_occ_;
    nav_msgs::msg::Odometry::SharedPtr lastOdom; 
    std::queue<sensor_msgs::msg::LaserScan> scan_data;
    std::queue<nav_msgs::msg::Odometry> odom_data;
    std::thread run_thread_;

    ValueConversionTables *conversion_tables_;
    std::unique_ptr<ProbabilityGrid> grid_;
    std::unique_ptr<ProbabilityGridRangeDataInserter2D> RangeDataInsert_;
    std::string scan_topic_;
    std::string odom_topic_;
    double hit_, miss_;
    double resolution_;

    void run();
    void getParams(){
        scan_topic_ = this->declare_parameter("scan_topic", "");
        odom_topic_ = this->declare_parameter("odom_topic", "");
        hit_ = this->declare_parameter("hit", 0.55);
        miss_ = this->declare_parameter("miss", 0.49);
        resolution_ = this->declare_parameter("resolution", 0.05);
    }
};

rmw_qos_profile_t qos_profile_map{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,  // 历史记录策略：KEEP_LAST
    10,                                // 队列深度：10
    RMW_QOS_POLICY_RELIABILITY_RELIABLE, // 可靠性策略：RELIABLE
    RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL, // 持久性策略：TRANSIENT_LOCAL
    RMW_QOS_DEADLINE_DEFAULT,          // 默认期限
    RMW_QOS_LIFESPAN_DEFAULT,          // 默认生命周期
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT, // 默认活跃性策略
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT, // 默认活跃性租约时长
    false                              // 避免 ROS 命名空间前缀
};

auto qos_map = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_map.history,
        qos_profile_map.depth
    ),
    qos_profile_map);
