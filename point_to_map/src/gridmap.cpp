#include "point_to_map/gridmap.hpp"

gridmap::gridmap(const rclcpp::NodeOptions &options) : 
    rclcpp::Node("gridmap", options){
    getParams();

    lastOdom = std::make_shared<nav_msgs::msg::Odometry>();

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(scan_topic_,
        rclcpp::SensorDataQoS(), std::bind(&gridmap::scanCallback, this, std::placeholders::_1));
    
    odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic_, 10,
        std::bind(&gridmap::odomCallback, this, std::placeholders::_1));
    
    pub_occ_ = create_publisher<nav_msgs::msg::OccupancyGrid>("mapping/map", qos_map);
    // conversion_tables_ = 
    grid_ = CreateGrid(Eigen::Vector2f::Zero());
    RangeDataInsert_ = std::make_unique<ProbabilityGridRangeDataInserter2D>(hit_, miss_);
    run_thread_ = std::thread(std::bind(&gridmap::run, this));
};

gridmap::~gridmap(){
    run_thread_.join();
}

std::unique_ptr<ProbabilityGrid> gridmap::CreateGrid(const Eigen::Vector2f &origin){
    constexpr int kInitialSubmapSize = 100;
    return std::make_unique<ProbabilityGrid>(
        MapLimits(resolution_, origin.cast<double>() + 
            0.5*kInitialSubmapSize*resolution_*Eigen::Vector2d::Ones(),
        CellLimits(kInitialSubmapSize, kInitialSubmapSize)),
            conversion_tables_);
}

void gridmap::InsertRangeData(RangeData &range_data, 
            const ProbabilityGridRangeDataInserter2D* range_data_inserter){
    CHECK(grid_);
    
    range_data_inserter->Insert(range_data, grid_.get());
}

void gridmap::RosToRangeData(nav_msgs::msg::Odometry odom, sensor_msgs::msg::LaserScan scan,
        RangeData &rangeData){
    rangeData.origin = Eigen::Vector3f(odom.pose.pose.position.x, odom.pose.pose.position.y, 
        odom.pose.pose.position.z);
    
    Eigen::Vector3f translation(rangeData.origin);
    Eigen::Quaternionf quat(odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y, odom.pose.pose.orientation.z);
    Eigen::Matrix3f rotation = quat.toRotationMatrix();

    for(size_t i=0; i<scan.ranges.size(); ++i){
        const float range = scan.ranges[i];
        const float angle = scan.angle_min + i*scan.angle_increment;
        
        if(std::isnan(range) || range < scan.range_min){
            continue;
        }
        
        Eigen::Vector3f point;
        Eigen::Vector3f point_transform;

        if(range <= 5){ // 建图的scan范围控制在5米
            point.x() = range * std::cos(angle);
            point.y() = range * std::sin(angle);
            point.z() = 0.0f;
            point_transform = rotation * point + translation;
            rangeData.returns.emplace_back(point_transform);
        }
        else{
            point.x() = 5 * std::cos(angle);
            point.y() = 5 * std::sin(angle);
            point.z() = 0.0f;
            point_transform = rotation * point + translation;
            rangeData.misses.emplace_back(point_transform);
        } 
    }
}

void gridmap::pubOccMap(std_msgs::msg::Header header, std::unique_ptr<Grid2D> &grid){
    if(!grid){
        RCLCPP_WARN(this->get_logger(), "No vaild map!");
        return; 
    }
    nav_msgs::msg::OccupancyGrid occ_grid;
    occ_grid.header.stamp = header.stamp;
    occ_grid.header.frame_id = "map";
    
    CellLimits cell_limits;
    Eigen::Array2i offset;
    float resolution = grid->limits().resolution();
    int width = grid->limits().cell_limits().num_x_cells;
    int height = grid->limits().cell_limits().num_y_cells;
    grid->ComputeCroppedLimits(&offset, &cell_limits);
    const Eigen::Vector2d max =
        grid->limits().max() - resolution * Eigen::Vector2d(offset.y(), offset.x());
    occ_grid.info.origin.position.x = max.x();
    occ_grid.info.origin.position.y = max.y();
    occ_grid.info.origin.position.z = 0.0;
    occ_grid.info.origin.orientation.w = 0.0;
    occ_grid.info.origin.orientation.x = -0.70710678;
    occ_grid.info.origin.orientation.y = 0.70710678;
    occ_grid.info.origin.orientation.z = 0.0; 
    occ_grid.info.resolution = resolution;
    occ_grid.info.width = width;
    occ_grid.info.height = height;

    occ_grid.data.resize(width*height, -1);
    for(int y=0; y<height; ++y){
        for(int x=0; x<width; ++x){
            float probability = CorrespondenceCostToProbability(grid->GetCorrespondenceCost({x, y}));
            int index = y * width + x;
            if(probability < 0){
                occ_grid.data[index] = -1;
            }
            else{
                occ_grid.data[index] = static_cast<int8_t>(probability*100);
            }
        }
    }
    pub_occ_->publish(occ_grid);
}

void gridmap::run(){
    while(rclcpp::ok()){ 
        if(!scan_data.empty() && !odom_data.empty()){
            while((stamp2Sec(scan_data.front().header.stamp) - 
                stamp2Sec(odom_data.front().header.stamp)) > 0.1){
                odom_data.pop(); 
            }
            auto cur_scan = scan_data.front();
            auto cur_odom = odom_data.front();
            scan_data.pop();
            odom_data.pop();
 
            RangeData curRangeData;
            RosToRangeData(cur_odom, cur_scan, curRangeData); 
            InsertRangeData(curRangeData, RangeDataInsert_.get());
            auto new_grid = grid_->ComputeCroppedGrid();
            pubOccMap(cur_scan.header, new_grid);
        }
    }
}

void gridmap::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg){
    scan_data.push(*msg);
}

void gridmap::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg){ 
    odom_data.push(*msg);    
}


#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(gridmap)