#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"
// 激光点曲率和索引
struct smoothness_t{ 
    float value;
    size_t ind;
};
/**
 * @brief 对激光点曲率从小到大排序
 */
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:
    // 订阅从imageProjection发布出来的去畸变点云信息
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo;
    
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo; // 发布增加了角点、平面点的点云信息
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints; // 发布角点
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints; // 发布平面点

    pcl::PointCloud<PointType>::Ptr extractedCloud; // 当前激光帧去畸变后的有效点云
    pcl::PointCloud<PointType>::Ptr cornerCloud;    // 当前激光帧提取的角点点云
    pcl::PointCloud<PointType>::Ptr surfaceCloud;   // 当前激光帧提取的平面点云

    pcl::VoxelGrid<PointType> downSizeFilter;       // 用来做平面特征点点云降采样的过滤器

    lio_sam::msg::CloudInfo cloudInfo;              // 最终要发布的数据包
    std_msgs::msg::Header cloudHeader;              // 接收到的当前帧信息的数据头

    std::vector<smoothness_t> cloudSmoothness;      // 点云曲率存储
    float *cloudCurvature;                          // 点云曲率，长度为N_SCAN*Horizon_SCAN的数组
    int *cloudNeighborPicked; // 特征提取标志为，1表示遮挡、平行或者已经进行特征提取的点，0表示未进行特征提取
    int *cloudLabel;          // 1表示角点、-1表示平面点、0表示没有被选择为特征点                         

    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("lio_sam_featureExtraction", options)
    {   // 订阅点云信息(运动畸变矫正后的)
        subLaserCloudInfo = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos,
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));
        // 发布经过特征提取后的点云信息
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos);
        pubCornerPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_surface", 1);
        // 初始化向量、数组以及标志位的值
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);
        // 平面点云降采样器，默认voxel是0.4米
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }
    /**
     * @brief cloudInfo的回调函数
     * @details 接收从imaeProjection中发布出来的一个去畸变点云信息
     * 计算每个点的曲率
     * 标记遮挡点、平行点，后续这些点不会被采纳为特征点
     * 特征提取
     * 混合信息，发布自定的数据包话题
     * @param msgIn 接收到的去畸变点云信息
     */
    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction
        // 计算每个点云的曲率
        calculateSmoothness();
        // 标记遮挡点和平行点，后续这些点将不被采纳为特征点
        markOccludedPoints();
        // 提取特征
        extractFeatures();
        // 整合信息并发布
        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {   // 注释掉的为作者采用的10个点，角点数量会少
            // float diffRange = cloudInfo.point_range[i-5] + cloudInfo.point_range[i-4]
            //                 + cloudInfo.point_range[i-3] + cloudInfo.point_range[i-2]
            //                 + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 10
            //                 + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2]
            //                 + cloudInfo.point_range[i+3] + cloudInfo.point_range[i+4]
            //                 + cloudInfo.point_range[i+5];
            // 此处采样4个(可根据场景算法来调)
            float diffRange = 
                            cloudInfo.point_range[i-2]  + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 4
                            + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2];    

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            // 清空点云的各个标志位
            cloudNeighborPicked[i] = 0; 
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting 将曲率和索引保存，用于后面的排序
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }
    /**
     * @brief 标记遮挡点和平行的点，将cloudNeighborPicked标志位设置为1
     * @details 对于遮挡物体，会呈现一条类似边缘的线，与其左边或者右边的点会呈现断层状态，可通过rang的差距判断
     *          对于与激光线平行的点，左右几个点之间的rang差距应该都会比较大
     */
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points 标记遮挡点和平行点
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.point_range[i];
            float depth2 = cloudInfo.point_range[i+1];
            int columnDiff = std::abs(int(cloudInfo.point_col_ind[i+1] - cloudInfo.point_col_ind[i]));
            if (columnDiff < 10){  // 两个点之间的列索引小于10个像素，认为是同一块区域
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){  // 当前点距离大于右边点0.3m，则认为当前点及左边6个点无效
                    // cloudNeighborPicked[i - 5] = 1;
                    // cloudNeighborPicked[i - 4] = 1;
                    // cloudNeighborPicked[i - 3] = 1;
                    // cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){ // 当前点距离小于右边点距离0.3m，则认为右边6个点无效
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    // cloudNeighborPicked[i + 3] = 1;
                    // cloudNeighborPicked[i + 4] = 1;
                    // cloudNeighborPicked[i + 5] = 1;
                    // cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]));
            float diff2 = std::abs(float(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]));
            // 当前点与左右两点的距离均大于阈值，则认为当前点是处于平行面的点
            if (diff1 > 0.1 * cloudInfo.point_range[i] && diff2 > 0.1 * cloudInfo.point_range[i])
                cloudNeighborPicked[i] = 1;
        }
    }
    /**
     * @brief 特征点提取，遍历扫描线，每跟线扫描的一周点云划分为6段
     * @details 1、提取角点：针对每段最多20个角点。设置cloudLabel=1标志该点为角点，cloudNeighorPicked=1
     * 标志该点已经进行过特征提取。对该角点左右各5个点，如果两点之间的索引小于10，则抛弃
     * 周围的点，避免对同一块区域重复提取角点。
     * 2、提取平面点：不限制平面点的数量，后面会对其进行降采样。设置cloudLabel=-1标志为平面
     * 点，cloudNeighorPicked=1表示已经进行过特征提取。对该平面点左右各5个点如果列索引小于
     * 10，则抛弃周围点，避免对同一块区域提取平面点。
     * 3、将该段中除了角点之外的点加入平面点集合。！感觉这样（2）有点多余。
     * 4、将平面特征点进行降采样。
     */
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());
        // 将每一条激光线分成6段
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {
                // 计算该段的起始点和结束点的索引
                int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
                int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;
                // 对当前段的雷达点云曲率进行排序
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());
                // 角点提取
                int largestPickedNum = 0;
                // 由于cloudSmoothness是由小到大排序，那么这里对每段就是从大到小遍历
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) // 未被pick并且区域大于角点阈值
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 40){ // 对于每段最多提取40个角点（原代码20个）
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        // 对于该角点左右各5个点，如果两点之间的列索引小于10，则抛弃，避免重复提取
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 平面点特征提取，从sp到ep，曲率由小到大遍历。不限制平面点数量
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 该段中除角点外点加入平面点集合。！！！
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }
            // 对平面点集合做采样
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }
    /**
     * @brief 清空准备发布的点云信息中无效的字段，节省数据量
     */
    void freeCloudInfoMemory()
    {
        cloudInfo.start_ring_index.clear();
        cloudInfo.end_ring_index.clear();
        cloudInfo.point_col_ind.clear();
        cloudInfo.point_range.clear();
    }
    /**
     * @brief 发布特征提取后的点云
     */
    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo->publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto FE = std::make_shared<FeatureExtraction>(options);

    exec.add_node(FE);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Feature Extraction Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
