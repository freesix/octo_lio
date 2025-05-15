#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <pcl/filters/passthrough.h>

#include <gtsam/nonlinear/ISAM2.h>

#include "lio_sam/Scancontext.hpp"

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel
using symbol_shorthand::B; // Bias 
using symbol_shorthand::G; // GPS pose

struct PointXYZIRPYT{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, roll, roll)
    (float, pitch, pitch)
    (float, yaw, yaw)
    (double, time, time)
)

typedef PointXYZIRPYT PointTypePose;

enum class SCInputType{
    SINGLE_SCAN_FULL,
    SINGLE_SCAN_FEAT,
    MULTI_SCAN_FEAT
};

class mapOptimization : public ParamServer{

public:
    NonlinearFactorGraph gtSAMgraph; // 因子图
    Values initialEstimate;          // 因子图变量初始值
    Values optimizedEstimate;        // 优化后变量值
    ISAM2* isam;                     // 非线性优化器
    Values isamCurrentEstimate;      // 优化器当前优化结果
    Eigen::MatrixXd poseCovariance;  // 优化器当前优化结果协方差矩阵，其大小影响GPS或者回环加入

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;
    // 发布回环
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;
    // 订阅从特征提取模块中发布出来的点云信息
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS; // GPS数据
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subLoop;

    std::deque<nav_msgs::msg::Odometry> gpsQueue;
    lio_sam::msg::CloudInfo cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 所有关键帧角点点云
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames; // 所有关键帧平面点点云

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses2D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudRaw;
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS;
    double laserCloudRawTime;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // 当前帧角点点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // 当前帧平面点点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // 当前帧角点点云下采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // 当前帧平面点点云下采样

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // 用于并行计算
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec;
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;
    // 变换到odom坐标系下的关键帧点云字典，加速缓存历史关键帧变换后的点云
    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap; // 局部地图角点点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;  // 局部地图平面点点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSC;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // 构建局部地图时挑选关键帧做降采样
    // std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    // std::shared_ptr<tf2_ros::TransformListener> tf_listener;

    rclcpp::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;
    // 当GPS或回环信号加入，标志为true，因子图执行多次优化，然后将所有历史帧位置更新一次。
    bool aLoopIsClosed = false;

    multimap<int, int> loopIndexContainer;

    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;

    vector<gtsam::SharedNoiseModel> loopNoiseQueue;

    deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

    nav_msgs::msg::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> br;

    SCManager scManager;

    std::fstream pgSaveStream;
    std::fstream pgTimeSaveStream;
    std::vector<std::string> edges_str;
    std::vector<std::string> vertices_str;

    std::string saveSCDDirectory;
    std::string saveNodePCDDirectory;

    mapOptimization(const rclcpp::NodeOptions &optiongs) : ParamServer("lio_sam_mapOpt", optiongs){
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry> ("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath = create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);
        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        // tf_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        // tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        subCloud = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos,
            std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        subGPS = create_subscription<nav_msgs::msg::Odometry>(
            gpsTopic, 200,
            std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));

        subLoop  = create_subscription<std_msgs::msg::Float64MultiArray>(
            "lio_loop/loop_closure_detection", qos,
            std::bind(&mapOptimization::loopInfoHandler, this, std::placeholders::_1));

        pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = create_publisher<visualization_msgs::msg::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        const float kSCFilterSize = 0.5; // giseop
        downSizeFilterSC.setLeafSize(kSCFilterSize, kSCFilterSize, kSCFilterSize); // giseop

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());

        saveSCDDirectory = savePCDDirectory + "SCDs/";
        unused = system((std::string("exec rm -r ") + saveSCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveSCDDirectory).c_str());

        saveNodePCDDirectory = savePCDDirectory + "Scans/";
        unused = system((std::string("exec rm -r ") + saveNodePCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveNodePCDDirectory).c_str());

        pgSaveStream = std::fstream(savePCDDirectory + "singlesession_posegraph.txt", std::fstream::out);
        pgTimeSaveStream = std::fstream(savePCDDirectory + "times.txt", std::fstream::out);
        pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    
    }
    /**
     * @brief 内存分配
     */
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses2D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudRaw.reset(new pcl::PointCloud<PointType>()); // giseop
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>()); // giseop

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
    }

    void writeVertex(const int _node_idx, const gtsam::Pose3 &_initPose){
        gtsam::Point3 t = _initPose.translation();
        gtsam::Rot3 R = _initPose.rotation();

        std::string curVertexInfo{
            "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
            + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z()) + " "
            + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
            + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) }; 
        
        vertices_str.emplace_back(curVertexInfo);
    }

    void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose)
    {
        gtsam::Point3 t = _relPose.translation();
        gtsam::Rot3 R = _relPose.rotation();

        std::string curEdgeInfo {
            "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
            + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
            + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
            + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

        edges_str.emplace_back(curEdgeInfo);
    }

    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn){
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw);
        laserCloudRawTime = stamp2Sec(cloudInfo.header.stamp);

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if(timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval){
            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();  // 当前帧位姿初始化

            extractSurroundingKeyFrames(); // 构建局部地图

            downsampleCurrentScan(); // 对当前帧点云做降采样

            scan2MapOptimization(); // 将当前帧点云匹配到构建的局部地图，优化当前位姿

            saveKeyFramesAndFactor(); // 计算是否将当前帧设为关键帧并加入因子图优化

            correctPoses(); // 当新的GPS因子或者回环加入因子图时，对历史帧进行位姿更新

            publishOdometry(); // 发布激光里程计

            publishFrames(); // 发布当前帧对齐到地图坐标系的点云和完整轨迹
        } 
    }

    /**
     * @brief GPS回调函数
     */
    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg){
        gpsQueue.push_back(*gpsMsg); 
    }
    
    /**
     * @brief 回环回调函数
     */
    void loopInfoHandler(const std_msgs::msg::Float64MultiArray::SharedPtr loopMsg){
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if(loopMsg->data.size() != 2){
            return; 
        }
        loopInfoVec.push_back(*loopMsg);
        while(loopInfoVec.size() > 5){
            loopInfoVec.pop_front(); 
        }
    }
    /**
     * @brief 当前帧位姿初始化
     */
    void updateInitialGuess(){
        // 把上一次的激光里程计结果缓存到increamentalOdometryAffineFront中  
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // 第一帧做初始化，直接使用imu输出的RPY作为当前帧的初始位姿，如果不是9轴imu（即不与
        // 地磁角绑定），yaw初始化为0
        if(cloudKeyPoses3D->points.empty()){
            transformTobeMapped[0] = cloudInfo.imu_roll_init;
            transformTobeMapped[1] = cloudInfo.imu_pitch_init;
            transformTobeMapped[2] = cloudInfo.imu_yaw_init;

            if(!useImuHeadingInitialization){
                transformTobeMapped[2] = 0;
            }
            lastImuTransformation = pcl::getTransformation(0, 0, 0,
                cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
            return; 
        }

        // 后续帧位姿初始化
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if(cloudInfo.odom_available == true){
            Eigen::Affine3f transBack = pcl::getTransformation(
                cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
                cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imu_available == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }
 
    }

    /**
     * @brief 构建局部地图，直接调用extracNearby()函数
     */
    void extractSurroundingKeyFrames(){
        if(cloudKeyPoses3D->points.empty() == true){
            return; 
        }

        extractNearby();
    }

    /**
     * @brief 提取与最后一个关键帧空间、时间邻近的帧并提取点云构建局部地图
     */
    void extractNearby(){
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis; 

        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // 构建kd树
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(),
            (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis); // 寻找指定半径内和最后一帧相近的关键帧
        for(int i=0; i<(int)pointSearchInd.size(); ++i){
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);  // 存储搜索到关键帧包含的点云
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS); // 点云降采样
        // for(auto &pt : surroundingKeyPosesDS->points){
        //     kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
        //     pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        // }

        int numPoses = cloudKeyPoses3D->size();
        for(int i=numPoses-1; i>=0; --i){
            if(timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0){
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]); 
            }
            else{
                break; 
            } 
        }

        extractCloud(surroundingKeyPosesDS);
    }
    
    /**
     * @brief 根据传入的关键帧姿态集合，提取点云，转换到odom(Map)坐标系下，融合，降采样。
     * 此函数在里程计线程和回环检测线程中都被使用到
     */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for(int i=0; i<(int)cloudToExtract->size(); ++i){
            if(pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius){
                continue;  // 如果集合中点离最后一帧太远，跳过
            }

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if(laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()){
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second; 
            }
            else{
                pcl::PointCloud<PointType> laserCloudCornerTemp = 
                    *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = 
                    *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap += laserCloudSurfTemp; 
            }
        }
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
        // clear map cache if too large
        if(laserCloudMapContainer.size() > 1000){
            laserCloudMapContainer.clear(); 
        } 
    }

    /**
     * @brief 对当前帧点云做降采样，角点和平面点
     */
    void downsampleCurrentScan(){
        // 对当前帧用于SC匹配的点云做降采样
        laserCloudRawDS->clear();
        downSizeFilterSC.setInputCloud(laserCloudRaw);
        downSizeFilterSC.filter(*laserCloudRawDS);
    
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS); 
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    /**
     * @brief 将当前帧与局部地图匹配，优化当前位姿
     */
    void scan2MapOptimization(){
        if(cloudKeyPoses3D->points.empty()){
            return; 
        } 
        if(laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum){
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS); // 降采样后局部地图角点
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS); // 降采样后局部地图平面点

            for(int iterCount=0; iterCount<30; iterCount++){  // 迭代30次，mapOptmization中改为了20次
                laserCloudOri->clear();
                coeffSel->clear();
                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if(LMOptimization(iterCount) == true){
                    break; 
                }   
            }
            transformUpdate();          
        }
        else{
            RCLCPP_WARN(get_logger(), "Not enough features! Only %d edge and %d planar features available.",
                laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }
    
    /**
     * @brief 计算角点点集中每个点到局部地图中匹配到的之间的距离和法向量
     */
    void cornerOptimization(){
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)  // 好像没起作用
        for(int i=0; i<laserCloudCornerLastDSNum; i++){
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); // 坐标系转换
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));  // 检索出的5个点的协方差矩阵
            cv::Mat matD1(3, 3, CV_32F, cv::Scalar::all(0));  // 协方差矩阵的特征值
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));  // 协方差矩阵的特征向量
            
            if(pointSearchSqDis[4] < 1.0){ // 判断找到的第五个点(距离最大点)小于1m，认为搜索结果有效
                float cx=0, cy=0, cz=0;
                for(int j=0; j<5; j++){
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z; 
                }
                cx /= 5; cy /= 5; cz /= 5;
                float a11=0, a12=0, a13=0, a22=0, a23=0, a33=0;
                for(int j=0; j<5; j++){
                    float ax=laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay=laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az=laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11+=ax*ax; a12+=ax*ay; a13+=ax*az;
                    a22+=ay*ay; a23+=ay*az;
                    a33+=az*az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;
                // 存储协方差的值到matA1
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;
                // 对协方差矩阵做特征值分解，最大特征值对应的特征向量为这5个点的主方向
                cv::eigen(matA1, matD1, matV1);
                // 如果最大的特征值要远大于第二个特征值，则认为5个点能够构成一条直线。
                // 类似PCA主成分分析，数据协方差的最大特征值对应的特征向量为主方向。
                if(matD1.at<float>(0, 0) > 3*matD1.at<float>(0,1)){
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 满足if条件，可以近似对应最大特征值的特征向量(matV1第一行)为5个点
                    // 构成的直线方向。并在这直线上取两点(x1,y1,z1)和(x2,y2,z2)
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                    // 求解向量(x0-x1,y0-y1,z0-z1)和向量(x0-x2,y0-y2,z0-z2)的叉乘结果的向量的模长
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                    // 求解两点之间距离
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                    // 求解直线的法向量(la,lb,lc)
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12; // 点(x0,y0,z0)到直线的距离
                    // 根据点到直线距离，赋予s的大小(比重)
                    float s = 1 - 0.9 * fabs(ld2);
                    // 使用系数对法向量进行加权，相当于对导数(雅可比矩阵加权)
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                    // 经验阈值，判断点到直线的距离是否足够近，足够近才采纳为优化目标点
                    if(s>0.1){
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true; 
                    }
                } 
            }
        }
    }

    /**
     * @brief 计算平面点点集中每个点到局部地图中匹配到的平面的距离和法向量
     */
    void surfOptimization(){
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for(int i=0; i<laserCloudSurfLastDSNum; i++){
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i]; 
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            // 下面过程是求解Ax+By+Cz+1=0的平面方程
            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;
                // (pa,pb,pc)是平面的法向量，这里对法向量进行归一化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        } 
    }

    /**
     * @brief 将cornerOptimization和surfOptimization两个函数计算出来的角点、平面点、到
     * 局部地图的距离、法向量集合在一起
     */
    void combineOptimizationCoeffs(){
        for(int i=0; i<laserCloudCornerLastDSNum; ++i){
            if(laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]); 
            } 
        }
        for(int i=0; i<laserCloudSurfLastDSNum; ++i){
            if(laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]); 
            } 
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false); 
    }

    /**
     * @brief 这部分代码是基于高斯牛顿法的优化。目标函数(点到线、点到平面的距离)对位姿(rpy、
     * tx、ty、tz的表达)求导。计算高斯牛顿更新方向和步长，然后对transformTobeMapped进行
     * 更新。
     * @param iterCount 迭代更新次数，默认30
     */
    bool LMOptimization(int iterCount){
        // 计算三轴欧拉角的sin、cos 后续会用到
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if(laserCloudSelNum < 50){ // 如果参与优化的点少于50，跳过此次优化
            return false; 
        }

        // matA是Jacobians矩阵
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        // matB是目标函数
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        // matX是高斯-牛顿法计算出的更新向量
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for(int i=0; i<laserCloudSelNum; i++){
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;

            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity; 
            // arx是目标函数相对于roll的导数
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;
            // ary是目标函数相对于pitch的导数
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                + ((-cry*crz - srx*sry*srz)*pointOri.x 
                + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
            // arz是目标函数相对于yaw的导数
            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            // 目标函数相对于tx的导数等于法向量的x
            matA.at<float>(i, 3) = coeff.z;
            // 目标函数相对于ty的导数等于法向量的y
            matA.at<float>(i, 4) = coeff.x;
            // 目标函数相对于tz的导数等于法向量的z
            matA.at<float>(i, 5) = coeff.y;

            // matB存储的是目标函数（距离）的负值，因为：J^{T}J\Delta{x} = -Jf(x)
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        cv::transpose(matA, matAt);
        matAtA = matAt*matA;
        matAtB = matAt*matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
        // 如果是第一次迭代，判断求解出的近似Hessian矩阵是否退化
        if(iterCount == 0){
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV); // 对海森矩阵进行特征值分解
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for(int i=5; i>=0; i--){
                if(matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                }else {
                    break;
                }
            }
            matP = matV.inv() * matV2; 
        }
        // 当第一次迭代判断到海森矩阵退化，后面会使用计算出来的权重matP对增量matX做加权
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        // 将增量叠加到变量(位姿)中
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
        // 计算R、P、Y的迭代步长
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        // 计算tx、ty、tz的迭代步长
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
        // 如果迭代的步长达到设定阈值，认为优化已经收敛
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    /**
     * @brief 使用IMU的原始输出rpy于当前估计的rpy加权融合，注意只对rpy加权融合，同时有一个
     * 权重控制IMU的比重(默认0.01)
     */
    void transformUpdate(){
        if(cloudInfo.imu_available == true){
            if(std::abs(cloudInfo.imu_pitch_init) < 1.4){
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        // 这里就已经记录了incrementalOdometryAffineBack，后续被用来计算odometry_incremental
        // 也就是说，incremental的激光里程计没有使用到因子图优化的结果，因此更加光滑，没有跳变，可以给IMU预积分模块使用
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped); 
    }
    /**
     * @brief 角度限制函数，配置文件中可以选择对roll、pitch角做限制
     */
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;
        return value;
    }


    /**
     * @brief 添加因子并执行图优化，更新当前位姿
     */
    void saveKeyFramesAndFactor(){
        // 是否将当前帧列为关键帧
        if(saveFrame() == false){
            return; 
        }
        // 添加激光里程计因子
        addOdomFactor();

        // 添加GPS因子
        addGPSFactor();

        // 添加回环因子
        addLoopFactor();
        // 迭代一次优化器
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        // 如果但前帧有新的gps因子或回环因子加入，执行多次迭代更新，且后面会更新所有的历史帧
        if(aLoopIsClosed == true){
            isam->update();
            isam->update();
            isam->update(); 
            isam->update(); 
            isam->update(); 
        }
        // 清空因子图和初始值
        gtSAMgraph.resize(0);
        initialEstimate.clear();
        // 从优化器中拿出当前帧的位姿存入关键帧位姿序列
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
        // 从优化器中拿出最近一帧的优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // 将当前帧经过优化的结果存入关键帧位姿序列
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.roll = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();
        // 将当前帧的角点，平面点点云存入关键帧点云缓存队列
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // Scan Context
        const SCInputType sc_input_type = SCInputType::SINGLE_SCAN_FULL;
        if(sc_input_type == SCInputType::SINGLE_SCAN_FULL){
            pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudRawDS, *thisRawCloudKeyFrame);
            scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame); 
        }
        else if(sc_input_type == SCInputType::SINGLE_SCAN_FEAT){
            scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame); 
        }
        else if(sc_input_type == SCInputType::MULTI_SCAN_FEAT){
            pcl::PointCloud<PointType>::Ptr multiKeyFrameFeatureCloud(new pcl::PointCloud<PointType>());
            loopFindNearKeyframes(multiKeyFrameFeatureCloud, cloudKeyPoses6D->size()-1, historyKeyframeSearchNum);
            scManager.makeAndSaveScancontextAndKeys(*multiKeyFrameFeatureCloud);
        }

        // 保存sc
        const auto &curr_scd = scManager.getConstRefRecentSCD();
        std::string curr_scd_node_idx = padZeros(scManager.polarcontexts_.size()-1);
        saveSCD(saveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);

        // 保存pcd
        bool saveRawCloud{true};
        pcl::PointCloud<PointType>::Ptr thisKeyFrameCloud(new pcl::PointCloud<PointType>());
        if(saveRawCloud){
            *thisKeyFrameCloud += *laserCloudRaw; 
        }
        else{
            *thisKeyFrameCloud += *thisCornerKeyFrame;
            *thisKeyFrameCloud += *thisSurfKeyFrame;
        }
        pcl::io::savePCDFileBinary(saveNodePCDDirectory + curr_scd_node_idx + ".pcd", *thisKeyFrameCloud);
        pgTimeSaveStream << laserCloudRawTime << std::endl;

        updatePath(thisPose6D);
    }
 
    /**
     * @brief 判断是否将当前帧选择为关键帧
     * @details 当距离不够且角度不够时，不会将当前帧选择为关键帧。对于非关键帧的点云，只做
     * 点云配准校准里程计，而关键帧则会加入因子图进行优化
     */
    bool saveFrame(){
        if(cloudKeyPoses3D->points.empty()){
            return true; 
        }
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back()); // 获取上一关键帧位姿
        Eigen::Affine3f transFinal = pcl::getTransformation(
            transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); 

        if(abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingAngleThreshold){
            return false;     
        }
        return true;
    }
    
    /**
     * @brief 添加激光里程计因子
     * @details 若是第一帧，则构建prior因子，赋予较大的方差。后续帧则根据当前的位姿估计，
     * 以及上一个关键帧的位姿，计算位姿增量，添加间隔因子，同时将当前帧的位姿估计作为因子图
     * 当前变量的初始值。
     */
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));

            writeVertex(0, trans2gtsamPose(transformTobeMapped));

        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtsam::Pose3 relPose = poseFrom.between(poseTo);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), relPose, odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);

            writeVertex(cloudKeyPoses3D->size(), poseTo);
            writeEdge({cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size()}, relPose); // giseop
        }
    }

    /**
     * @brief 添加GPS因子
     * @details GPS队列为空或者关键帧序列为空，或者前后关键帧距离小于5m，或者位姿协方差较小，
     * 直接返回，认为不需要加入GPS矫正。从GPS队列中找到与当前帧时间最接近得GPS数据。GPS数据
     * 方差大于阈值或者与上一次采用的GPS位置小于5m，直接返回。从GPS数据中提取x、y、z和协方差，
     * 构建GPS因子加入因子图，其中的z坐标可以设置为不使用GPS的输出。设置aLoopGpsIsClosed
     * 标志位为true，后面因子图优化时会多次迭代且更新所有历史关键帧位姿
     */
    void addGPSFactor(){
        if(gpsQueue.empty()){
            return; 
        }
        if(cloudKeyPoses3D->points.empty()){
            return; 
        }
        else{
            if(pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0){
                return; 
            } 
        }
        if(poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold){
            return; 
        }

        // last gps position
        static PointType lastGPSPoint;

        while(!gpsQueue.empty()){
            if(stamp2Sec(gpsQueue.front().header.stamp) < timeLaserInfoCur - 0.2){  // GPS队列中数据太老
                gpsQueue.pop_front(); 
            }
            else if(stamp2Sec(gpsQueue.front().header.stamp) > timeLaserInfoCur + 0.2){ // GPS队列中数据太新
                break; 
            }
            else{
                nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if(noise_x > gpsCovThreshold || noise_y > gpsCovThreshold){  // 当前GPS信号噪声太大
                    continue; 
                }
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6){
                    continue;
                }

                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if(pointDistance(curGPSPoint, lastGPSPoint) < 5.0){
                    continue; 
                }
                else{
                    lastGPSPoint = curGPSPoint;
                }

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    /**
     * @brief 添加回环检测因子
     */
    void addLoopFactor(){
        if(loopIndexQueue.empty()){
            return; 
        }

        for(int i=0; i<(int)loopIndexQueue.size(); ++i){
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            auto noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));

            writeEdge({indexFrom, indexTo}, poseBetween); // 保存回环关系
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();

        aLoopIsClosed = true; 
    }
    
    /**
     * @brief 将传入的位姿加入轨迹
     */
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf2::Quaternion q;
        q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();
 
        globalPath.poses.push_back(pose_stamped);
    }

    /**
     * @brief 更新所有历史关键帧位姿
     */
    void correctPoses(){
        if(cloudKeyPoses3D->points.empty()){
            return; 
        }
        if(aLoopIsClosed == true){ // 有新的回环或GPS加入因子图
            laserCloudMapContainer.clear(); // 此处缓存的转换到map下的点云，在更新轨迹后要清空
            globalPath.poses.clear(); // 把缓存的轨迹清空
            // 从优化器中拿出历史所有时刻的位姿并更新到位姿序列
            int numPoses = isamCurrentEstimate.size();
            for(int i=0; i<numPoses; ++i){
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z(); 

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }
            aLoopIsClosed = false; 
        } 
    }

    /**
     * @brief 发布激光里程计
     * @details 1、发布当前帧位姿。2、发布TF坐标系
     */
    void publishOdometry(){
        nav_msgs::msg::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        tf2::Quaternion quat_tf;
        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        geometry_msgs::msg::Quaternion quat_msg;
        tf2::convert(quat_tf, quat_msg);
        laserOdometryROS.pose.pose.orientation = quat_msg;
        if (isDegenerate){
            laserOdometryROS.pose.covariance[0] = 1;
        }
        else{
            laserOdometryROS.pose.covariance[0] = 0;
        }
        pubLaserOdometryGlobal->publish(laserOdometryROS);

        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        tf2::Transform t_odom_to_lidar = tf2::Transform(quat_tf, tf2::Vector3(
            transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
        tf2::Stamped<tf2::Transform> temp_odom_to_lidar(t_odom_to_lidar, time_point, odometryFrame);
        geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
        tf2::convert(temp_odom_to_lidar, trans_odom_to_lidar);
        trans_odom_to_lidar.child_frame_id = "base_link";
        br->sendTransform(trans_odom_to_lidar);

        // publish odometry for ROS (incremental)
        // 发布光滑的雷达里程计结果，貌似用处不大，作者在issue中也提到了
        static bool lastIncreOdomPubFlag = false;

        static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imu_available == true)
            {
                if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf2::Quaternion imuQuaternion;
                    tf2::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(roll, pitch, yaw);
            geometry_msgs::msg::Quaternion quat_msg;
            tf2::convert(quat_tf, quat_msg);
            laserOdomIncremental.pose.pose.orientation = quat_msg;
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental->publish(laserOdomIncremental);
    }

    /**
     * @brief 发布点云及轨迹信息
     */
    void publishFrames(){
        if(cloudKeyPoses3D->points.empty()){
            return; 
        }
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr cloudtmp(new pcl::PointCloud<PointType>());
            pcl::PassThrough<PointType> pass;
            pass.setFilterFieldName("z");
            pass.setFilterLimits(-std::numeric_limits<float>::max(), 1.5);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            // *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            cloudtmp = transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
            pass.setInputCloud(cloudtmp);
            pcl::PointCloud<PointType>::Ptr cloudCornerFiltered(new pcl::PointCloud<PointType>());
            pass.filter(*cloudCornerFiltered);
            *cloudOut += *cloudCornerFiltered;

            // 处理 surf 点云
            cloudtmp = transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
            pass.setInputCloud(cloudtmp);
            pcl::PointCloud<PointType>::Ptr cloudSurfFiltered(new pcl::PointCloud<PointType>());
            pass.filter(*cloudSurfFiltered);
            *cloudOut += *cloudSurfFiltered;
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath->get_subscription_count() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath->publish(globalPath);
        } 
    }


    void updatePointAssociateToMap(){
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped); 
    }
    /**
     * @brief 将雷达坐标系下的点转换到地图坐标系
     */
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    /**
     * @brief 将输入点云按照转换矩阵进行转换
     */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    /**
     * @brief 将6D位姿转化成gtsam位姿的函数
     * @param thisPoint (x,y,z,r,p,y)6D位姿
     * @return gtsam位姿表示
     */
    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }
    /**
     * @brief 将数组表示的6D位姿转换成gtsam的位姿表示
     */
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }
    /**
     * @brief 将PointTypePose表示的6D位姿转换为Eigen表示
     */
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }
    /**
     * @brief 将数组表示的6D位姿转换为Eigen表示
     */
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }
    /**
     * @brief 将数组表示的6D位姿转换为PointTypePose
     */
    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }
    
    /**
     * @brief 回环中寻找当前帧与对应帧的索引
     */
    bool detectLoopClosureExternal(int *latestID, int *closestID){
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if(loopInfoVec.empty()){
            return false; 
        }

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();
        // 匹配帧之间时间过短，返回
        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff){
            return false;
        }
        // 关键帧数量少于2，返回
        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2){
            return false;
        }
        // 下面是获取最后一个关键帧的时间戳
        loopKeyCur = cloudSize - 1;
        for(int i=cloudSize-1; i>=0; --i){
            if(copy_cloudKeyPoses6D->points[i].time >= loopTimeCur){
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity); 
            }
            else{
                break; 
            } 
        }
        // 获取最早于回环时间的关键帧时间戳
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre){
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            }
            else{
                break;
            }
        }
        // 两帧时间相同，返回false
        if(loopKeyCur == loopKeyPre){
            return false; 
        }
        // 已经检测过回环了，返回false
        auto it = loopIndexContainer.find(loopKeyCur);
        if(it != loopIndexContainer.end()){
            return false; 
        }

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * @brief 根据位置关系寻找当前帧对应回环帧的索引
     */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // 确认最后一帧关键帧没有被加入过回环关系
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop; // unused 
        // kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
 
        for (int i = 0; i < (int)copy_cloudKeyPoses2D->size(); i++) // giseop
            copy_cloudKeyPoses2D->points[i].z = 1.1; // to relieve the z-axis drift, 1.1 is just foo val

        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses2D); // giseop
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses2D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0); // giseop
        
        // std::cout << "the number of RS-loop candidates  " << pointSearchIndLoop.size() << "." << std::endl;
        // 确保空间距离相近的帧是较久前采集的，排除是前面几个关键帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }
        // 如果没有找到位置关系、时间关系都符合要求的关键帧，则返回false
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * @brief 根据当前帧索引key，从前后多帧构建局部地图
     * @param nearKeyframes 传出参数，构建出的局部地图
     * @param key 当前帧的索引
     * @param searchNum 从当前帧前后各searchNum个关键帧构建局部地图
     */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum){
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for(int i=-searchNum; i<=searchNum; ++i){
            int keyNear = key+i;
            if(keyNear<0 || keyNear >= cloudSize){
                continue; 
            }
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); 
        } 

        if(nearKeyframes->empty()){
            return; 
        }

        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void loopFindNearKeyframesWithRespectTo(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
        const int &key, const int &searchNum, const int _wrt_key){

        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for(int i=-searchNum; i<=searchNum; ++i){
            int keyNear = key+i;
            if(keyNear<0 || keyNear>=cloudSize){
                continue; 
            }
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[_wrt_key]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[_wrt_key]);
        }
        if(nearKeyframes->empty()){
            return; 
        }

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;     
    }


    /**
     * @brief 回环检测函数，ICP方法的回环检测
     * @details 将最后一帧关键帧作为当前帧进行回环检测，如果当前帧已经在回环关系中则返回，
     * 如果找到的回环对应帧相差时间过短也返回false，回环关系用一个map存储。
     */
    void performRSLoopClosure(){
        if(cloudKeyPoses3D->points.empty() == true){  // 关键帧队列为空，直接返回
            return;  
        }
        // 加锁拷贝关键帧3D和6D位姿，避免多线程干扰
        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        copy_cloudKeyPoses2D->clear();
        *copy_cloudKeyPoses2D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if(detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false){
            if(detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false){
                return; 
            } 
        }
        RCLCPP_INFO(get_logger(), "RS loop found between %i and %i", loopKeyCur, loopKeyPre);

        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {   
            // 将当前帧转换到map坐标系并降采样，(这里0表示不加上前后其它帧)
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 对匹配的前后几帧转换到map坐标系下融合降采样，构建局部地图
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if(cureKeyframeCloud->size()<300 || prevKeyframeCloud->size() < 1000){
                return; 
            }
            if(pubHistoryKeyFrames->get_subscription_count() != 0){
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame); 
            }
        }

        // 调用ICP将当前帧匹配到局部地图，得到当前帧的位姿偏差，将偏差应用到当前帧的位姿，
        // 得到修正后的当前帧位姿。
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150);  // use a value can cover 2*HistoryKeyframeSearchNum range in meter
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);
        // 判断icp是否收敛和得分是否满足要求(判断是否有icp回环)
        if(icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore){
            RCLCPP_INFO(get_logger(), "ICP fitness test failed %d > %f. Reject this RS loop",
                icp.getFitnessScore(), historyKeyframeFitnessScore); 
            return;
        }
        else{
            RCLCPP_INFO(get_logger(), "ICP fitness test passed %d < %f. Add this RS loop",
                icp.getFitnessScore(), historyKeyframeFitnessScore); 
        }

        if(pubIcpKeyFrames->get_subscription_count() != 0){
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame); 
        }
        // 根据修正后的当前帧位姿和匹配帧的位姿，计算帧间相对位姿，这个位姿用作回环因子，同时将icp的匹配分数当作因子的噪声模型
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation(); // 获取回环帧之间的位姿偏差

        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6); 

        // 将回环索引、回环间相对位姿、回环噪声模型加入全局变量
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
    }

    /**
     * @brief SC方法的回环检测
     */
    void performSCLoopClosure(){
        if(cloudKeyPoses3D->points.empty() == true){
            return; 
        }
        // 寻找关键帧
        auto detectResult = scManager.detectLoopClosureID(); // 寻找当前帧与历史帧sc相匹配的帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; // 当前关键帧ID
        int loopKeyPre = detectResult.first;  // 匹配到的关键帧ID
        float yawDiffRad = detectResult.second; // 暂时没有使用，还有点问题
        if(loopKeyPre == -1){ // 没有找到回环
            return; 
        }
        RCLCPP_INFO(get_logger(), "SC loop found between %i and %i .", loopKeyCur, loopKeyPre);
        // 提取点云
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            int base_key = 0;
            loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, base_key);
            loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key);

            if(cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000){
                return; 
            }
            if(pubHistoryKeyFrames->get_subscription_count() != 0){
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
            }
        }

        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);
        // TODO icp align with initial

        if(icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore){
            RCLCPP_INFO(get_logger(), "ICP fitness test failed %d > %f. Reject this SC loop",
                icp.getFitnessScore(), historyKeyframeFitnessScore); 
            return;
        }
        else{
            RCLCPP_INFO(get_logger(), "ICP fitness test passed %d < %f. Add this SC loop",
                icp.getFitnessScore(), historyKeyframeFitnessScore); 
        }

        if(pubIcpKeyFrames->get_subscription_count() != 0){
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame); 
        }

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

        float robustNoiseScore = 0.5;
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        noiseModel::Base::shared_ptr robustConstraintNoise;
        robustConstraintNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
        ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(robustConstraintNoise);
        mtx.unlock();
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
    }

    /**
     * @brief 可视化回环关系，在Rivz中
     */
    void visualizeLoopClosure(){
        visualization_msgs::msg::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::msg::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::msg::Marker::ADD;
        markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::msg::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::msg::Marker::ADD;
        markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1; markerEdge.scale.y = 0.1; markerEdge.scale.z = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::msg::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge->publish(markerArray);
    }

    /**
     * @brief 回环检测独立线程
     * @details 由于回环检测较为耗时，所以单独用作一个线程运行。新的回环关系检测出来后被主
     * 线程加入因子图中优化
     */
    void loopClosureThread(){
        if(loopClosureEnableFlag == false){ // 不开启回环检测
            return; 
        }
        rclcpp::Rate rate(loopClosureFrequency); // 控制回环检测的频率
        while(rclcpp::ok()){
            rate.sleep();
            performRSLoopClosure();
            performSCLoopClosure();  // SC方法的回环检测
            visualizeLoopClosure(); 
        }
    }

    /**
     * @brief 可视化全局地图的独立线程。由于全局地图点云数较多，且需要多帧拼接，因此在独立
     * 线程中完成。
     */
    void visualizeGlobalMapThread(){
        rclcpp::Rate rate(0.2);
        while(rclcpp::ok()){
            rate.sleep();
            publishGlobalMap(); 
        }

        if(savePCD == false){
            return; 
        }
        
        cout << "*****************************************" << endl;
        cout << "Saving the posegraph ... " << endl;
        for(auto &_line : vertices_str){
            pgSaveStream << _line << std::endl; 
        }
        for(auto &_line : edges_str){
            pgSaveStream << _line << std::endl; 
        }
        pgSaveStream.close();

        cout << "*****************************************" << endl;
        cout << "Saving map to pcd files ... " << endl;
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for(int i=0; i<(int)cloudKeyPoses3D->size(); i++){
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    /**
     * @brief 发布全局地图点云
     * @details 对所有关键帧3D位姿构建kd树。以最后一帧关键帧为索引找出一定半径范围内的关键帧。
     * 对找出的关键帧做降采样。对所有关键帧的点云拼接(投影到map坐标系)。对地图点云做降采样。
     * 发布全局地图点云。
     */
    void publishGlobalMap(){
        if(pubLaserCloudSurround->get_subscription_count() == 0){
            return; 
        }
        if(cloudKeyPoses3D->points.empty() == true){
            return; 
        }

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, 
            pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for(int i=0; i<(int)pointSearchIndGlobalMap.size(); ++i){
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]); 
        }
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity,
            globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity);
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius){
                continue;
            }
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }
};



int main(int argc, char** argv){
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Map Opt Started.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, MO);
    std::thread visualizeMapthread(&mapOptimization::visualizeGlobalMapThread, MO);

    exec.spin();

    rclcpp::shutdown();

    loopthread.join();
    visualizeMapthread.join();

    return 0;
}

