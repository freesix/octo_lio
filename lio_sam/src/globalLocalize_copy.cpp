#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"
#include "lio_sam/Scancontext.hpp"

#include <filesystem>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/ndt.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
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
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel
using symbol_shorthand::B; // Bias (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGEN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
(float, x, x) (float, y, y) (float, z, z) 
(float, intensity, intensity)
(float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
(double, time, time))

typedef PointXYZIRPYT PointTypePose; // 将自定义注册的pcl点云类型取个别名

class mapOptimization : public ParamServer
{   
public:

    NonlinearFactorGraph gtSAMgraph; // 非线性因子图
    Values initialEstimate;           // 初始估计
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMappedROS;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses; // 发布关键帧
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;  // 发布轨迹
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;

    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo; // 订阅特征提取后的点云信息
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS; // 订阅GPS信息

    std::deque<nav_msgs::msg::Odometry> gpsQueue; // GPS消息队列
    lio_sam::msg::CloudInfo cloudInfo;  // 当前点云信息

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 所有关键帧的角点点云
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 所有关键帧的平面点点云

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; // 所有关键帧的三维位姿(x,y,z)
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 所有关键帧的六维位姿(x,y,z,r,p,y)

    std::mutex mtxWin;
    std::deque<PointType> win_cloudKeyPoses3D;
    // pcl::PointCloud<PointType> win_cloudKeyPoses3D;
    std::deque<PointTypePose> win_cloudKeyPoses6D;
    // pcl::PointCloud<PointTypePose> win_cloudKeyPoses6D;

    std::deque<pcl::PointCloud<PointType>::Ptr> win_cornerCloudKeyFrames;
    std::deque<pcl::PointCloud<PointType>::Ptr> win_surfCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // 当前帧的角点点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;   // 当前帧平面点点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // 当前帧的角点点云降采样器
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;  // 当前帧的平面点点云降采样器

    //  在做点云配准过程中使用的中间变量
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;
    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    // 局部地图角点点云(odom坐标系)
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    // 局部地图平面点点云(odom坐标系)
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    // 局部地图角点点云降采样
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    // 局部地图平面点点云降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;  // 角点点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterSurf;    // 平面点点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterICP;     // 做回环检测时使用ICP时的点云将采样器
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;  // 构建局部地图时挑选关键帧做降采样

    rclcpp::Time timeLaserInfoStamp; //当前雷达帧的时间戳
    double timeLaserInfoCur; // 当前雷达帧的时间戳(double类型)
    // 这个变量很重要，缓存的是当前帧“最新”位姿，无论在哪个环节对位姿的更新都会被缓存到此变量给下个环节使用
    float transformTobeMapped[6];

    std::mutex mtx;  // 点云信息回调函数锁

    double timeLastProcessing = -1;
    
    bool isDegenerate = false; // 此标志位标识点云匹配结果是否较差，true表示差，会在imuPreintegration中根据此选择不同的噪声模型给因子图
    Eigen::Matrix<float, 6, 6> matP;

    int winSize = 30;
    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    int imuPreintegrationResetId = 0;

    nav_msgs::msg::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront; // 用于增量发布，没用但保留
    Eigen::Affine3f incrementalOdometryAffineBack;

    pcl::PointCloud<PointType>::Ptr cloudGlobalMap; //  全局地图
    pcl::PointCloud<PointType>::Ptr cloudGlobalMapDS; // 降采样后的全局地图
    pcl::PointCloud<PointType>::Ptr cloudScanForInitialize;
    pcl::PointCloud<PointType>::Ptr laserCloudRaw;
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS;
    Eigen::MatrixXd matrix_poses;

    double laserCloudRawTime;

    pcl::VoxelGrid<PointType> downSizeFilterSC;

    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr subIniPoseFromRviz;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudInWorld;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMapWorld;

    float transformInTheWorld[6];// the pose in the world, i.e. the prebuilt map
    float transformOdomToWorld[6];
    int globalLocaSkipFrames = 3;
    int frameNum = 1;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tfOdom2Map;
    std::mutex mtxtranformOdomToWorld;
    std::mutex mtx_general;
    bool globalLocalizeInitialized = false;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;

    enum InitializedFlag{
        NonInitialized,
        Initializing,
        Initialized 
    };
    InitializedFlag initializedFlag;

    geometry_msgs::msg::PoseStamped poseOdomToMap;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pubOdomToMapPose;

    std::string loadSCDDirectory;
    std::string loadNodePCDDirectory;
    std::string loadPosesDirectory;
    SCManager scManager;            //create a SCManager  object

    std::unique_ptr<tf2_ros::TransformBroadcaster> br;
    
    mapOptimization(const rclcpp::NodeOptions &options) : ParamServer("lio_sam_globalLocalize", options){
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/trajectory", 1);  
        pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubOdomAftMappedROS = create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry", qos);
        pubPath = create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);

        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        subLaserCloudInfo = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos,
            std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        subGPS = create_subscription<nav_msgs::msg::Odometry>(
            gpsTopic, 200,
            std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));
        subIniPoseFromRviz = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 8,
            std::bind(&mapOptimization::initialpose_callback, this, std::placeholders::_1)); // 从rviz订阅初始位置
        
        pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos);

        pubMapWorld = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_map_map", 1);
        pubLaserCloudInWorld = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/lasercloud_in_world", 1);
        pubOdomToMapPose = create_publisher<geometry_msgs::msg::PoseStamped>("lio_sam/mapping/pose_odomTo_map", 1);

        pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);

        pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);

        loadSCDDirectory = std::getenv("HOME") + savePCDDirectory + "SCDs/";
        loadNodePCDDirectory = std::getenv("HOME") + savePCDDirectory + "Scans/";
        int count = 0;
        for(const auto &entry : std::filesystem::directory_iterator(loadSCDDirectory)){
            if(std::filesystem::is_regular_file(entry)){count++;}
        }
        for(int i=0; i<count; ++i){
            std::string filename = padZeros(i);
            std::string scd_path = loadSCDDirectory + filename + ".scd";
            Eigen::MatrixXd load_sc;
            loadSCD(scd_path, load_sc);
            // std::cout << "load_sc: \n" << load_sc << endl;

            // load key
            Eigen::MatrixXd ringkey = scManager.makeRingkeyFromScancontext(load_sc);
            Eigen::MatrixXd sectorkey = scManager.makeSectorkeyFromScancontext(load_sc);
            std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);

            scManager.polarcontexts_.push_back(load_sc);
            scManager.polarcontext_invkeys_.push_back(ringkey);
            scManager.polarcontext_vkeys_.push_back(sectorkey);
            scManager.polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);
        }
        const float kSCFilterSize = 0.5;
        downSizeFilterSC.setLeafSize(kSCFilterSize, kSCFilterSize, kSCFilterSize);

        loadPosesDirectory = std::getenv("HOME") + savePCDDirectory + "singlesession_posegraph.txt";
        loadPoses(loadPosesDirectory, matrix_poses);
        // std::cout << "marix_poses: \n" << matrix_poses << endl;

        allocateMemory();
    }
    /**
     * @brief 申请内存
     */
    void allocateMemory(){
        laserCloudRaw.reset(new pcl::PointCloud<PointType>());
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>());

        cloudGlobalMap.reset(new pcl::PointCloud<PointType>());
        cloudGlobalMapDS.reset(new pcl::PointCloud<PointType>());
        cloudScanForInitialize.reset(new pcl::PointCloud<PointType>()); 

        resetLIO();

        for(int i=0; i<6; ++i){
            transformInTheWorld[i]=0; 
        }
        for(int i=0; i<6; ++i){
            transformOdomToWorld[i]=0; 
        }
        initializedFlag = NonInitialized;
        cloudGlobalLoad(); 
    }
    /**
     * @brief 这是LIO-SAM原本一些变量、指针的初始化
     */
    void resetLIO()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        //kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        //kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

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

        latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
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
     * @brief 在某一帧前后搜索一定数量的点云帧
     */
    void loopFindNearKeyframesWithRespectTo(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
        const int &key, const int &searchNum, const int _wrt_key){

        nearKeyframes->clear();
        int cloudSize = scManager.polarcontexts_.size(); // 点云帧和SC帧数量相同
        for(int i=-searchNum; i<=searchNum; ++i){
            int keyNear = key+i;
            if(keyNear<0 || keyNear>=cloudSize){
                continue; 
            }

            pcl::PointCloud<PointType>::Ptr scan_temp(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile(loadNodePCDDirectory + padZeros(keyNear) + ".pcd", *scan_temp);
            Eigen::MatrixXd pose_temp = matrix_poses.row(keyNear); // 获取关键帧位姿
            Eigen::Quaterniond q_temp(pose_temp(7), pose_temp(4), pose_temp(5), pose_temp(6));
            Eigen::Vector3d euler_angles = q_temp.toRotationMatrix().eulerAngles(2, 1, 0);

            PointTypePose pointTypePose_temp;
            pointTypePose_temp.x = pose_temp(1);
            pointTypePose_temp.y = pose_temp(2);
            pointTypePose_temp.z = pose_temp(3);
            pointTypePose_temp.roll = euler_angles[2];
            pointTypePose_temp.pitch = euler_angles[1];
            pointTypePose_temp.yaw = euler_angles[0];

            *nearKeyframes += *transformPointCloud(scan_temp, &pointTypePose_temp);
        } 

        if(nearKeyframes->empty()){
            return;
        }

        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    /**
     * @brief 利用SC进行重定位
     */
    void performSCLoopClosure(){
        auto detectResult = scManager.detectLoopClosureID();
        int loopKeyCur = scManager.polarcontexts_.size() - 1;
        int loopKeyPre = detectResult.first;
        float yawDiffRad = detectResult.second;
        if(loopKeyPre == -1){  // SC没有匹配成功
            RCLCPP_INFO_STREAM(get_logger(), "SC loop not found");
            return; 
        }
        RCLCPP_INFO_STREAM(get_logger(), "SC loop found between " << loopKeyCur << " and " << loopKeyPre);

        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

        int base_key = 0;
        Eigen::MatrixXd pose_temp = matrix_poses.row(loopKeyPre); // 从位姿图中获取关键帧位姿
        Eigen::Quaterniond q_temp(pose_temp(7), pose_temp(4), pose_temp(5), pose_temp(6));
        Eigen::Vector3d euler_angles = q_temp.toRotationMatrix().eulerAngles(2, 1, 0);

        PointTypePose pointTypePose_temp;
        pointTypePose_temp.x = pose_temp(1);
        pointTypePose_temp.y = pose_temp(2);
        pointTypePose_temp.z = pose_temp(3);
        pointTypePose_temp.roll = euler_angles[2];
        pointTypePose_temp.pitch = euler_angles[1];
        pointTypePose_temp.yaw = euler_angles[0];
        *cureKeyframeCloud += *transformPointCloud(laserCloudRawDS, &pointTypePose_temp); // 将点云转换到匹配关键帧下

        loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key);

        if(cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000){
            return;
        }

        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150); // use a value can cover 2*historyKeyframeSearchNum range in meter 
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr result_icp(new pcl::PointCloud<PointType>());
        icp.align(*result_icp);
        // TODO 此处即使SC匹配到了，但getFintessScore匹配得分依旧很大，不满足阈值
        if(icp.hasConverged()==false || icp.getFitnessScore()>historyKeyframeFitnessScore){
            RCLCPP_INFO_STREAM(get_logger(), "ICP fitness test failed " << icp.getFitnessScore() <<
                ">" << historyKeyframeFitnessScore << ". Reject");
            return; 
        }
        else{
            RCLCPP_INFO_STREAM(get_logger(), "ICP fitness test passed " << icp.getFitnessScore() <<
                "<" << historyKeyframeFitnessScore << ". Add in"); 
        }

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        Eigen::Affine3f loopKeyPreTransformInTheWorld = pclPointToAffine3f(pointTypePose_temp); // 匹配到的关键帧到世界坐标系转换关系
        Eigen::Affine3f loopKeyCurTransformInTheWorld = loopKeyPreTransformInTheWorld * correctionLidarFrame; // 当前帧到世界坐标系的转换关系

        pcl::getTranslationAndEulerAngles(loopKeyCurTransformInTheWorld, x, y, z, roll, pitch, yaw);
        transformInTheWorld[0] = roll;
        transformInTheWorld[1] = pitch;
        transformInTheWorld[2] = yaw;
        transformInTheWorld[3] = x; 
        transformInTheWorld[4] = y;
        transformInTheWorld[5] = z;

        RCLCPP_DEBUG_STREAM(get_logger(), "The pose loop is x:" << x << " y:" << y
            << " z:" << z);
    }

    /**
     * @brief 寻找初始位姿
     */
    void SystemInit(){
        if(initializedFlag == NonInitialized || initializedFlag == Initializing){
            if(cloudScanForInitialize->points.size() == 0){  // TODO 此处待改进，重定位初始化只用到了第一帧
                downsampleCurrentScan();
                mtx_general.lock();
                *cloudScanForInitialize += *laserCloudCornerLastDS;
                *cloudScanForInitialize += *laserCloudSurfLastDS;
                mtx_general.unlock();

                laserCloudCornerLastDS->clear();
		        laserCloudSurfLastDS->clear();
		        laserCloudCornerLastDSNum = 0;
		        laserCloudSurfLastDSNum = 0;

                transformTobeMapped[0] = cloudInfo.imu_roll_init;
                transformTobeMapped[1] = cloudInfo.imu_pitch_init;
                transformTobeMapped[2] = cloudInfo.imu_yaw_init;
                if (!useImuHeadingInitialization){
                    transformTobeMapped[2] = 0;
                }

                pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
                pcl::copyPointCloud(*laserCloudRawDS, *thisRawCloudKeyFrame);
                scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);

                performSCLoopClosure(); // 利用SC进行重定位(借鉴于SC回环检测，故函数名未变) 
            } 
            return; 
        }
        frameNum++; 
    }

    /**
     * @brief 特征点云的回调函数
     */
    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn){
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudSurfLast);
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw);
        laserCloudRawTime = stamp2Sec(cloudInfo.header.stamp);

        SystemInit();
        if(initializedFlag != Initialized){ // 没有初始化完成
            return; 
        }

        std::lock_guard<std::mutex> lock(mtx);
        if(timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval){
            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();  // 当前帧位姿初始化

            extractSurroundingKeyFrames(); // 构建局部地图

            downsampleCurrentScan();

            scan2MapOptimization(); // 将当前帧点云匹配到局部地图，优化位姿

            saveKeyFramesAndFactor(); // 执行图优化

            publishOdometry();

            publishFrames();
        
        }


    }

    void updateInitialGuess(){
        // 把上一次的激光里程计结果缓存到increamentalOdometryAffineFront中  
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
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

    void extractSurroundingKeyFrames(){
        if(cloudKeyPoses3D->points.empty() == true){
            return; 
        }

        extractForLoopClosure(); 
    }

    void extractForLoopClosure(){
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = win_cloudKeyPoses3D.size();
        for(int i=numPoses-1; i>=0; --i){
            cloudToExtract->push_back(win_cloudKeyPoses3D[i]); 
        }
        extractCloud(cloudToExtract); 
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract){
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

        #pragma omp parallel for num_threads(numberOfCores) // 不一定有用
        for(int i=0; i<(int)cloudToExtract->size(); ++i){
            PointTypePose thisPose6D;
            thisPose6D = win_cloudKeyPoses6D[i];
            laserCloudCornerSurroundingVec[i] = *transformPointCloud(win_cornerCloudKeyFrames[i], &thisPose6D); 
            laserCloudSurfSurroundingVec[i] = *transformPointCloud(win_surfCloudKeyFrames[i], &thisPose6D);
        }

        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for(int i=0; i<(int)cloudToExtract->size(); ++i){
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap += laserCloudSurfSurroundingVec[i]; 
        }

        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
    }

    void scan2MapOptimization(){
        if(cloudKeyPoses3D->points.empty()){
            return; 
        }

        if(laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum){
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);            

            for(int iterCount=0; iterCount < 30; iterCount++){
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

    void cornerOptimization(){
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for(int i=0; i<laserCloudCornerLastDSNum; i++){
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            // caclulate pointOri location in the map using the prediction pose
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0)); // the covariance matrix of search points
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0)); // eigenvalues of the covariance matrix
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0)); // eigenvectors of the covariance matrix

            if(pointSearchSqDis[4] < 1.0){ // this distance of all the points found is less than 1.0m
                float cx=0, cy=0, cz=0;
                for(int j=0; j<5; j++){
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z; 
                }
                cx /= 5; cy /= 5; cz /= 5;
                // calculate the convarince matrix
                float a11=0, a12=0, a13=0, a22=0, a23=0, a33=0;
                for(int j=0; j<5; j++){
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az; 
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1); // decomposition

                if(matD1.at<float>(0,0) > 3*matD1.at<float>(0,1)){ // the largest is much greater than second
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // select two points on the line
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                        + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                        + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                 - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                 + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization(){
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

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

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

        bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if(laserCloudSelNum < 50){
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for(int i = 0; i < laserCloudSelNum; i++){

            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;

            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;

            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                        + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                        + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                         + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                        + ((-cry*crz - srx*sry*srz)*pointOri.x
                           + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                        + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                        + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                pow(matX.at<float>(3, 0) * 100, 2) +
                pow(matX.at<float>(4, 0) * 100, 2) +
                pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

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

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);  // not use
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void saveKeyFramesAndFactor(){
        if(saveFrame() == false){
            return; 
        }

        addOdomFactor();

        // TODO 是否要加入GPS因子和回环(重定位回环)因子

        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // use as index
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

        mtxWin.lock();
        win_cloudKeyPoses3D.push_back(thisPose3D);
        win_cloudKeyPoses6D.push_back(thisPose6D);
        if(win_cloudKeyPoses3D.size() > winSize){
            win_cloudKeyPoses3D.pop_front();
            win_cloudKeyPoses6D.pop_front(); 
        }

        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        win_cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        win_surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        if(win_cornerCloudKeyFrames.size() > winSize){
            win_cornerCloudKeyFrames.pop_front();
            win_surfCloudKeyFrames.pop_front(); 
        }
        mtxWin.unlock();

        updatePath(thisPose6D);

    }

    void updatePath(const PointTypePose& pose_in){
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

    void addOdomFactor(){
        if(cloudKeyPoses3D->points.empty()){
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished());
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtsam::Pose3 relPose = poseFrom.between(poseTo);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), relPose, odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold){
            return false;
        }

        return true;
    }

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
        laserOdometryROS.pose.covariance[0] = double(imuPreintegrationResetId);
        pubOdomAftMappedROS->publish(laserOdometryROS);
        
        // Publish odometry for ROS (incremental)
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

    void publishFrames(){
        if(cloudKeyPoses3D->points.empty()){
            return; 
        }

        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        if(pubLaserCloudInWorld->get_subscription_count() != 0){
            pcl::PointCloud<PointType>::Ptr cloudInBase(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr cloudOutInWorld(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6DInOdom = trans2PointTypePose(transformTobeMapped);
            Eigen::Affine3f T_thisPose6DInOdom = pclPointToAffine3f(thisPose6DInOdom);
            mtxtranformOdomToWorld.lock();
            PointTypePose pose_Odom_Map = trans2PointTypePose(transformOdomToWorld);
            mtxtranformOdomToWorld.unlock();
            Eigen::Affine3f T_pose_Odom_Map = pclPointToAffine3f(pose_Odom_Map);

            Eigen::Affine3f T_poseInMap = T_pose_Odom_Map * T_thisPose6DInOdom;
            *cloudInBase += *laserCloudCornerLastDS;
            *cloudInBase += *laserCloudSurfLastDS;
            pcl::transformPointCloud(*cloudInBase, *cloudOutInWorld, T_poseInMap.matrix());
            publishCloud(pubLaserCloudInWorld, cloudOutInWorld, timeLaserInfoStamp, "map"); 
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

    /**
     * @brief gps的回调函数
     */
    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg){
        gpsQueue.push_back(*gpsMsg); 
    }

    /**
     * @brief 给定初始位姿的回调函数(从rviz中传入的初始位姿)
     */
    void initialpose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr){
    
    }
    /**
     * @brief 加载点云文件
     */
    void cloudGlobalLoad(){
        pcl::io::loadPCDFile(std::getenv("HOME")+savePCDDirectory+"cloudGlobal.pcd", *cloudGlobalMap);

        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(cloudGlobalMap);
        downSizeFilterICP.filter(*cloud_temp);
        *cloudGlobalMapDS = *cloud_temp;

        RCLCPP_INFO(get_logger(), "test 0.01 the size of global cloud: %i", cloudGlobalMap->points.size());
        RCLCPP_INFO(get_logger(), "test 0.02 the size of global cloud: %i", cloudGlobalMapDS->points.size());
        // std::cout << "test 0.01  the size of global cloud: " << cloudGlobalMap->points.size() << std::endl;
        // std::cout << "test 0.02  the size of global map after filter: " << cloudGlobalMapDS->points.size() << std::endl;
    }

    /**
     * @brief 重定位初始化
     */
    void ICPLocalizeInitialize(){
        pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());
        mtx_general.lock();
        *laserCloudIn += *cloudScanForInitialize;
        mtx_general.unlock();

        if(laserCloudIn->points.size() == 0){
            return; 
        }
        if(transformInTheWorld[0] == 0 && transformInTheWorld[1] == 0 && transformInTheWorld[2] == 0 
            && transformInTheWorld[3] == 0 && transformInTheWorld[4] == 0 && transformInTheWorld[5] == 0){
            return;     
        }
        RCLCPP_DEBUG(get_logger(), "the size of lasercloud for LocalizeInitialize is %i", laserCloudIn->points.size());    

        pcl::NormalDistributionsTransform<PointType, PointType> ndt; 
        ndt.setTransformationEpsilon(0.01);
        ndt.setResolution(1.0);    // ndt分辨率

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        ndt.setInputSource(laserCloudIn);
        ndt.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr result_ndt(new pcl::PointCloud<PointType>());
        // 初始化匹配到的匹配帧到世界坐标系的位姿
        PointTypePose thisPose6DInWorld = trans2PointTypePose(transformInTheWorld);
        Eigen::Affine3f T_thisPose6DInWorld = pclPointToAffine3f(thisPose6DInWorld);
        ndt.align(*result_ndt, T_thisPose6DInWorld.matrix());

        icp.setInputSource(laserCloudIn);
        icp.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr result_icp(new pcl::PointCloud<PointType>());
        icp.align(*result_icp, ndt.getFinalTransformation());

        RCLCPP_INFO_STREAM(get_logger(), "the pose before initializing is, x:" << 
            transformInTheWorld[3] <<" y:" << transformInTheWorld[4] << " z:" <<
            transformInTheWorld[5]);
        RCLCPP_INFO_STREAM(get_logger(), "the pose in odom before initializing is, x:" <<
            transformOdomToWorld[3] << " y:" << transformOdomToWorld[4] << " z:" <<
            transformInTheWorld[5]);
        RCLCPP_INFO_STREAM(get_logger(), "icp score in initializing process is: " <<
            icp.getFitnessScore());
        RCLCPP_INFO_STREAM(get_logger(), "pose after initializing process is: " <<
            icp.getFinalTransformation());

        PointTypePose thisPose6DInOdom = trans2PointTypePose(transformTobeMapped);

        RCLCPP_INFO_STREAM(get_logger(), "transformTobeMapped X_Y_Z: " << transformTobeMapped[3] << " " <<
            transformTobeMapped[4] << " " << transformTobeMapped[5]); 

        Eigen::Affine3f T_thisPose6DInOdom = pclPointToAffine3f(thisPose6DInOdom);

        Eigen::Affine3f T_thisPose6DInMap;
        T_thisPose6DInMap = icp.getFinalTransformation();
        float x_g, y_g, z_g, R_g, P_g, Y_g;
        pcl::getTranslationAndEulerAngles (T_thisPose6DInMap, x_g, y_g, z_g, R_g, P_g, Y_g);
        transformInTheWorld[0] = R_g;
        transformInTheWorld[1] = P_g;
        transformInTheWorld[2] = Y_g;
        transformInTheWorld[3] = x_g;
        transformInTheWorld[4] = y_g;
        transformInTheWorld[5] = z_g;

        Eigen::Affine3f transOdomToMap = T_thisPose6DInMap * T_thisPose6DInOdom.inverse();
        float deltax, deltay, deltaz, deltaR, deltaP, deltaY;
        pcl::getTranslationAndEulerAngles(transOdomToMap, deltax, deltay, deltaz, deltaR, deltaP, deltaY);

        mtxtranformOdomToWorld.lock();
        transformOdomToWorld[0] = deltaR;
        transformOdomToWorld[1] = deltaP;
        transformOdomToWorld[2] = deltaY;
        transformOdomToWorld[3] = deltax;
        transformOdomToWorld[4] = deltay;
        transformOdomToWorld[5] = deltaz;
        mtxtranformOdomToWorld.unlock();
        RCLCPP_INFO_STREAM(get_logger(), "The pose of odom relative to Map x:" << transformOdomToWorld[3]
            <<" y:" << transformOdomToWorld[4] << " z:" << transformOdomToWorld[5]);
        publishCloud(pubLaserCloudInWorld, result_icp, timeLaserInfoStamp, "map");
        publishCloud(pubMapWorld, cloudGlobalMapDS, timeLaserInfoStamp, "map");

        if(icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore){
            initializedFlag = Initializing;
            RCLCPP_INFO_STREAM(get_logger(), "Initializing Fail " << icp.hasConverged());
            return; 
        }
        else{
            initializedFlag = Initialized;
            RCLCPP_INFO_STREAM(get_logger(), "Initializing Success");
/*            geometry_msgs::msg::PoseStamped pose_odomToMap;
            tf2::Quaternion q_odomToMap;
            q_odomToMap.setRPY(deltaR, deltaP, deltaY);

            pose_odomToMap.header.stamp = timeLaserInfoStamp;
            pose_odomToMap.header.frame_id = "map";
            pose_odomToMap.pose.position.x = deltax;
            pose_odomToMap.pose.position.y = deltay;
            pose_odomToMap.pose.position.z = deltaz;
            pose_odomToMap.pose.orientation.x = q_odomToMap.x();
            pose_odomToMap.pose.orientation.y = q_odomToMap.y();
            pose_odomToMap.pose.orientation.z = q_odomToMap.z();
            pose_odomToMap.pose.orientation.w = q_odomToMap.w();
            pubOdomToMapPose->publish(pose_odomToMap);
        */
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(deltaR, deltaP, deltaY);
            tf2::Transform t_odom_to_map = tf2::Transform(quat_tf, tf2::Vector3(deltax, deltay, deltaz));
            tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
            tf2::Stamped<tf2::Transform> temp_odom_to_map(t_odom_to_map, time_point, "map");
            geometry_msgs::msg::TransformStamped trans_odom_to_map;
            tf2::convert(temp_odom_to_map, trans_odom_to_map);
            trans_odom_to_map.child_frame_id = odometryFrame;
            br->sendTransform(trans_odom_to_map);
        }

    }

    void ICPscanMatchGlobal(){
        if(cloudKeyPoses3D->points.empty() == true){
            return; 
        }

        mtxWin.lock();
        int latestFrameIDGlobalLocalize; 
        latestFrameIDGlobalLocalize = win_cloudKeyPoses3D.size()-1;

        pcl::PointCloud<PointType>::Ptr latestCloudIn(new pcl::PointCloud<PointType>());
        *latestCloudIn += *transformPointCloud(win_cornerCloudKeyFrames[latestFrameIDGlobalLocalize],
            &win_cloudKeyPoses6D[latestFrameIDGlobalLocalize]);
        *latestCloudIn += *transformPointCloud(win_surfCloudKeyFrames[latestFrameIDGlobalLocalize],
            &win_cloudKeyPoses6D[latestFrameIDGlobalLocalize]);
        
        RCLCPP_INFO_STREAM(get_logger(), "the size of input cloud: " << latestCloudIn->points.size());
        mtxWin.unlock();

        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.01);
        ndt.setResolution(1.0);

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        mtxtranformOdomToWorld.lock();
        Eigen::Affine3f transodomToWorld_init = pcl::getTransformation(transformOdomToWorld[3], transformOdomToWorld[4],transformOdomToWorld[5],transformOdomToWorld[0],transformOdomToWorld[1],transformOdomToWorld[2]);
        mtxtranformOdomToWorld.unlock();

        Eigen::Matrix4f matrixInitGuess = transodomToWorld_init.matrix();

        ndt.setInputSource(latestCloudIn);
        ndt.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr result_ndt(new pcl::PointCloud<PointType>());
        ndt.align(*result_ndt, matrixInitGuess);
        
        icp.setInputSource(latestCloudIn);
        icp.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr result_icp(new pcl::PointCloud<PointType>());
        icp.align(*result_icp, ndt.getFinalTransformation());

        RCLCPP_INFO_STREAM(get_logger(), "ICP converg flag: " << icp.hasConverged() <<
            ". Fitness score: " << icp.getFitnessScore());

        Eigen::Affine3f transodomToWorld_New;
        transodomToWorld_New = icp.getFinalTransformation();
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transodomToWorld_New, x, y, z, roll, pitch, yaw);

        mtxtranformOdomToWorld.lock();
        transformOdomToWorld[0] = roll;
        transformOdomToWorld[1] = pitch;
        transformOdomToWorld[2] = yaw;
        transformOdomToWorld[3] = x;
        transformOdomToWorld[4] = y;
        transformOdomToWorld[5] = z;
        mtxtranformOdomToWorld.unlock();

        publishCloud(pubMapWorld, cloudGlobalMapDS, timeLaserInfoStamp, "map");

        if(icp.hasConverged() == true && icp.getFitnessScore() < historyKeyframeFitnessScore){
        /*    geometry_msgs::msg::PoseStamped pose_odomTo_map;

            tf2::Quaternion q_odomTo_map;
            q_odomTo_map.setRPY(roll, pitch, yaw);

            pose_odomTo_map.header.stamp = timeLaserInfoStamp;
            pose_odomTo_map.header.frame_id = "map";
            pose_odomTo_map.pose.position.x = x;
            pose_odomTo_map.pose.position.y = y;
            pose_odomTo_map.pose.position.z = z;
            pose_odomTo_map.pose.orientation.x = q_odomTo_map.x();
            pose_odomTo_map.pose.orientation.y = q_odomTo_map.y();
            pose_odomTo_map.pose.orientation.z = q_odomTo_map.z();
            pose_odomTo_map.pose.orientation.w = q_odomTo_map.w();

            pubOdomToMapPose->publish(pose_odomTo_map); 
            */ 
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(roll, pitch, yaw);
            tf2::Transform t_odom_to_map = tf2::Transform(quat_tf, tf2::Vector3(x, y, z));
            tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
            tf2::Stamped<tf2::Transform> temp_odom_to_map(t_odom_to_map, time_point, "map");
            geometry_msgs::msg::TransformStamped trans_odom_to_map;
            tf2::convert(temp_odom_to_map, trans_odom_to_map);
            trans_odom_to_map.child_frame_id = odometryFrame;
            br->sendTransform(trans_odom_to_map);

        }

    }


    /**
     * @brief 全局重定位线程
     */
    void globalLocalizeThread(){
        while(rclcpp::ok()){
            if(initializedFlag == NonInitialized){     // 重定位还未初始化成功
                RCLCPP_DEBUG(get_logger(), "Attempt to relocalize!");
                ICPLocalizeInitialize();
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if(initializedFlag == Initializing){
                RCLCPP_INFO(get_logger(), "Offer a new Guess Please");
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else{
                rclcpp::sleep_for(std::chrono::seconds(5));
                // RCLCPP_INFO(get_logger(), "Loacliztion");
                ICPscanMatchGlobal(); 
            }      
        } 
    }


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

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }
    
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }
};




int main(int argc, char** argv){

    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Global  Localization Started.\033[0m");

    std::thread loaclizeInWorldThreadthread(&mapOptimization::globalLocalizeThread, MO);

    exec.spin();

    rclcpp::shutdown();

    loaclizeInWorldThreadthread.join();

    return 0;
}