#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"
#include "lio_sam/srv/save_map.hpp"
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
#include <filesystem>
#include <pcl/registration/ndt.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>

#include "lio_sam/Scancontext.hpp"
using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance; 
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdometryMap;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudInWorld;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMapWorld;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;
    rclcpp::Service<lio_sam::srv::SaveMap>::SharedPtr srvSaveMap;
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subLoop;

    rclcpp::TimerBase::SharedPtr timer;
    std::deque<nav_msgs::msg::Odometry> gpsQueue; // GPS消息队列
    lio_sam::msg::CloudInfo cloudInfo; // 当前点云信息

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
 
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; // 所有关键帧的三维位姿(x,y,z)
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 所有关键帧的六维位姿(x,y,z,r,p,y)
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D; // copy_* 是在回环中使用，为了不干扰主线程
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization
    
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;
    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    rclcpp::Time timeLaserInfoStamp;
    double timeLaserInfoCur;
    float transformTobeMapped[6];
    float copy_transformTobeMapped[6]; 

    std::mutex mtx;  // 点云信息回调函数锁
    std::mutex mtxLoopInfo; // 回环检测线程锁

    bool isDegenerate = false; // 此标志位标识点云匹配结果是否较差，true表示差，会在imuPreintegration中根据此选择不同的噪声模型给因子图
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer;
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

    nav_msgs::msg::Path globalPath;
    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;
    // TF发布器
    std::mutex mtx_general;
    std::mutex mtxtranformOdomToWorld;
    std::unique_ptr<tf2_ros::TransformBroadcaster> br;
    // geometry_msgs::msg::TransformStamped trans_odom_to_map;
    std::mutex mtx_trans;
    // std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    // std::shared_ptr<tf2_ros::TransformListener> tf_listener;
    pcl::PointCloud<PointType>::Ptr cloudGlobalMap; //  全局地图
    pcl::PointCloud<PointType>::Ptr cloudGlobalMapDS; // 降采样后的全局地图
    pcl::PointCloud<PointType>::Ptr cloudScanForInitialize;
    pcl::PointCloud<PointType>::Ptr laserCloudRaw;
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS;
    Eigen::MatrixXd matrix_poses;
    float transformInTheWorld[6];
    float transformOdomToWorld[6];

    enum InitializedFlag{
        NonInitialized,
        Initializing,
        Initialized 
    };
    InitializedFlag initializedFlag;

    std::string loadSCDDirectory;
    std::string loadNodePCDDirectory;
    std::string loadPosesDirectory;
    SCManager scManager;

    pcl::VoxelGrid<PointType> downSizeFilterSC;

    mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options)
    {
        ISAM2Params parameters;  // isam 优化器参数
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry", qos);
        pubOdometryMap = create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/mapodom", qos);
        pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos);
        pubPath = create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);
        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        // timer = create_wall_timer(5ms, std::bind(&mapOptimization::trans_callback, this));
        // tf_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        // tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        subCloud = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos,
            std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        subGPS = create_subscription<nav_msgs::msg::Odometry>(
            gpsTopic, 200,
            std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));
        subLoop = create_subscription<std_msgs::msg::Float64MultiArray>(
            "lio_loop/loop_closure_detection", qos,
            std::bind(&mapOptimization::loopInfoHandler, this, std::placeholders::_1));
        /* 
        auto saveMapService = [this](const std::shared_ptr<rmw_request_id_t> request_header, const std::shared_ptr<lio_sam::srv::SaveMap::Request> req, std::shared_ptr<lio_sam::srv::SaveMap::Response> res) -> void {
            (void)request_header;
            string saveMapDirectory;
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files ..." << endl;
            if(req->destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
            else saveMapDirectory = std::getenv("HOME") + req->destination;
            cout << "Save destination: " << saveMapDirectory << endl;
            // create directory and remove old files;
            int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
            unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
            // save key frame transformations
            pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
            // extract global point cloud map
            pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) 
            {
                *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
                *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
                cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
            }
            if(req->resolution != 0)
            {
               cout << "\n\nSave resolution: " << req->resolution << endl;
               // down-sample and save corner cloud
               downSizeFilterCorner.setInputCloud(globalCornerCloud);
               downSizeFilterCorner.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterCorner.filter(*globalCornerCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
               // down-sample and save surf cloud
               downSizeFilterSurf.setInputCloud(globalSurfCloud);
               downSizeFilterSurf.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterSurf.filter(*globalSurfCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
            }
            else
            {
            // save corner cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
               // save surf cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
            }
            // save global point cloud map
            *globalMapCloud += *globalCornerCloud;
            *globalMapCloud += *globalSurfCloud;
            int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
            res->success = ret == 0;
            downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
            downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files completed\n" << endl;
            return;
        };
        // 地图保存服务 
        srvSaveMap = create_service<lio_sam::srv::SaveMap>("lio_sam/save_map", saveMapService);
        */
        pubMapWorld = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_map_map", 1);
        pubLaserCloudInWorld = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/lasercloud_in_world", 1);
        pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubLoopConstraintEdge = create_publisher<visualization_msgs::msg::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

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

    void allocateMemory()
    {   
        laserCloudRaw.reset(new pcl::PointCloud<PointType>());
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>());
        cloudGlobalMap.reset(new pcl::PointCloud<PointType>());
        cloudGlobalMapDS.reset(new pcl::PointCloud<PointType>());
        cloudScanForInitialize.reset(new pcl::PointCloud<PointType>());
        initializedFlag = NonInitialized;

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

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
            copy_transformTobeMapped[i] = 0;
            transformInTheWorld[i] = 0;
            transformOdomToWorld[i] = 0;
        }
        matP.setZero();

        cloudGlobalLoad();
    }
    
    void cloudGlobalLoad(){
        pcl::io::loadPCDFile(std::getenv("HOME")+savePCDDirectory+"cloudGlobal.pcd", *cloudGlobalMap);

        // pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        // downSizeFilterICP.setInputCloud(cloudGlobalMap);
        // downSizeFilterICP.filter(*cloud_temp);
        *cloudGlobalMapDS = *cloudGlobalMap;

        RCLCPP_INFO(get_logger(), "test 0.01 the size of global cloud: %i", cloudGlobalMap->points.size());
        RCLCPP_INFO(get_logger(), "test 0.02 the size of global cloud: %i", cloudGlobalMapDS->points.size());
    }

    void SystemInit(){
        if(initializedFlag == NonInitialized || initializedFlag == Initializing){
            if(cloudScanForInitialize->points.size() == 0){
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
                if(!useImuHeadingInitialization){
                    transformTobeMapped[2] = 0; 
                }
                pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
                pcl::copyPointCloud(*laserCloudRawDS, *thisRawCloudKeyFrame);
                scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);

                performSCLoopClosure();
                 
            } 
        } 
    }


    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw);

        SystemInit();
        if(initializedFlag != Initialized){
            return; 
        }
        std::lock_guard<std::mutex> lock(mtx);
        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) // 两帧之间时间间隔大于mappingProcessInterval才会进行处理
        {
            timeLastProcessing = timeLaserInfoCur;
            // 当前帧位姿初始化
            updateInitialGuess();
            // 构建局部地图
            extractSurroundingKeyFrames();
            // 对当前帧点云做降采样
            downsampleCurrentScan();
            // 将当前帧点云匹配到构建的局部地图，优化当前位姿
            scan2MapOptimization();
            // 计算是否将当前帧采纳为关键帧，加入因子图优化
            saveKeyFramesAndFactor();
            // 当新的回环因子或者GPS因子加入因子图时，对历史帧执行位姿更新
            correctPoses();
            // 发布激光里程计
            publishOdometry();
            // 发布当前帧对齐到地图坐标系的点云和完整轨迹
            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }
    /**
     * @brief 这里又来一个？ 还是代码复制的时候弄错了？
     */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,  Eigen::Affine3f transCur)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
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

    void visualizeGlobalMapThread()
    {
        rclcpp::Rate rate(0.2);
        while (rclcpp::ok()){
            rate.sleep();
            publishGlobalMap();
        }
        /*
        if (savePCD == false)
            return;
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
        */
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround->get_subscription_count() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
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

    void globalLocalizeThread(){
        while(rclcpp::ok()){
            if(initializedFlag == NonInitialized){
                RCLCPP_DEBUG(get_logger(), "Attempt to relocalize!");
                ICPLocalizeInitialize(); 
            }
            else if(initializedFlag == Initializing){
                RCLCPP_INFO(get_logger(), "Offer a new Guess Please");
                rclcpp::sleep_for(std::chrono::seconds(1)); 
            }
            else{
                rclcpp::sleep_for(std::chrono::seconds(1));
                RCLCPP_INFO(get_logger(), "Localization"); 
                // ICPscanMatchGlobal();
                publishCloud(pubMapWorld, cloudGlobalMapDS, timeLaserInfoStamp, "map");
            }  
        } 
    }

    void ICPscanMatchGlobal(){
        if(cloudKeyPoses3D->points.empty() == true){
            return; 
        }

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        for(int i=0; i<6; ++i){
            copy_transformTobeMapped[i] = transformTobeMapped[i]; 
        }
        int latestFrameID = copy_cloudKeyPoses3D->size()-1; 
        pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());
        *laserCloudIn += *transformPointCloud(cornerCloudKeyFrames[latestFrameID],
            &copy_cloudKeyPoses6D->points[latestFrameID]);
        *laserCloudIn += *transformPointCloud(surfCloudKeyFrames[latestFrameID],
            &copy_cloudKeyPoses6D->points[latestFrameID]);
        mtx.unlock();
        RCLCPP_INFO_STREAM(get_logger(), "the size of input cloud: " << laserCloudIn->points.size());

        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.01);
        ndt.setResolution(1.0);

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        Eigen::Affine3f transToOdom_temp = pcl::getTransformation(copy_transformTobeMapped[3], copy_transformTobeMapped[4], copy_transformTobeMapped[5], copy_transformTobeMapped[0], copy_transformTobeMapped[1], copy_transformTobeMapped[2]);
        Eigen::Matrix4f matrixInitGuess = transToOdom_temp.matrix();

        ndt.setInputSource(laserCloudIn);
        ndt.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr result_ndt(new pcl::PointCloud<PointType>());
        ndt.align(*result_ndt, matrixInitGuess);

        icp.setInputSource(laserCloudIn);
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
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(transformOdomToWorld[0],  transformOdomToWorld[1],  transformOdomToWorld[2]);
            tf2::Transform t_odom_to_map = tf2::Transform(quat_tf, tf2::Vector3(transformOdomToWorld[3],  transformOdomToWorld[4],  transformOdomToWorld[5]));
            tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
            tf2::Stamped<tf2::Transform> temp_odom_to_map(t_odom_to_map, time_point, "map");
            geometry_msgs::msg::TransformStamped trans_odom_to_map;
            tf2::convert(temp_odom_to_map, trans_odom_to_map);
            trans_odom_to_map.child_frame_id = odometryFrame;
            br->sendTransform(trans_odom_to_map); 
        }
    }


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


    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        rclcpp::Rate rate(loopClosureFrequency);
        while (rclcpp::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::msg::Float64MultiArray::SharedPtr loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performSCLoopClosure(){
        auto detectResult = scManager.detectLoopClosureID();
        int loopKeyCur = scManager.polarcontexts_.size() - 1;
        int loopKeyPre = detectResult.first;
        float yawDiffRad = detectResult.second;
        if(loopKeyPre == -1){
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

    void performLoopClosure()
    {   // 关键帧队列为空直接返回
        if (cloudKeyPoses3D->points.empty() == true)
            return;
        // 加锁拷贝，防止多线程干扰
        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); // 将当前帧转换到map坐标系并降采样。参数0表示不加上前后其它帧。
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); // 对匹配前后几帧转换到map坐标系，融合并降采样，构建局部地图
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames->get_subscription_count() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
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

        // Add pose constraint
        // 将回环索引、回环间相对位姿、回环噪声模型加入全局变量
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); 
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }
        // 如果没有找到位置关系，时间关系都符合要求的关键帧，则返回false
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframesWithRespectTo(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
        const int &key, const int &searchNum, const int _wrt_key){

        nearKeyframes->clear();
        int cloudSize = scManager.polarcontexts_.size();
        for(int i=-searchNum; i<=searchNum; ++i){
            int keyNear = key+i;
            if(keyNear<0 || keyNear>=cloudSize){
                continue; 
            }

            pcl::PointCloud<PointType>::Ptr scan_temp(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile(loadNodePCDDirectory + padZeros(keyNear) + ".pcd", *scan_temp);
            Eigen::MatrixXd pose_temp = matrix_poses.row(keyNear);
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

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

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
        markerEdge.scale.x = 0.1;
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







    


    void updateInitialGuess()
    {
        // save current transformation before any processing
        // 把上一次激光里程计的结果缓存到incrementalOdometryAffineFront中
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
        // lastImuTransformation用于缓存上一帧IMU输出的roll、pitch、yaw。
        static Eigen::Affine3f lastImuTransformation;
        // initialization
        // 第一帧做初始化，直接使用imu输出的RPY作为当前帧的初始位姿，如果不是9轴imu（即rpy
        // 不和地磁角绑定），则yaw角直接初始化为0。
        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = cloudInfo.imu_roll_init;
            transformTobeMapped[1] = cloudInfo.imu_pitch_init;
            transformTobeMapped[2] = cloudInfo.imu_yaw_init;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        // 后续帧的帧位姿初始化。
        // 如果IMU里程计可用（已经融合了激光里程计结果），使用其作为6D位姿初始化
        // 如果IMU里程计不可用当IMU原始角度可用，则在上一帧位姿上叠加IMU角度变化作为初始化
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odom_available == true) // 如果IMU里程计可用
        {
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
        // 使用IMU输出的角度作为初始化，只采用角度。
        // 如果IMU原始角度和IMU里程计不可用的情况下，transformTobeMapped没有被改变，也就是使用上一帧结果作为当前帧初始值。
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

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan()
    {   
        laserCloudRawDS->clear();
        downSizeFilterSC.setInputCloud(laserCloudRaw);
        downSizeFilterSC.filter(*laserCloudRawDS);

        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap(); // 方便后面用旋转平移关系对选中的点转换到地图坐标系

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {   // pointOri是雷达坐标系下的边缘点，pointSel是转换到地图坐标系下的点，coeff存储经过距离加权后的点到平面向量
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); // 将雷达坐标系下的pointOri转换到地图坐标系PointSel
            // 从局部地图中找到最近的5个点，pointSel为索引点，pointSearchInd检索结果点索引，pointSearchSqDis为索引点到这些点距离的平方。
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0)); // 检索出的5个点的协方差矩阵
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0)); // 协方差矩阵的特征值
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0)); // 协方差矩阵的特征向量
            // 如果找到的第5个点(距离最大)小于1m，认为为检索结果有效 
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0; // cx，cy，cz是检索出的5个点的中心坐标
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;
                // 协方差矩阵是对称矩阵
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
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
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    // 以下部分是计算当前点pointSel到检索出的直线的距离和方向，如果距离够近，则认为匹配成功
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // matV1的第一行就是5个点形成的直线的方向，cx，cy，cz是5个点的中心点
                    // 因此x1，y1，z1和x2，y2，z2是经过中心点的直线上的零位两个点，两点之间距离为0.2m
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

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12; // 此处就是点到直线的距离

                    float s = 1 - 0.9 * fabs(ld2); // 如果点刚好在直线上则ld2=0，s=1。距离越远，s越小，赋予比重越低。
                    // 使用系数对法向量加权，实际上相当于对导数(雅可比矩阵加权)
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                    // 经验阈值，判断点到直线距离是否够近，足够近才采纳为优化目标点
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); // 同样寻找到最近的5个点
            // 下面的过程是要求解Ax+By+Cz+1=0平面方程
            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;  // A,B,C

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // 求解方程matA0 * matX0 = matB0
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;
                // (pa,pb,pc)是平面的法向量，这里做归一化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;
                // 这里判断对此5个点构成的平面求解的法向量是否合格。只有当5个点到拟合平面
                // 距离都小于0.2m才认为拟合的平面合格，否则跳过此点。
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }
                // 如果选出的5个点拟合的平面合格，计算pointSel到平面的距离
                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd; // 点到平面的距离(有正负号)
                    // 类似前面，使用距离远近计算一个权重。pd2是在地图坐标系下计算，而分母
                    // pointOri是在雷达坐标系下
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;
                    // 如果权重符合要求，则采纳为加入优化的点
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
    /**
     * @brief 这部分代码是高斯牛顿法的优化，并不是LM优化。目标函数（点到线，点到面的距离）对位姿，
     * 使用的是欧拉角、tx，ty，tz 求导，计算高斯-牛顿法更新方向和步长，然后对transformTobeMapped
     * （存放当前雷达点云位姿）进行更新。
     * @param iterCount 迭代更新次数，默认最大迭代次数为30
     */
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

        // lidar -> camera
        // 计算三轴欧拉角的sin、cos。后面使用旋转矩阵对欧拉角求导使用
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);
        // 找到有匹配关系的角点和平面点，如果找到的可供优化的点数太少，则跳过此次优化
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }
        // matA是雅可比矩阵J
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        // matB是目标函数，也就是距离
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        // matX是高斯-牛顿法计算出的更新向量
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 求雅可比矩阵的值，也就是求目标函数(点到线、平面距离)对(tx,ty,tz,rx,ry,rz)的导数
            // 高斯牛顿法：J^{T}J\Delta{x}=-Jf(x)，\Delta{x}就是要求的更新向量matX。
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
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z; // 目标函数对tx的导数等于法向量的x
            matA.at<float>(i, 4) = coeff.x; // 目标函数对ty的导数等于法向量的y
            matA.at<float>(i, 5) = coeff.y; // 目标函数对tz的导数等于法向量的z
            matB.at<float>(i, 0) = -coeff.intensity; // matB存储的是目标函数的负值，因为J^{T}J\Delta{x}=-Jf(x)
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); // 求解高斯-牛顿法中的增量
        // 如果是第一次求解，判断求解出来的近似Hessian矩阵，也就是J^{T}J=matAtA是否退化
        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
            // 对近似海森矩阵做特征值分解
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
        // 当第一次迭代判断Hessian矩阵退化，后面会使用计算出来的权重matP对增量matX做加权组合
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        // 将增量matX叠加到变量(位姿)transformaTobeMapped中
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
        // 计算roll、pitch、yaw的迭代步长
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        // 计算tx、ty、tz的迭代步长
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
        // 如果迭代的步长达到设定阈值，则认为已经收敛
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty()) // 当前帧为空，返回
            return;
        // 提取的角点、平面点数量要大于指定阈值
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS); // 构建kd树
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 20; iterCount++) // 迭代优化上限20次(原代码30次)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true) // 迭代优化，收敛就跳出迭代
                    break;              
            }

            transformUpdate(); // 用原始RP与当前估计的RP融合(约束z)
        } else { // 角点，平面点数量过少不进行优化
            RCLCPP_WARN(get_logger(), "Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imu_available == true)
        {
            if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }
    /**
     * @brief 角度限制函数，配置文件中可以对RP角做限制，实际感觉不需要
     */
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (stamp2Sec(gpsQueue.front().header.stamp) < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (stamp2Sec(gpsQueue.front().header.stamp) > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

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

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
        // 从优化器拿出最近一帧优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform 将当前帧经过优化的结果更新到全局缓存变量
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D); // 将当前帧的位姿更新到全局轨迹中
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
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

    /* void trans_callback(){
        if(!tf_buffer->_frameExists("map")){
            return;
        }
        mtx_trans.lock();
        br->sendTransform(trans_odom_to_map);
        mtx_trans.unlock();
    } */

    void publishOdometry()
    {
        // Publish odometry for ROS (global) 这里发布的是雷达里程计
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
        pubLaserOdometryGlobal->publish(laserOdometryROS);


        PointTypePose thisPose6DInOdom = trans2PointTypePose(transformTobeMapped); 
        Eigen::Affine3f T_thisPose6DInOdom = pclPointToAffine3f(thisPose6DInOdom);
        mtxtranformOdomToWorld.lock();
        PointTypePose thisPose6DInWorld = trans2PointTypePose(transformOdomToWorld);
        mtxtranformOdomToWorld.unlock();
        Eigen::Affine3f T_thisPose6DInMap = pclPointToAffine3f(thisPose6DInWorld);
        // for(size_t i=0; i<6; ++i){
            // std::cout<<"trans: "<<transformOdomToWorld[i]<<std::endl;
        // }
        /* 发布在map坐标系下的里程计，不直接发布tf关系是发现map->odom和odom->base_link在时间戳对不齐或频率过低，会导致后面对tf树的监听出问题 */
        // 此处也是不对的，重定位结果是robot在map坐标系下位姿，然后联合上面里程计求解map->odom变换。但本代码只在重定位
        // 初始化时进行了一次重定位，后续定位仅依靠里程计，正确做法应不断重定位，优化map->odom的变换。
        Eigen::Affine3f transOdomToMap = T_thisPose6DInMap; 
        // Eigen::Affine3f transOdomToMap = T_thisPose6DInMap * T_thisPose6DInOdom;
        Eigen::Vector3f t = transOdomToMap.translation();
        Eigen::Quaternionf q(transOdomToMap.rotation());
        nav_msgs::msg::Odometry OdometryMap;
        OdometryMap.header.stamp = timeLaserInfoStamp;
        OdometryMap.header.frame_id = "map";
        OdometryMap.child_frame_id = "map_mapping";
        OdometryMap.pose.pose.position.x = t.x();
        OdometryMap.pose.pose.position.y = t.y();
        OdometryMap.pose.pose.position.z = t.z();
        OdometryMap.pose.pose.orientation.x = q.x();
        OdometryMap.pose.pose.orientation.y = q.y();
        OdometryMap.pose.pose.orientation.z = q.z();
        OdometryMap.pose.pose.orientation.w = q.w();
        pubOdometryMap->publish(OdometryMap);

        /* float deltax, deltay, deltaz, deltaR, deltaP, deltaY;
        pcl::getTranslationAndEulerAngles(transOdomToMap, deltax, deltay, deltaz, deltaR, deltaP, deltaY);

        quat_tf.setRPY(deltaR, deltaP, deltaY);
        tf2::Transform t_odom_to_map = tf2::Transform(quat_tf, tf2::Vector3(deltax, deltay, deltaz));
        tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
        tf2::Stamped<tf2::Transform> temp_odom_to_map(t_odom_to_map, time_point, "map");
        geometry_msgs::msg::TransformStamped trans_odom_to_map;
        // mtx_trans.lock();
        tf2::convert(temp_odom_to_map, trans_odom_to_map);
        trans_odom_to_map.child_frame_id = odometryFrame;
        // mtx_trans.unlock();
        br->sendTransform(trans_odom_to_map); */

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

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
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
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
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
        // publish path
        
        if (pubPath->get_subscription_count() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath->publish(globalPath);
        }
        
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Global Localization Started.\033[0m");
    
    std::thread localizeThread(&mapOptimization::globalLocalizeThread, MO);
    std::thread loopthread(&mapOptimization::loopClosureThread, MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, MO);

    exec.spin();

    rclcpp::shutdown();

    localizeThread.join(); 
    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
