#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "lio_sam/utility.hpp"

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

class TransformFusion : public ParamServer {
   public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subMapOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImuOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLaserOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupMapOdometry;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;

    Eigen::Isometry3d lidarOdomAffine;
    Eigen::Isometry3d mapOdomAffine;
    Eigen::Isometry3d imuOdomAffineFront;
    Eigen::Isometry3d imuOdomAffineBack;

    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    tf2::Stamped<tf2::Transform> lidar2Baselink;

    double lidarOdomTime = -1;
    double mapOdomTime = -1;
    deque<nav_msgs::msg::Odometry> imuOdomQueue;

    TransformFusion(const rclcpp::NodeOptions& options) : ParamServer("lio_sam_transformFusion", options) {
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

        callbackGroupImuOdometry = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupLaserOdometry = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupMapOdometry = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOdomOpt = rclcpp::SubscriptionOptions();
        imuOdomOpt.callback_group = callbackGroupImuOdometry;
        auto laserOdomOpt = rclcpp::SubscriptionOptions();
        laserOdomOpt.callback_group = callbackGroupLaserOdometry;
        auto mapOdomOpt = rclcpp::SubscriptionOptions();
        mapOdomOpt.callback_group = callbackGroupMapOdometry;


        subLaserOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry", qos,
            std::bind(&TransformFusion::lidarOdometryHandler, this, std::placeholders::_1), laserOdomOpt);
        subMapOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/mapodom", qos, 
            std::bind(&TransformFusion::MapOdometryHandler, this, std::placeholders::_1), mapOdomOpt); 
        subImuOdometry = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&TransformFusion::imuOdometryHandler, this, std::placeholders::_1), imuOdomOpt);

        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic, qos_imu);
        pubImuPath = create_publisher<nav_msgs::msg::Path>("lio_sam/imu/path", qos);

        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    Eigen::Isometry3d odom2affine(nav_msgs::msg::Odometry odom) {
        tf2::Transform t;
        tf2::fromMsg(odom.pose.pose, t);
        return tf2::transformToEigen(tf2::toMsg(t));
    }

    void lidarOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg){
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = stamp2Sec(odomMsg->header.stamp);
    }

    void MapOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg){
        std::lock_guard<std::mutex> lock(mtx);
        mapOdomAffine = odom2affine(*odomMsg);
        mapOdomTime = stamp2Sec(odomMsg->header.stamp);
    }

    void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg) {
        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1 || mapOdomTime == -1) return;
        while (!imuOdomQueue.empty()) {
            if (stamp2Sec(imuOdomQueue.front().header.stamp) <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Isometry3d imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Isometry3d imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Isometry3d imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Isometry3d imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        auto t = tf2::eigenToTransform(imuOdomAffineLast);
        tf2::Stamped<tf2::Transform> tCur;
        tf2::convert(t, tCur);
        // Eigen::Isometry3d mapOdomAffineLast = mapOdomAffine * imuOdomAffineLast;
        auto tt = tf2::eigenToTransform(mapOdomAffine);
        tf2::Stamped<tf2::Transform> ttCur;
        tf2::convert(tt, ttCur);


        // publish latest odometry
        nav_msgs::msg::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = t.transform.translation.x;
        laserOdometry.pose.pose.position.y = t.transform.translation.y;
        laserOdometry.pose.pose.position.z = t.transform.translation.z;
        laserOdometry.pose.pose.orientation = t.transform.rotation;
        pubImuOdometry->publish(laserOdometry);

        // publish tf
        if (lidarFrame != baselinkFrame) {
            try {
                tf2::fromMsg(tfBuffer->lookupTransform(lidarFrame, baselinkFrame, rclcpp::Time(0)), lidar2Baselink);
            } catch (tf2::TransformException ex) {
                RCLCPP_ERROR(get_logger(), "%s", ex.what());
            }
            tf2::Stamped<tf2::Transform> tb(tCur * tf2::Transform::getIdentity(), tf2_ros::fromMsg(odomMsg->header.stamp),
                                            odometryFrame);
            tf2::Stamped<tf2::Transform> ttb(ttCur, tf2_ros::fromMsg(odomMsg->header.stamp), mapFrame);
            tCur = tb;
            ttCur = ttb;
        }
        geometry_msgs::msg::TransformStamped ts;
        tf2::convert(tCur, ts);
        ts.child_frame_id = baselinkFrame;
        geometry_msgs::msg::TransformStamped tts;
        tf2::convert(ttCur, tts);
        tts.child_frame_id = odometryFrame;
        tfBroadcaster->sendTransform(ts);
        tfBroadcaster->sendTransform(tts);

        // publish IMU path
        static nav_msgs::msg::Path imuPath;
        static double last_path_time = -1;
        double imuTime = stamp2Sec(imuOdomQueue.back().header.stamp);
        if (imuTime - last_path_time > 0.1) {
            last_path_time = imuTime;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while (!imuPath.poses.empty() && stamp2Sec(imuPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath->get_subscription_count() != 0) {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath->publish(imuPath);
            }
        }
    }
};
/**
 * @brief 这个类主要功能是订阅激光里程计(来自mapOptimization)，对两帧激光里程计中间过程使用的imu
 *        数据积分，得到高频的imu里程计。
 * @details IMU信号积分使用的是gtsam预积分。共使用了两个队列imuQueImu和imuQueOpt，以及两个
 * 预积分器imuPreintegratorImu和imuPreintegratorOpt。imuQueOpt和imuPreintegratorOpt
 * 主要是根据里斯信息计算IMU数据bias给真正的里程计预积分器使用，imuQueImu和imuPreintegratorImu
 * 是真正用来做IMU里程计优化的。模块中有两个handler，分别处理雷达里程计和IMU原始数据。雷达
 * 里程计的handler中，主要是将新到来的雷达里程计之前的IMU做积分，得出bias。IMU的handler主
 * 要是对当前里程计之后，下一时刻雷达里程计到来之前时刻的imu数据积分，并发布IMU里程计。 
 * 
 * 订阅：
 * 1、imu原始数据
 * 2、lidar里程计(来自mapOptimization)
 * 发布：
 * 1、imu里程计
 */
class IMUPreintegration : public ParamServer {
   public:
    std::mutex mtx;
    // 订阅imu原始数据
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    // 订阅lidar里程计（来自mapOptimization）
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    // 发布imu里程计
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;

    // 系统初始化标志，主要用于标志gtsam的初始化
    bool systemInitialized = false;
    // priorXXXNoise是因子图中添加第一个因子时指定的噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    // correctionNoise是小噪声，correctionNoise2是大噪声。在雷达里程计方差大的情况下使用大噪声，反之小噪声。
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias; // imu的bias噪声模型
    // imuIntegratorOpt_ is responsile for pre-integrating the IMU data between 
    // two laser odometry frames and estimating the IMU bias。
    gtsam::PreintegratedImuMeasurements* imuIntegratorOpt_;
    // 根据最新的激光里程计，以及后续imu数据，预测从当前激光雷达里程计往后的位姿
    gtsam::PreintegratedImuMeasurements* imuIntegratorImu_;
    // imuQueOpt为imuIntegratorOpt_提供数据来源，将当前激光帧之前的imu数据做积分，优化出bias
    std::deque<sensor_msgs::msg::Imu> imuQueOpt;
    // imuQueImu为imuIntegratorIMU_提供数据来源，优化当前激光里程计之后，下一帧激光里程计到来之前的位姿。
    std::deque<sensor_msgs::msg::Imu> imuQueImu;
    // 因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;
    // 第一帧初始化标签
    bool doneFirstOpt = false;
    // 上一个imu数据的时间(在imu的handler中使用)
    double lastImuT_imu = -1;
    // using in laser odometry handler
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer; // 因子图优化器  
    gtsam::NonlinearFactorGraph graphFactors; // 非线性因子图
    gtsam::Values graphValues; // 因子图变量的值

    const double delta_t = 0; // 在做IMU数据和雷达里程计同步过程中的时间间隔

    int key = 1;
    // imu坐标系和lidar坐标系之间的转换，在最开始处理imu数据时候已经在旋转关系上对齐了，
    // 因此这里只对两坐标系的平移做对齐
    gtsam::Pose3 imu2Lidar =
        gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu =
        gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration(const rclcpp::NodeOptions& options) : ParamServer("lio_sam_imu_preintegration", options) {
        callbackGroupImu = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive); // 指定执行器类型
        callbackGroupOdom = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        // 添加进各自的回调组
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu, std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1), imuOpt);
        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos,
            std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1), odomOpt);
        // 发布IMU里程计信息
        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic + "_incremental", qos_imu);
        // IMU预积分器初始化
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        // 预计分器的角速度、加速度、积分结果的初始协方差设置
        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);  // acc white noise in continuous
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);  // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);  // error committed in integrating position from velocities
        // imu积分器的初始偏置都设置为0
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
           // assume zero initial bias
        // 因子图中先验变量的噪声模型，以及IMU偏差的噪声模型
        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());  // rad,rad,rad,m, m, m
        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);               // m/s
        priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);             // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());  // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());  // rad,rad,rad,m, m, m
        noiseModelBetweenBias =
            (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN)
                .finished();
        // IMU预积分器，在IMU里程计中使用
        imuIntegratorImu_ =
            new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);  // setting up the IMU integration for IMU
                                                                         // message thread
        // IMU偏差预积分器，在雷达里程计线程中使用
        imuIntegratorOpt_ =
            new gtsam::PreintegratedImuMeasurements(p,
                                                    prior_imu_bias);  // setting up the IMU integration for optimization
    }
    /**
     * @brief 重置优化器和因子图，在第一帧雷达里程计到来的时候重置，每100帧激光里程计也重置
     * 一次参数
     */
    void resetOptimization() {
        // 重置ISAM2优化器
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);
        // 重置因子图
        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;
        // 重置因子图中变量
        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }
    /**
     * @brief 重置中间变量，在优化失败的时候会进行一次重置，让整个系统重新初始化。
     */
    void resetParams() {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }
    /**
     * @brief 雷达里程计的回调函数
     */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg) {
        // 加锁，防止imuQueOpt队列被imuHandler线程修改
        std::lock_guard<std::mutex> lock(mtx);
        // 当前雷达里程计时间
        double currentCorrectionTime = stamp2Sec(odomMsg->header.stamp);

        // make sure we have imu data to integrate
        if (imuQueOpt.empty()) return;
        // 提取lidar里程计位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        // 检查雷达里程计是否准确，convriance[0]在mapOptimization中被置位，如果degenrate
        // 为true，则雷达里程计不准确，使用大的噪声模型
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        gtsam::Pose3 lidarPose =
            gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // 0. initialize system
        // 初始化系统，在第一帧或者优化出错的情况下重新初始化
        if (systemInitialized == false) {
            resetOptimization();

            // pop old IMU message 同步imuQueOpt中与雷达的时间戳
            while (!imuQueOpt.empty()) {
                if (stamp2Sec(imuQueOpt.front().header.stamp) < currentCorrectionTime - delta_t) {
                    lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp);
                    imuQueOpt.pop_front();
                } else
                    break;
            }
            // initial pose 初始化因子图的先验状态
            prevPose_ = lidarPose.compose(lidar2Imu); //将雷达里程计位姿平移到IMU坐标系，只做了平移
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise); // 构造位姿因子
            graphFactors.add(priorPose); // 将位姿因子加入因子图
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values 将初始状态设置为因子图变量的初始值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once 因子图和初始变量加入优化器做一次优化
            optimizer.update(graphFactors, graphValues);
            // 清空因子图。因子已经被记录到优化器中。这是参考gtsam官方的example
            // example:gtsam/examples/VisualSAM2Example.cpp
            graphFactors.resize(0);
            graphValues.clear();
            // 重置两个预积分器
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            // 重置起始帧的key值
            key = 1;
            systemInitialized = true; // 设置系统已经初始化标志(主要是对gtsam的初始化)
            return;
        }
        // 每100帧lidar里程计重置一下优化器，删除旧因子图，加快优化速度
        // reset graph for speed
        if (key == 100) {
            // get updated noise before reset
            // 从优化器中先缓存一下当前优化出来的变量的方差
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise =
                gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise =
                gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise =
                gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
            // reset graph
            resetOptimization(); // 重置优化器和因子图
            // add pose 把上一次优化出的位姿作为重新初始化的priorPose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity 把上一次优化出的速度作为重新初始化的priorVel
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias 把上一次优化出的bias作为重新初始化的priorBias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }

        // 1. integrate imu data and optimize
        // 将imuQueOpt队列中所有早于当前雷达里程计的数据进行积分，获取最新的IMU bias
        while (!imuQueOpt.empty()) {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::msg::Imu* thisImu = &imuQueOpt.front();
            double imuTime = stamp2Sec(thisImu->header.stamp);
            if (imuTime < currentCorrectionTime - delta_t) {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y,
                                   thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y,
                                   thisImu->angular_velocity.z),
                    dt);

                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            } else
                break;
        }
        // add imu factor to graph
        // 这下面是将imu预计分结果加入因子图中，同时加入雷达里程计因子
        const gtsam::PreintegratedImuMeasurements& preint_imu =
            dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        // 添加imu偏置漂移(两个imu数据之间的bias差距)。detaTij()获取的是imu预积分器从上次
        // 积分输入到当前输入的时间。noiseModelBetweenBias是提前标定好的imu偏差漂移噪声
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor 将雷达的位姿平移到IMU坐标系中(只做了平移)
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        // 添加Pose因子，degenerate用来选择噪声大小
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        // imuIntegratorOpt->predict()输入之前的状态和偏差，预测当前时刻的状态
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // 将IMU预积分的结果作为当前时刻因子图变量的初值。
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        optimizer.update(graphFactors, graphValues); // 更新一次因子图
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        // 从优化器中获取当前经过优化后的估计值
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_ = result.at<gtsam::Pose3>(X(key));
        prevVel_ = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        // 重置预积分器
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        // 检查优化器结果，当前优化出的速度过快(30m/s)，或一些优化指标或环节出现问题，重置优化器。
        if (failureDetection(prevVel_, prevBias_)) {
            resetParams();
            return;
        }

        // 2. after optiization, re-propagate imu odometry preintegration
        // 这一部分是将前面优化得到的bias结果传递到IMU里程计的预积分器中，让IMU里程计预积分
        // 器使用最新估计出来的bias进行积分
        prevStateOdom = prevState_; // 缓存当前的状态
        prevBiasOdom = prevBias_; // 缓存当前的bias
        // first pop imu message older than current correction data
        // 做imuQueImu和雷达里程计的时间同步
        double lastImuQT = -1;
        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t) {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp);
            imuQueImu.pop_front();
        }
        // repropogate // 对当前IMU队列中的数据全部积分
        if (!imuQueImu.empty()) {
            // reset bias use the newly optimized bias
            // 将bias优化器优化出的最新偏差重置到IMU里程计预积分器中
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            // 将IMU队列中从当前雷达里程计之后的IMU数据进行积分。后续每个新到来的IMU数据
            // 在(ImuHandler中)可以直接进行积分。
            for (int i = 0; i < (int)imuQueImu.size(); ++i) {
                sensor_msgs::msg::Imu* thisImu = &imuQueImu[i];
                double imuTime = stamp2Sec(thisImu->header.stamp);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y,
                                   thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y,
                                   thisImu->angular_velocity.z),
                    dt);
                lastImuQT = imuTime;
            }
        }
        // 对因子图的当前帧索引加1
        ++key;
        // 设置首次优化标志位为真，如果这段代码没有先执行，imuHandler无法对新到来的IMU数据进行积分
        doneFirstOpt = true;
    }
    /**
     * @brief 检查优化器优化结果，当优化速度过快，或者bias过大，则认为优化或者某个环节有问题，重置优化器。
     * @param 
     */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur) {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30) {
            RCLCPP_WARN(get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0) {
            RCLCPP_WARN(get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {
        std::lock_guard<std::mutex> lock(mtx);

        sensor_msgs::msg::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false) return;

        double imuTime = stamp2Sec(thisImu.header.stamp);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(
            gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
            gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry->publish(odometry);

    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor e;

    auto ImuP = std::make_shared<IMUPreintegration>(options);
    auto TF = std::make_shared<TransformFusion>(options);
    e.add_node(ImuP);
    e.add_node(TF);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
