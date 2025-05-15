#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

/*
Veledyne激光雷达的数据类型
+ 这里使用16byts对齐，是因为SIMD(Single Instruction Multiple Data)架构需要使用128bit对齐，跟现代CPU架构和指令集有关
+ 在struct定义union结构体但不赋予union结构体名字，使用了`匿名union`用法，union的成员被认为定义在域中：
https://stackoverflow.com/questions/13624760/how-to-use-c-union-nested-in-struct-with-no-name
+ pcl注册自定义点云类型：https://pcl.readthedocs.io/projects/tutorials/en/latest/adding_custom_ptype.html
*/
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)
/**
 * @brief 岚沃自定义消息类型
 */
struct LiovxPointCustomMsg
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float time;
    uint16_t ring;
    uint16_t tag;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (LiovxPointCustomMsg,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity) (float, time, time)
    (uint16_t, ring, ring) (uint16_t, tag, tag)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = LiovxPointCustomMsg;

const int queueLength = 2000; // IMU数据队列长度

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock; // IMU队列线程锁
    std::mutex odoLock; // IMU里程计线程锁
    // 订阅原始雷达数据
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr subLaserCloud;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;
    // 发布去畸变后的点云信息及整合信息（原始点云、去畸变点云、该帧点云的初始旋转角（来自IMU原始RPY）、该帧点云初始位姿（来自IMU里程计）     
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    // 订阅原始IMU数据
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    std::deque<sensor_msgs::msg::Imu> imuQueue;
    // 订阅IMU里程计数据，来自imuPreintegration
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    std::deque<nav_msgs::msg::Odometry> odomQueue;
    // livox类型点云队列和提取出的当前帧点云
    std::deque<livox_ros_driver2::msg::CustomMsg> cloudQueue;
    livox_ros_driver2::msg::CustomMsg currentCloudMsg;
    // 记录每一帧点云从其实到结束过程所有的IMU数据，imuRot*是这段时间内的角速度累加的结果
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];
    // 记录每一帧点云气质过程中imuTime、imuRotXYZ的实际数据长度
    int imuPointerCur;
    // 处理第一个点时，将该点的旋转取逆，记录到transStarInverse中，后续方便计算旋转的增量
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag; // 当点云的time/t字段不可用，也就是点云中不包含该点的时间戳，无法进行去畸变，直接返回原始点云
    cv::Mat rangeMat; // 存储点云的range图像

    bool odomDeskewFlag; // 是否有合适的里程计数据
    // 记录从IMU里程计计算出来的平移增量，用来做平移去畸变，实际中没有使用到
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::msg::CloudInfo cloudInfo; // 用于发布的数据结构
    double timeScanCur;                // 当前雷达帧的开始时间
    double timeScanEnd;                // 当前雷达帧的结束时间
    std_msgs::msg::Header cloudHeader; // 当前雷达帧的header

    vector<int> columnIdnCountVec;


public:
    ImageProjection(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imageProjection", options), deskewFlag(0)
    {   
        /*
        这里区别于普通的监听器在于使用ROS2的callbackGroup来指定下面三个callback都是不可并行的MutuallyExclusive.
        MutuallyExclusive 保证了所有缓存队列都是时间有序的.
        PS:根据ROS2的文档，没有指定callbackGroup的回调函数会注册到默认的callbackGroup，
        默认的callbackGroup本身就是Mutually Exclusive CallbackGroups，但是这样多个callback就没法并行执行。
        期望不同的callback可以并行执行，但是相同的callback不并行，推荐的做法就是将不同的callback注册到不同的
        Mutually Exclusive CallbackGroups
        + 关于ROS2的执行器Executors： https://docs.ros.org/en/foxy/Concepts/About-Executors.html
        + Using Callback Groups: https://docs.ros.org/en/foxy/How-To-Guides/Using-callback-groups.html
        */
        callbackGroupLidar = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidarOpt = rclcpp::SubscriptionOptions();
        lidarOpt.callback_group = callbackGroupLidar;
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&ImageProjection::imuHandler, this, std::placeholders::_1),
            imuOpt);
        subOdom = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&ImageProjection::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        subLaserCloud = create_subscription<livox_ros_driver2::msg::CustomMsg>(
            pointCloudTopic, qos_lidar,
            std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1),
            lidarOpt);

        pubExtractedCloud = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos);
        // 为动态指针和动态数组分配内存
        // 类似pcl::PointCloud<PointType>::Ptr类型的类成员变量，声明是没有进行初始化，直接调用会报错
        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        // 最大的点云数不会超过N_SCAN*Horizon_SCAN
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        // 在点云去畸变的时候将range数据展开为一维向量
        // start_ring_index记录每条ring在一维向量中的开始位置，同理end_ring_index为结束位置
        cloudInfo.start_ring_index.assign(N_SCAN, 0);
        cloudInfo.end_ring_index.assign(N_SCAN, 0);
        // 记录一维range数据中每一个点在原始range图片中属于哪一个列
        cloudInfo.point_col_ind.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.point_range.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }
    // @brief 重置各项参数，注意，这部分代码的执行逻辑是接收一帧雷达数据就进行一次完整的处理
    // 因此，每次处理完都会调用这个函数清空缓存变量
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}
    /**
     * @brief IMU原始数据的回调函数，主要功能是将IMU原始数据对齐到Lidar坐标系，存入缓存
     * @param imuMsg 原始的IMU数据
     */
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {   // 将IMU坐标系下的IMU数据转换到雷达坐标系(做旋转对齐)
        sensor_msgs::msg::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }
    /**
     * @brief IMUl里程计话题的回调函数，来自imuPreintegration发布的IMU里程计
     * @param odomtryMsg IMU里程计消息
     */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }
    /**
     * @brief 原始雷达点云话题的回调函数
     * @details 1、订阅原始lidar数据，转换点云为统一格式，提取点云信息。添加一帧点云到队列
     * 取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性。
     * 2、从IMU数据和IMU里程计中提取去畸变信息。遍历当前激光帧起止时刻之间的imu数据，初始时
     * 刻对应的imu姿态角RPY设为当前帧的初始姿态角。用角速度、时间积分，计算每一时刻相对于初
     * 始时刻的旋转量，初始旋转量设为0。对于IMU里程计数据，遍历当前帧起止时刻的IMU里程计数据，
     * 初始时刻对应IMU里程计设为当前帧的初始位姿。用起止时刻对应的IMU里程计，计算相对位姿变
     * 换，保存平移增量。
     * 3、当前帧激光点云去畸变：检查激光点距离、扫描线是否合规。激光运动畸变矫正，保存激光点。
     * 4、提取优先激光点，整合并发布。
     * 5、发布当前帧矫正后点云，有效点和其它信息。
     * 6、重置参数，接收每帧lidar数据都要重置这些参数。
     */
    void cloudHandler(const livox_ros_driver2::msg::CustomMsg::SharedPtr laserCloudMsg)
    {   // 提取、转换点云为统一格式
        if (!cachePointCloud(laserCloudMsg))
            return;
        // 从imu原始数据和imu里程计数据中提取去畸变信息
        if (!deskewInfo())
            return;
        // 当前帧激光点云畸变矫正
        projectPointCloud();
        // 提取有效激光点，集合信息到准备发布的数据包
        cloudExtraction();
        // 发布当前帧校正后点云，有效点和其它信息
        publishClouds();
        // 重置参数
        resetParameters();
    }
    /**
     * @brief 将原始雷达数据转换为统一格式
     * @param Msg 原始点云数据
     * @param cloud  转换后的点云格式(此处为前面自定义注册到pcl的格式)
     */
    void moveFromCustomMsg(livox_ros_driver2::msg::CustomMsg &Msg, pcl::PointCloud<PointXYZIRT> & cloud)
    {
        cloud.clear();
        cloud.reserve(Msg.point_num);
        PointXYZIRT point;

        cloud.header.frame_id=Msg.header.frame_id;
        cloud.header.stamp= (uint64_t)((Msg.header.stamp.sec*1e9 + Msg.header.stamp.nanosec)/1000) ;
        // cloud.header.seq=Msg.header.seq;

        for(uint i=0;i<Msg.point_num-1;i++)
        {   
            Eigen::Vector3d p(Msg.points[i].x, Msg.points[i].y, Msg.points[i].z);
            Eigen::Vector3d p_base = lidar2baseRot * p + lidar2baseTrans;
            // std::cout << "lidar2baseRot = \n" << lidar2baseRot << std::endl;
            // std::cout << "lidar2baseTrans = " << lidar2baseTrans.transpose() << std::endl;
            point.x=p_base.x(); 
            point.y=p_base.y(); 
            point.z=p_base.z(); 
            /* point.x=Msg.points[i].x; 
            point.y=Msg.points[i].y; 
            point.z=Msg.points[i].z;  */
            point.intensity=Msg.points[i].reflectivity; 
            point.tag=Msg.points[i].tag; 
            point.time=Msg.points[i].offset_time*1e-9; 
            point.ring=Msg.points[i].line; 
            cloud.push_back(point);
        }
    }
    /**
     * @brief 转换点云为统一格式，提取点云信息
     * 添加一帧激光点云到队列，取出最早一帧作为当前帧。计算起止时间戳，检查数据有效性。 
     */
    bool cachePointCloud(const livox_ros_driver2::msg::CustomMsg::SharedPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;
 
        // convert cloud 将点云转换为统一的格式
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();

        if (sensor == SensorType::LIVOX){
            moveFromCustomMsg(currentCloudMsg, *laserCloudIn);
        }
        else
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
            rclcpp::shutdown();
        }
        // 计算当前帧点云的开始时间和结束时间，开始时间用点云header中时间戳，结束时间用最后
        // 一个点相对于开始时间间隔加上当前帧开始时间得到。
        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = stamp2Sec(cloudHeader.stamp);  // 当前帧开始时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // 当前帧结束时间

        // check dense flag 检查点云是否dense格式（去除了NaN数据）
        if (laserCloudIn->is_dense == false)
        {
            RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
            rclcpp::shutdown();
        }
        // 下面为检查是否包含ring字段和time字段，源代码未注释 (适配岚沃注释掉的)
        // // check ring channel
        // static int ringFlag = 0;
        // if (ringFlag == 0)
        // {
        //     ringFlag = -1;
        //     for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
        //     {
        //         if (currentCloudMsg.fields[i].name == "ring")
        //         {
        //             ringFlag = 1;
        //             break;
        //         }
        //     }
        //     if (ringFlag == -1)
        //     {
        //         RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
        //         rclcpp::shutdown();
        //     }
        // }

        // // check point time
        // if (deskewFlag == 0)
        // {
        //     deskewFlag = -1;
        //     for (auto &field : currentCloudMsg.fields)
        //     {
        //         if (field.name == "time" || field.name == "t")
        //         {
        //             deskewFlag = 1;
        //             break;
        //         }
        //     }
        //     if (deskewFlag == -1)
        //         RCLCPP_WARN(get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        // }

        return true;
    }
    /**
     * @brief 从imu数据和IMU里程计中提取畸变信息
     * @details 1、imu数据：遍历当前激光帧起始时间内的imu数据，初始时刻对应imu的姿态角RPY
     * 设为当前帧的初始姿态角。用角速度、时间积分，计算每一时刻相对初始时刻的旋转量。
     * 2、IMU里程计数据：遍历当前激光帧起始时间内的imu里程计数据，初始时刻位姿对应imu里程计
     * 的位姿。用起始终止时刻对应的imu里程计，计算相对位姿变换，保存平移增量。
     */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock); // imu数据锁
        std::lock_guard<std::mutex> lock2(odoLock); // IMU里程计锁

        // 确保目前imu队列中数据能够覆盖当前帧的起止时间。由于去畸变部分必须依赖高频的imu，
        // 所以这里没有检查通过，当前点云帧会被跳过。
        if (imuQueue.empty() ||
            stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
            stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
        {
            RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
            return false;
        }
        
        // 从imu队列中计算去畸变信息
        imuDeskewInfo();
        // 从IMU里程计中计算去畸变信息
        odomDeskewInfo();

        return true;
    }
    /**
     * @brief 从imu数据中提取去畸变信息
     * @details 遍历当前激光帧起止时间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初
     * 始姿态角。用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0。
     */
    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false; // cloudInfo是要发布出去的话题数据，等imu数据处理完成才设为true
        // imu数据和当前帧点云时间对齐
        while (!imuQueue.empty())
        {   // 去除早于当前点云帧开始时间-0.01的imu数据
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;
        // 这里就是对imu的加速度和角速度做积分，若点云为10hz，则每一帧扫描时间大约为100ms
        // 时间非常短，所以这里的积分采用的是近似算法。
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i]; // 获取一个imu数据
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp); // 当前imu数据对应时间
            // 从imu数据中提取当前帧开始时刻的roll、pitch、yaw(原文从9轴imu中来，mid360经过了处理)
            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            if (currentImuTime > timeScanEnd + 0.01) // 当前帧点云对应的imu数据处理完成，跳出循环
                break;

            if (imuPointerCur == 0){ //如果是第一次的imu数据，初始的位姿设置为0
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity 获取角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation 对角度做积分。这里是简单的角速度乘以时间间隔近似，对小量近似
            // 但是这不是角速度的积分公式，切记
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imu_available = true;
    }
    /**
     * @brief 从IMU里程计数据中提取去畸变信息
     * @details 遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧
     * 的初始位姿。用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量。
     */
    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;
        // IMU里程计数据和激光帧时刻对齐
        while (!odomQueue.empty())
        {
            if (stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }
        // 如果没有可用的IMU里程计数据，直接返回
        if (odomQueue.empty())
            return;
        // 再次检查数据时间戳，确保起始的IMU里程计数据在激光帧开始时间之前
        if (stamp2Sec(odomQueue.front().header.stamp) > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::msg::Odometry startOdomMsg;
        // 获取点云帧起始时刻对应的imu里程计数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
                continue;
            else
                break;
        }
        // 下面是将点云帧起始时刻对应的imu里程数据提取，塞到cloudInfo
        tf2::Quaternion orientation;
        tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;

        cloudInfo.odom_available = true;

        // get end odometry at the end of the scan
        // 是否使用IMU里程计数据对点云去畸变的标志位
        // 前面找到了点云起始时刻的IMU里程计，只有在同时找到结束时刻的IMU里程计数据，才可以
        // 利用时间插值算出每一个点对应时刻的位姿做去畸变
        // 注意：！在官方代码中，最后并没有使用IMU里程计数据做去畸变。所以这个标志位世纪没有被使用，
        // 下面的这些代码实际上也没有被使用到
        // 为什么没有使用IMU里程计做去畸变处理？
        // - 原代码中的注释写的是速度较低的情况下不需要做平移去畸变
        odomDeskewFlag = false;

        if (stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd)
            return;

        nav_msgs::msg::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        // 下面这段代码从 startOdomMsg和endOdomMsg计算该帧点云开始和结束之间的位姿变化
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 求出点云从End坐标系转换到Start坐标系的变换
        // 作者一开始应该是用IMU里程计计算位移变换，所以真正有用的是odomIncreXYZ
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }
    /**
     * @brief 根据某一个点的时间戳从imu去畸变信息列表中找到对应的旋转角
     * @param pointTime 需要获取的时间点
     * @param rotXCur 输出的roll角
     * @param rotYCur 输出的pitch角
     * @param rotZCur 输出的yaw角
     */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // imuPointerFront为找到的IMU去畸变信息的索引
        // 引起if条件主要是因为可能最后一个IMU的时间戳依然是小于雷达点的时间戳，
        // 这个时候就直接使用最后一个IMU对应的角度
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else { // 根据时间戳比例进行角度插值，计算出更准确的旋转
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }
    /**
     * @brief 根据某一个点的时间戳从IMU里程计去畸变信息列表中找到对应的平移。
     * 这个函数根据时间间隔对平移量进行插值得到起止时间段之间任意时刻的位移
     * 
     * @param relTime 点云中某一个点的time字段（相对于起始点的时间）
     * @param poseXCur 输出的X轴方向平移量
     * @param poseYCur 输出的Y轴方向平移量
     * @param poseZCur 输出的Z轴方向平移量
    */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }
    /**
     * @brief 雷达单点运动畸变矫正
     * @param 点云中某个点位置
     * @param 该点相对于该帧点云开始时间的相对时间
     */
    PointType deskewPoint(PointType *point, double relTime)
    {   // 如果IMU去畸变信息或者IMU里程计去畸变信息不可用，返回原始点
        if (deskewFlag == -1 || cloudInfo.imu_available == false)
            return *point;

        double pointTime = timeScanCur + relTime; // 计算该点的绝对时间
        // 找到旋转量
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        // 找到平移量(前面没改的话，始终是(0,0,0))
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        // 缓存第一个点对应的位姿
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }
        // 计算当前点到起始点的变换矩阵
        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        // 三维点坐标变换，这里可以直接调用PCL的接口
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }
    /**
     * @brief 执行点云去畸变
     * @details 检查激光点距离、扫描线是否合规。单点去畸变。保存激光点。
     */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size(); // 获取激光帧点云数
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // range从点云的x,y,z中计算出来，没有使用话题中的。range映射成2D矩阵时，
            // 行所引等于该点的ring，列索引根据角度和分辨率计算得到
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 可用于执行线束级别降采样
            if (rowIdn % downsampleRate != 0)
                continue;
            // 计算该点的列索引(此处根据mid360做了不一样的处理)
            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }


            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            // 单点进行去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // 将该点的range记录到rangeMat中
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 将去完畸变的点云存储在中间变量中
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }
    /**
     * @brief 提取有效激光点，集合信息到准备发布的cloudInfo数据包中
     */
    void cloudExtraction()
    {
        int count = 0; // 有效激光点数量
        // extract segmented cloud for lidar odometry
        // 提取特征的时候，每一条线束的前5个点和最后5个点不考虑，因为在计算曲率过程中用前后
        // 总共十个点计算，因此每一条线束前后5个点无法计算。
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.start_ring_index[i] = count - 1 + 5;
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.point_col_ind[count] = j;
                    // save range info
                    cloudInfo.point_range[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.end_ring_index[i] = count -1 - 5;
        }
    }
    /**
     * @brief 发布当前帧矫正后的点云以及其它有效信息
     */
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<ImageProjection>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Image Projection Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}


