#pragma once

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm> 
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "lio_sam/nanoflann.hpp"
#include "lio_sam/KDTreeVectorOfVectorsAdaptor.hpp"

#include "lio_sam/tictoc.hpp"

using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;
// 点云数据类型
using SCPointType = pcl::PointXYZI; // using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
using KeyMat = std::vector<std::vector<float> >; // 二维容器类型数组
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >; // kd树


// namespace SC2
// {
/**
 * @brief 用于打印scan context库导入信息
 */
void coreImportTest ( void );

// 一些辅助函数
// sc param-independent helper functions 
/**
 * @brief 输出向量(0,0)->(x,y)的角度，逆时针
 */
float xy2theta( const float & _x, const float & _y );
/**
 * @brief 将一个矩阵按列循环向右移位
 * @param _mat 待操作数组
 * @param _num_shift 偏移量
 * @return 移位后的数组
 */
Eigen::MatrixXd circshift( Eigen::MatrixXd &_mat, int _num_shift );
/**
 * @brief 将eigen类型数组转换为std::vector
 */
std::vector<float> eig2stdvec( Eigen::MatrixXd _eigmat );

/**
 * @brief SC类
 */
class SCManager
{
public:
    // 构造函数，这里可以预留空间来提高性能(提前分配std::vector的存储空间)
    SCManager( ) = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.
    /**
     * @brief SC描述子的构建函数，将点云数据映射到一个二维极坐标网格中
     * @param _scan_down 降采样后的点云
     * @return 一个二维矩阵形式的坐标描述子，每个网格存储对应极坐标内最大高度z值
     */
    Eigen::MatrixXd makeScancontext( pcl::PointCloud<SCPointType> & _scan_down );
    /**
     * @brief 二维矩阵行均值
     * @param _desc 输入的二维矩阵
     * @return 返回一维均值
     */
    Eigen::MatrixXd makeRingkeyFromScancontext( Eigen::MatrixXd &_desc );
    /**
     * @brief 二维矩阵列均值      
     */
    Eigen::MatrixXd makeSectorkeyFromScancontext( Eigen::MatrixXd &_desc );
    /**
     * @brief 获取两个矩阵之间差的范数的最小值对应的移位量
     * @param _vkey1 矩阵1
     * @param _vkey2 矩阵2
     * @return 返回对应的移位量
     */
    int fastAlignUsingVkey ( Eigen::MatrixXd & _vkey1, Eigen::MatrixXd & _vkey2 ); 
    /**
     * @brief 度量两个SC矩阵之间的相似度
     * @param _sc1 SC矩阵1
     * @param _sc2 SC矩阵2
     * @return 返回相似度度量
     */
    double distDirectSC ( Eigen::MatrixXd &_sc1, Eigen::MatrixXd &_sc2 ); // "d" (eq 5) in the original paper (IROS 18)
    /**
     * @brief 计算两个SC矩阵最小距离和对应的偏移量，用到了一些快速搜索的方法
     */
    std::pair<double, int> distanceBtnScanContext ( Eigen::MatrixXd &_sc1, Eigen::MatrixXd &_sc2 ); // "D" (eq 6) in the original paper (IROS 18)

    // User-side API
    /**
     * @brief 输入点云，生成SC描述子矩阵及相关的key
     */
    void makeAndSaveScancontextAndKeys( pcl::PointCloud<SCPointType> & _scan_down );
    /**
     * @brief 检测闭环，通过SC和特征键在历史点云数据中找到与当前点云匹配的候选帧，判断是否存在闭环
     */
    std::pair<int, float> detectLoopClosureID( void ); // int: nearest node index, float: relative yaw  

    // for ltmapper 
    const Eigen::MatrixXd& getConstRefRecentSCD(void);

public:
    // hyper parameters ()
    const double LIDAR_HEIGHT = 2.0; // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.

    const int    PC_NUM_RING = 20; // 20 in the original paper (IROS 18)
    const int    PC_NUM_SECTOR = 60; // 60 in the original paper (IROS 18)
    const double PC_MAX_RADIUS = 80.0; // 80 meter max in the original paper (IROS 18)
    const double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    const double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    const int    NUM_EXCLUDE_RECENT = 30; // simply just keyframe gap (related with loopClosureFrequency in yaml), but node position distance-based exclusion is ok. 
    const int    NUM_CANDIDATES_FROM_TREE = 3; // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    const double SEARCH_RATIO = 0.1; // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.
    // const double SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)
    const double SC_DIST_THRES = 0.3; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15
    // const double SC_DIST_THRES = 0.7; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // config 
    const int    TREE_MAKING_PERIOD_ = 10; // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / in the LeGO-LOAM integration, it is synchronized with the loop detection callback (which is 1Hz) so it means the tree is updated evrey 10 sec. But you can use the smaller value because it is enough fast ~ 5-50ms wrt N.
    int          tree_making_period_conter = 0;

    // data 
    std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;

}; // SCManager

// } // namespace SC2