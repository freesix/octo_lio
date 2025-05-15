#pragma once

#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include "xy_index.hpp"

/**
 * note: 地图坐标系可视化展示
 * ros的地图坐标系    cartographer的地图坐标系     cartographer地图的像素坐标系 
 * 
 * ^ y                            ^ x              0------> x
 * |                              |                |
 * |                              |                |
 * 0 ------> x           y <------0                y       
 * 
 * ros的地图坐标系: 左下角为原点, 向右为x正方向, 向上为y正方向, 角度以x轴正向为0度, 逆时针为正
 * cartographer的地图坐标系: 坐标系右下角为原点, 向上为x正方向, 向左为y正方向
 *             角度正方向以x轴正向为0度, 逆时针为正
 * cartographer地图的像素坐标系: 左上角为原点, 向右为x正方向, 向下为y正方向
 */
class MapLimits {
public:
  /**
   * @brief 构造函数
   * 
   * @param[in] resolution 地图分辨率
   * @param[in] max 左上角的坐标为地图坐标的最大值
   * @param[in] cell_limits 地图x方向与y方向的格子数
   */
    MapLimits(const double resolution, const Eigen::Vector2d& max,
            const CellLimits& cell_limits)
      : resolution_(resolution), max_(max), cell_limits_(cell_limits){
        CHECK_GT(resolution_, 0.);
        CHECK_GT(cell_limits.num_x_cells, 0.);
        CHECK_GT(cell_limits.num_y_cells, 0.);
    }
 
    double resolution() const { return resolution_; }

    // 返回左上角坐标, 左上角坐标为整个坐标系坐标的最大值
    const Eigen::Vector2d& max() const { return max_; }

    // Returns the limits of the grid in number of cells.
    const CellLimits& cell_limits() const { return cell_limits_; }

    // 计算物理坐标点的像素索引
    Eigen::Array2i GetCellIndex(const Eigen::Vector2f& point) const {
        return Eigen::Array2i(
            std::lround((max_.y() - point.y()) / resolution_ - 0.5),
            std::lround((max_.x() - point.x()) / resolution_ - 0.5));
    }

    // 根据像素索引算物理坐标
    Eigen::Vector2f GetCellCenter(const Eigen::Array2i cell_index) const {
        return {max_.x() - resolution() * (cell_index[1] + 0.5),
            max_.y() - resolution() * (cell_index[0] + 0.5)};
    }

    // 判断给定像素索引是否在栅格地图内部
    bool Contains(const Eigen::Array2i& cell_index) const {
        return (Eigen::Array2i(0, 0) <= cell_index).all() &&
           (cell_index < Eigen::Array2i(cell_limits_.num_x_cells, cell_limits_.num_y_cells)).all();
    }

private:
    double resolution_;
    Eigen::Vector2d max_;    // cartographer地图坐标系左上角为坐标系的坐标的最大值
    CellLimits cell_limits_;
};
