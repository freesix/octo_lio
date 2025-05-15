#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <Eigen/Core>
#include <glog/logging.h>

// 地图x方向与y方向的格子数
struct CellLimits {
    CellLimits() = default;
    CellLimits(int init_num_x_cells, int init_num_y_cells)
      : num_x_cells(init_num_x_cells), num_y_cells(init_num_y_cells) {}

    int num_x_cells = 0;
    int num_y_cells = 0;
};


class XYIndexRangeIterator
    : public std::iterator<std::input_iterator_tag, Eigen::Array2i> {
public:
    XYIndexRangeIterator(const Eigen::Array2i& min_xy_index,
                       const Eigen::Array2i& max_xy_index)
      : min_xy_index_(min_xy_index),
        max_xy_index_(max_xy_index),
        xy_index_(min_xy_index) {}

    explicit XYIndexRangeIterator(const CellLimits& cell_limits)
      : XYIndexRangeIterator(Eigen::Array2i::Zero(),
                             Eigen::Array2i(cell_limits.num_x_cells - 1,
                                            cell_limits.num_y_cells - 1)) {}

    XYIndexRangeIterator& operator++() {
        DCHECK(*this != end());
        if(xy_index_.x() < max_xy_index_.x()){
            ++xy_index_.x();
        } 
        else{
            xy_index_.x() = min_xy_index_.x();
            ++xy_index_.y();
        }
        return *this;
    }

    Eigen::Array2i& operator*(){ return xy_index_; }

    bool operator==(const XYIndexRangeIterator& other) const {
        return (xy_index_ == other.xy_index_).all();
    }

    bool operator!=(const XYIndexRangeIterator& other) const {
        return !operator==(other);
    }

    XYIndexRangeIterator begin() {
        return XYIndexRangeIterator(min_xy_index_, max_xy_index_);
    }

    XYIndexRangeIterator end() {
        XYIndexRangeIterator it = begin();
        it.xy_index_ = Eigen::Array2i(min_xy_index_.x(), max_xy_index_.y() + 1);
        return it;
    }

private:
    Eigen::Array2i min_xy_index_;
    Eigen::Array2i max_xy_index_;
    Eigen::Array2i xy_index_;
};
