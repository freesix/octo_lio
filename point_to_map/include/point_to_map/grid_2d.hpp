#pragma once

#include <vector>

#include "map_limits.hpp"
#include "probability_values.hpp"
#include "value_conversion_tables.hpp"


class Grid2D{
public:
    Grid2D(const MapLimits& limits, float min_correspondence_cost,
         float max_correspondence_cost,
         ValueConversionTables* conversion_tables);

    // Returns the limits of this Grid2D.
    const MapLimits& limits() const { return limits_; }

    // Finishes the update sequence.
    void FinishUpdate();

    // Returns the correspondence cost of the cell with 'cell_index'.
    float GetCorrespondenceCost(const Eigen::Array2i& cell_index) const {
        if(!limits().Contains(cell_index)){
            return max_correspondence_cost_;
        }
        return(*value_to_correspondence_cost_table_)[correspondence_cost_cells()[ToFlatIndex(cell_index)]];
    }

    float GetMinCorrespondenceCost() const { return min_correspondence_cost_; }

    float GetMaxCorrespondenceCost() const { return max_correspondence_cost_; }

    // 指定的栅格是否被更新过
    bool IsKnown(const Eigen::Array2i& cell_index) const {
        return limits_.Contains(cell_index) &&
           correspondence_cost_cells_[ToFlatIndex(cell_index)] != kUnknownCorrespondenceValue;
    }

    void ComputeCroppedLimits(Eigen::Array2i* const offset,
                            CellLimits* const limits) const;

    virtual void GrowLimits(const Eigen::Vector2f& point);

    virtual std::unique_ptr<Grid2D> ComputeCroppedGrid() const = 0;

 
// protected:
    void GrowLimits(const Eigen::Vector2f& point,
                  const std::vector<std::vector<u_int16_t>*>& grids,
                  const std::vector<u_int16_t>& grids_unknown_cell_values);

    // 返回不可以修改的栅格地图数组的引用
    const std::vector<u_int16_t>& correspondence_cost_cells() const {
        return correspondence_cost_cells_;
    }
    const std::vector<int>& update_indices() const { return update_indices_; }
    const Eigen::AlignedBox2i& known_cells_box() const {
        return known_cells_box_;
    }

    // 返回可以修改的栅格地图数组的指针
    std::vector<u_int16_t>* mutable_correspondence_cost_cells() {
        return &correspondence_cost_cells_;
    }

    std::vector<int>* mutable_update_indices() { return &update_indices_; }
    Eigen::AlignedBox2i* mutable_known_cells_box() { return &known_cells_box_; }

    // Converts a 'cell_index' into an index into 'cells_'.
    // 二维像素坐标转为一维索引坐标
    int ToFlatIndex(const Eigen::Array2i& cell_index) const {
        CHECK(limits_.Contains(cell_index)) << cell_index;
        return limits_.cell_limits().num_x_cells * cell_index.y() + cell_index.x();
    }

// private:
    MapLimits limits_;  // 地图大小边界, 包括x和y最大值, 分辨率, x和y方向栅格数

    // 地图栅格值, 存储的是free的概率转成u_int16_t后的[0, 32767]范围内的值, 0代表未知
    std::vector<u_int16_t> correspondence_cost_cells_; 
    float min_correspondence_cost_;
    float max_correspondence_cost_;
    std::vector<int> update_indices_;               // 记录已经更新过的索引

    Eigen::AlignedBox2i known_cells_box_;           // 栅格的bounding box, 存的是像素坐标
    // 将[0, 1~32767] 映射到 [0.9, 0.1~0.9] 的转换表
    const std::vector<float>* value_to_correspondence_cost_table_;
};

