#include "point_to_map/grid_2d.hpp"

/**
 * @brief 构造函数
 * 
 * @param[in] limits 地图坐标信息
 * @param[in] min_correspondence_cost 最小correspondence_cost 0.1
 * @param[in] max_correspondence_cost 最大correspondence_cost 0.9
 * @param[in] conversion_tables 传入的转换表指针
 */
Grid2D::Grid2D(const MapLimits& limits, float min_correspondence_cost,
               float max_correspondence_cost,
               ValueConversionTables* conversion_tables)
    : limits_(limits),
      correspondence_cost_cells_(
          limits_.cell_limits().num_x_cells * limits_.cell_limits().num_y_cells,
          kUnknownCorrespondenceValue),  // 0
      min_correspondence_cost_(min_correspondence_cost),  // 0.1
      max_correspondence_cost_(max_correspondence_cost),  // 0.9
      // 新建转换表
      value_to_correspondence_cost_table_(conversion_tables->GetConversionTable(
          max_correspondence_cost, min_correspondence_cost,
          max_correspondence_cost)) 
{
    CHECK_LT(min_correspondence_cost_, max_correspondence_cost_);
}

// 插入雷达数据结束
void Grid2D::FinishUpdate() {
    while(!update_indices_.empty()){
        DCHECK_GE(correspondence_cost_cells_[update_indices_.back()], kUpdateMarker);
        // 更新的时候加上了kUpdateMarker, 在这里减去
        correspondence_cost_cells_[update_indices_.back()] -= kUpdateMarker;
        update_indices_.pop_back();
    }
}

// 根据known_cells_box_更新limits
void Grid2D::ComputeCroppedLimits(Eigen::Array2i* const offset,
                                  CellLimits* const limits) const {
    if(known_cells_box_.isEmpty()) {
        *offset = Eigen::Array2i::Zero();
        *limits = CellLimits(1, 1);
        return;
    }
    *offset = known_cells_box_.min().array();
    *limits = CellLimits(known_cells_box_.sizes().x() + 1,
                       known_cells_box_.sizes().y() + 1);
}

// 根据坐标决定是否对地图进行扩大
void Grid2D::GrowLimits(const Eigen::Vector2f& point) {
    GrowLimits(point, {mutable_correspondence_cost_cells()},
        {kUnknownCorrespondenceValue});
}

// 根据坐标决定是否对地图进行扩大
void Grid2D::GrowLimits(const Eigen::Vector2f& point,
                        const std::vector<std::vector<u_int16_t>*>& grids,
                        const std::vector<u_int16_t>& grids_unknown_cell_values) {
    CHECK(update_indices_.empty());
    // 判断该点是否在地图坐标系内
    while(!limits_.Contains(limits_.GetCellIndex(point))){
        const int x_offset = limits_.cell_limits().num_x_cells / 2;
        const int y_offset = limits_.cell_limits().num_y_cells / 2;
        // 将xy扩大至2倍, 中心点不变, 向四周扩大
        const MapLimits new_limits(limits_.resolution(),
            limits_.max() + limits_.resolution() * Eigen::Vector2d(y_offset, x_offset),
            CellLimits(2 * limits_.cell_limits().num_x_cells, 2 * limits_.cell_limits().num_y_cells));
        const int stride = new_limits.cell_limits().num_x_cells;
        // 老坐标系的原点在新坐标系下的一维像素坐标
        const int offset = x_offset + stride * y_offset;
        const int new_size = new_limits.cell_limits().num_x_cells *
                         new_limits.cell_limits().num_y_cells;

        // grids.size()为1
        for(size_t grid_index = 0; grid_index < grids.size(); ++grid_index){
            std::vector<u_int16_t> new_cells(new_size, grids_unknown_cell_values[grid_index]);
            // 将老地图的栅格值复制到新地图上
            for(int i = 0; i < limits_.cell_limits().num_y_cells; ++i){
                for (int j = 0; j < limits_.cell_limits().num_x_cells; ++j){
                    new_cells[offset + j + i * stride] =
                    (*grids[grid_index])[j + i * limits_.cell_limits().num_x_cells];
                }
            }
            // 将新地图替换老地图, 拷贝
        *grids[grid_index] = new_cells;
        } // end for
        // 更新地图尺寸
        limits_ = new_limits;
        if(!known_cells_box_.isEmpty()){
        // 将known_cells_box_的x与y进行平移到老地图的范围上
        known_cells_box_.translate(Eigen::Vector2i(x_offset, y_offset));
        }
    }
}
