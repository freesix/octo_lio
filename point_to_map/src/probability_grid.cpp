#include "point_to_map/probability_grid.hpp"

#include <limits>
#include <memory>
#include "point_to_map/probability_values.hpp"
// #include "submaps.hpp"

/**
 * @brief ProbabilityGrid的构造函数
 * 
 * @param[in] limits 地图坐标信息
 * @param[in] conversion_tables 转换表
 */
ProbabilityGrid::ProbabilityGrid(const MapLimits& limits,
                                ValueConversionTables* conversion_tables)
    : Grid2D(limits, kMinCorrespondenceCost, kMaxCorrespondenceCost, conversion_tables),
      conversion_tables_(conversion_tables){}

// 将 索引 处单元格的概率设置为给定的概率, 仅当单元格之前处于未知状态时才允许
void ProbabilityGrid::SetProbability(const Eigen::Array2i& cell_index,
                                     const float probability){
    // 获取对应栅格的引用
    u_int16_t& cell = (*mutable_correspondence_cost_cells())[ToFlatIndex(cell_index)];
    CHECK_EQ(cell, kUnknownProbabilityValue);
    // 为栅格赋值 value
    cell =
        CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(probability));
    // 更新bounding_box
    mutable_known_cells_box()->extend(cell_index.matrix());
}

// 如果单元格尚未更新,则将调用 ComputeLookupTableToApplyOdds() 时指定的 'odds' 应用于单元格在 'cell_index' 处的概率
// 在调用 FinishUpdate() 之前，将忽略同一单元格的多次更新。如果单元格已更新，则返回 true
// 如果这是对指定单元格第一次调用 ApplyOdds(),则其值将设置为与 'odds' 对应的概率
// 使用查找表对指定栅格进行栅格值的更新
bool ProbabilityGrid::ApplyLookupTable(const Eigen::Array2i& cell_index,
                                       const std::vector<u_int16_t>& table){
    DCHECK_EQ(table.size(), kUpdateMarker);
    const int flat_index = ToFlatIndex(cell_index);
    // 获取对应栅格的指针
    u_int16_t* cell = &(*mutable_correspondence_cost_cells())[flat_index];
    // 对处于更新状态的栅格, 不再进行更新了
    if(*cell >= kUpdateMarker){
        return false;
    }
    // 标记这个索引的栅格已经被更新过
    mutable_update_indices()->push_back(flat_index);
    // 更新栅格值
    *cell = table[*cell];
    DCHECK_GE(*cell, kUpdateMarker);
    // 更新bounding_box
    mutable_known_cells_box()->extend(cell_index.matrix());
    return true;
}


// 获取 索引 处单元格的占用概率
float ProbabilityGrid::GetProbability(const Eigen::Array2i& cell_index) const {
    if(!limits().Contains(cell_index)) return kMinProbability;
    return CorrespondenceCostToProbability(ValueToCorrespondenceCost(
        correspondence_cost_cells()[ToFlatIndex(cell_index)]));
}

// 根据bounding_box对栅格地图进行裁剪到正好包含点云
std::unique_ptr<Grid2D> ProbabilityGrid::ComputeCroppedGrid() const {
    Eigen::Array2i offset;
    CellLimits cell_limits;
    // 根据bounding_box对栅格地图进行裁剪
    ComputeCroppedLimits(&offset, &cell_limits);
    const double resolution = limits().resolution();
    // 重新计算最大值坐标
    const Eigen::Vector2d max =
        limits().max() - resolution * Eigen::Vector2d(offset.y(), offset.x());
    // 重新定义概率栅格地图的大小
    std::unique_ptr<ProbabilityGrid> cropped_grid =
        std::make_unique<ProbabilityGrid>(
            MapLimits(resolution, max, cell_limits), conversion_tables_);
    // 给新栅格地图赋值
    for(const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)){
        if(!IsKnown(xy_index + offset)){
            continue;
        }
        cropped_grid->SetProbability(xy_index, GetProbability(xy_index + offset));
    }
    // 返回新地图的指针
    return std::unique_ptr<Grid2D>(cropped_grid.release());
}

