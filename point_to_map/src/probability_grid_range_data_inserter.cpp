#include "point_to_map/probability_grid_range_data_inserter.hpp"

#include <cstdlib>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>

namespace {

// Factor for subpixel accuracy of start and end point for ray casts.
constexpr int kSubpixelScale = 1000;

bool isEqual(const Eigen::Array2i& lhs, const Eigen::Array2i& rhs) {
    return ((lhs - rhs).matrix().lpNorm<1>() == 0);
}

// 根据点云的bounding box, 看是否需要对地图进行扩张
void GrowAsNeeded(const RangeData& range_data,
                  ProbabilityGrid* const probability_grid) {
    // 找到点云的bounding_box
    Eigen::AlignedBox2f bounding_box(range_data.origin.head<2>());
    constexpr float kPadding = 1e-6f; // 为了防止数值错误
    // 循环遍历hit点，找出小bounding_box
    for(const Eigen::Vector3f &hit : range_data.returns){
        bounding_box.extend(hit.head<2>());
    }
    // 同样对miss点
    for (const Eigen::Vector3f &miss : range_data.misses) {
        bounding_box.extend(miss.head<2>());
    }
    // 是否对地图进行扩张
    probability_grid->GrowLimits(bounding_box.min() -
                               kPadding * Eigen::Vector2f::Ones());
    probability_grid->GrowLimits(bounding_box.max() +
                               kPadding * Eigen::Vector2f::Ones());
}

/**
 * @brief 根据雷达点对栅格地图进行更新
 * 
 * @param[in] range_data 
 * @param[in] hit_table 更新占用栅格时的查找表
 * @param[in] miss_table 更新空闲栅格时的查找表
 * @param[in] insert_free_space 
 * @param[in] probability_grid 栅格地图
 */
void CastRays(const RangeData &range_data,
              const std::vector<u_int16_t> &hit_table,
              const std::vector<u_int16_t> &miss_table,
              const bool insert_free_space, ProbabilityGrid* probability_grid) {
    // 根据雷达数据调整地图范围
    GrowAsNeeded(range_data, probability_grid);

    const MapLimits& limits = probability_grid->limits(); // 获取概率地图大小
    const double superscaled_resolution = limits.resolution() / kSubpixelScale; // 超分辨率
    const MapLimits superscaled_limits( // 根据超分辨率构造新的地图限制对象
        superscaled_resolution, limits.max(),
        CellLimits(limits.cell_limits().num_x_cells * kSubpixelScale,
                    limits.cell_limits().num_y_cells * kSubpixelScale));
    // 雷达原点在地图中的像素坐标, 作为画线的起始坐标
    const Eigen::Array2i begin =
        superscaled_limits.GetCellIndex(range_data.origin.head<2>());
    // 计算并存储射线终点坐标
    std::vector<Eigen::Array2i> ends;
    ends.reserve(range_data.returns.size());
    for(const Eigen::Vector3f &hit : range_data.returns) {
        // 计算hit点在地图中的像素坐标, 作为画线的终止点坐标
        ends.push_back(superscaled_limits.GetCellIndex(hit.head<2>()));
        // 更新hit点的栅格值
        probability_grid->ApplyLookupTable(ends.back() / kSubpixelScale, hit_table);
    }
  
    // 如果不插入free空间就可以结束了
    if(!insert_free_space){
        return;
    }

    // Now add the misses.
    for(const Eigen::Array2i &end : ends){
        std::vector<Eigen::Array2i> ray = RayToPixelMask(begin, end, kSubpixelScale);
        for(const Eigen::Array2i &cell_index : ray){
            // 从起点到end点之前, 更新miss点的栅格值
            probability_grid->ApplyLookupTable(cell_index, miss_table);
        }
    }

    // Finally, compute and add empty rays based on misses in the range data.
    for(const Eigen::Vector3f &missing_echo : range_data.misses){
        std::vector<Eigen::Array2i> ray = RayToPixelMask(
            begin, superscaled_limits.GetCellIndex(missing_echo.head<2>()),
            kSubpixelScale);
        for (const Eigen::Array2i& cell_index : ray) {
            // 从起点到misses点之前, 更新miss点的栅格值
            probability_grid->ApplyLookupTable(cell_index, miss_table);
        }
    }
}
}  // namespace

std::vector<Eigen::Array2i> RayToPixelMask(const Eigen::Array2i& scaled_begin,
                                           const Eigen::Array2i& scaled_end,
                                           int subpixel_scale) {
    // 保持起始点的x小于终止点的x
    if(scaled_begin.x() > scaled_end.x()){
        return RayToPixelMask(scaled_end, scaled_begin, subpixel_scale);
    }

    CHECK_GE(scaled_begin.x(), 0);
    CHECK_GE(scaled_begin.y(), 0);
    CHECK_GE(scaled_end.y(), 0);
    std::vector<Eigen::Array2i> pixel_mask;
    if(scaled_begin.x() / subpixel_scale == scaled_end.x() / subpixel_scale){
        // y坐标小的位置是起始点
        Eigen::Array2i current(scaled_begin.x() / subpixel_scale,
                    std::min(scaled_begin.y(), scaled_end.y()) / subpixel_scale);
        pixel_mask.push_back(current);
        // y坐标大的位置是终止点
        const int end_y =
                    std::max(scaled_begin.y(), scaled_end.y()) / subpixel_scale;
        // y坐标进行+1的操作
        for(; current.y() <= end_y; ++current.y()){
            if (!isEqual(pixel_mask.back(), current)) pixel_mask.push_back(current);
        }
        return pixel_mask;
    }
    // 下边就是 breshman 的具体实现
    const int64_t dx = scaled_end.x() - scaled_begin.x();
    const int64_t dy = scaled_end.y() - scaled_begin.y();
    const int64_t denominator = 2 * subpixel_scale * dx;

    Eigen::Array2i current = scaled_begin / subpixel_scale;
    pixel_mask.push_back(current);

    // To represent subpixel centers, we use a factor of 2 * 'subpixel_scale' in
    // the denominator.
    // +-+-+-+ -- 1 = (2 * subpixel_scale) / (2 * subpixel_scale)
    // | | | |
    // +-+-+-+
    // | | | |
    // +-+-+-+ -- top edge of first subpixel = 2 / (2 * subpixel_scale)
    // | | | | -- center of first subpixel = 1 / (2 * subpixel_scale)
    // +-+-+-+ -- 0 = 0 / (2 * subpixel_scale)

    int64_t sub_y = (2 * (scaled_begin.y() % subpixel_scale) + 1) * dx;

    const int first_pixel =
        2 * subpixel_scale - 2 * (scaled_begin.x() % subpixel_scale) - 1;
    // The same from the left pixel border to 'scaled_end'.
    const int last_pixel = 2 * (scaled_end.x() % subpixel_scale) + 1;

    // The full pixel x coordinate of 'scaled_end'.
    const int end_x = std::max(scaled_begin.x(), scaled_end.x()) / subpixel_scale;

    // Move from 'scaled_begin' to the next pixel border to the right.
    // dy > 0 时的情况
    sub_y += dy * first_pixel;
    if(dy > 0){
        while(true){
            if(!isEqual(pixel_mask.back(), current)){
                pixel_mask.push_back(current);
            }
            while(sub_y > denominator){
                sub_y -= denominator;
                ++current.y();
                if(!isEqual(pixel_mask.back(), current)){
                    pixel_mask.push_back(current);
                }
            }
            ++current.x();
            if(sub_y == denominator){
                sub_y -= denominator;
                ++current.y();
            }
            if(current.x() == end_x){
                break;
            }
            // Move from one pixel border to the next.
            sub_y += dy * 2 * subpixel_scale;
        }
        // Move from the pixel border on the right to 'scaled_end'.
        sub_y += dy * last_pixel;
        if(!isEqual(pixel_mask.back(), current)){
            pixel_mask.push_back(current);
        }
        while(sub_y > denominator){
            sub_y -= denominator;
            ++current.y();
            if(!isEqual(pixel_mask.back(), current)){
                pixel_mask.push_back(current);
            }
        }
        CHECK_NE(sub_y, denominator);
        CHECK_EQ(current.y(), scaled_end.y() / subpixel_scale);
        return pixel_mask;
    }

    // dy <= 0 时的情况
    // Same for lines non-ascending in y coordinates.
    while(true){
        if(!isEqual(pixel_mask.back(), current)){
            pixel_mask.push_back(current);
        }
        while(sub_y < 0){
            sub_y += denominator;
            --current.y();
            if(!isEqual(pixel_mask.back(), current)){
                pixel_mask.push_back(current);
            }
        }
        ++current.x();
        if(sub_y == 0){
            sub_y += denominator;
            --current.y();
        }
        if(current.x() == end_x){
            break;
        }
        sub_y += dy * 2 * subpixel_scale;
    }
    sub_y += dy * last_pixel;
    if(!isEqual(pixel_mask.back(), current)){
        pixel_mask.push_back(current);
    }
    while(sub_y < 0){
        sub_y += denominator;
        --current.y();
        if(!isEqual(pixel_mask.back(), current)){
            pixel_mask.push_back(current);
        }
    }
    CHECK_NE(sub_y, 0);
    CHECK_EQ(current.y(), scaled_end.y() / subpixel_scale);
    return pixel_mask;
}

// 写入器的构造, 新建了2个查找表
ProbabilityGridRangeDataInserter2D::ProbabilityGridRangeDataInserter2D(
    double hit_probability, double miss_probability) :
    // 生成更新占用栅格时的查找表 // param: hit_probability
    hit_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(hit_probability))),    // 0.55
    // 生成更新空闲栅格时的查找表 // param: miss_probability
    miss_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(miss_probability))) {} // 0.49

/**
 * @brief 将点云写入栅格地图
 * 
 * @param[in] range_data 要写入地图的点云
 * @param[in] grid 栅格地图
 */
void ProbabilityGridRangeDataInserter2D::Insert(
            RangeData &range_data, Grid2D* const grid) const {
    ProbabilityGrid* const probability_grid = static_cast<ProbabilityGrid*>(grid);
    CHECK(probability_grid != nullptr);
    CastRays(range_data, hit_table_, miss_table_, true,
            probability_grid);
    probability_grid->FinishUpdate();
}
