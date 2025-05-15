#pragma once

#include <utility>
#include <vector>
#include <Eigen/Core>
#include "probability_grid.hpp"
#include "xy_index.hpp"

struct RangeData{
    Eigen::Vector3f origin;
    std::vector<Eigen::Vector3f> returns;
    std::vector<Eigen::Vector3f> misses;
};


std::vector<Eigen::Array2i> RayToPixelMask(const Eigen::Array2i& scaled_begin,
                                           const Eigen::Array2i& scaled_end,
                                           int subpixel_scale);

class ProbabilityGridRangeDataInserter2D{
public:
    ProbabilityGridRangeDataInserter2D(
        double hit_probability, double miss_probability);
    // ProbabilityGridRangeDataInserter2D(
        // const ProbabilityGridRangeDataInserter2D&) = delete;
    ProbabilityGridRangeDataInserter2D& operator=(
        const ProbabilityGridRangeDataInserter2D&) = delete;

    // Inserts 'range_data' into 'probability_grid'.
    void Insert(RangeData& range_data, Grid2D* const grid) const;

private:
    const std::vector<u_int16_t> hit_table_;
    const std::vector<u_int16_t> miss_table_;
};

