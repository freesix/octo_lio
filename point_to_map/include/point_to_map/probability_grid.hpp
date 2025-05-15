#pragma once

#include <vector>

#include "grid_2d.hpp"
#include "map_limits.hpp"
#include "xy_index.hpp"

class ProbabilityGrid : public Grid2D {
public:
    explicit ProbabilityGrid(const MapLimits& limits,
                           ValueConversionTables* conversion_tables);

    void SetProbability(const Eigen::Array2i& cell_index,
                        const float probability);

    bool ApplyLookupTable(const Eigen::Array2i& cell_index,
                        const std::vector<u_int16_t>& table);


    float GetProbability(const Eigen::Array2i& cell_index) const;

    std::unique_ptr<Grid2D> ComputeCroppedGrid() const override;
    
private:
    ValueConversionTables* conversion_tables_;
};