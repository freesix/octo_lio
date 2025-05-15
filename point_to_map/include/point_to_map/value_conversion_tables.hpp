#pragma once

#include <map>
#include <vector>
#include <memory>
#include "glog/logging.h"
#include <limits>
#include <iostream>


// 以将 uint16 值映射到 ['lower_bound', 'upper_bound'] 中的浮点数
// 表的第一个元素设置为 unknown_result 
class ValueConversionTables {
public:
      const std::vector<float>* GetConversionTable(float unknown_result,
                                               float lower_bound,
                                               float upper_bound);

private:
      std::map<const std::tuple<float /* unknown_result */, float /* lower_bound */,
                            float /* upper_bound */>,
           std::unique_ptr<const std::vector<float>>>
      bounds_to_lookup_table_;
};
