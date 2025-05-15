#include "point_to_map/value_conversion_tables.hpp"

namespace {

constexpr u_int16_t kUpdateMarker = 1u << 15;

/**
 * @brief 将[0, 1~32767] 映射到 [0.9, 0.1~0.9]
 * 
 * @param[in] value [0, 32767]的值, 0 对应0.9
 * @param[in] unknown_value 0
 * @param[in] unknown_result 0.9 
 * @param[in] lower_bound 0.1 下界
 * @param[in] upper_bound 0.9 上界
 * @return float 转换后的数值
 */
float SlowValueToBoundedFloat(const u_int16_t value, 
                              const u_int16_t unknown_value,
                              const float unknown_result,
                              const float lower_bound,
                              const float upper_bound) {
    CHECK_LE(value, 32767);
    if (value == unknown_value) return unknown_result;
    const float kScale = (upper_bound - lower_bound) / 32766.f;
    return value * kScale + (lower_bound - kScale);
}

/**
 * @brief 新建转换表
 * 
 * @param[in] unknown_value 0 
 * @param[in] unknown_result 0.9 
 * @param[in] lower_bound 0.1 
 * @param[in] upper_bound 0.9 
 * @return std::unique_ptr<std::vector<float>> 转换表的指针
 */
std::unique_ptr<std::vector<float>> PrecomputeValueToBoundedFloat(
        const u_int16_t unknown_value, const float unknown_result,
        const float lower_bound, const float upper_bound) {
    auto result = std::make_unique<std::vector<float>>();
    // num_values = 65536
    size_t num_values = std::numeric_limits<u_int16_t>::max() + 1; 
    // 申请空间
    result->reserve(num_values);

    // 将[0, 1~32767]映射成[0.9, 0.1~0.9]
    // vector的个数为65536, 所以存的是2遍[0-32767]的映射
    for (size_t value = 0; value != num_values; ++value) {
        result->push_back(SlowValueToBoundedFloat(
        static_cast<u_int16_t>(value) & ~kUpdateMarker, // 取右边15位的数据, 0-32767
        unknown_value,
        unknown_result, lower_bound, upper_bound));
    }
    return result;
}
}
/**
 * @brief 获取转换表, 这个函数只会调用1次
 * 
 * @param[in] unknown_result 0.9 未知时的值
 * @param[in] lower_bound 0.1 最小correspondence_cost
 * @param[in] upper_bound 0.9 最大correspondence_cost
 * @return const std::vector<float>* 
 */
const std::vector<float>* ValueConversionTables::GetConversionTable(
            float unknown_result, float lower_bound, float upper_bound) {
    // 将bounds作为key
    std::tuple<float, float, float> bounds =
        std::make_tuple(unknown_result, lower_bound, upper_bound);
    // auto lookup_table_iterator = bounds_to_lookup_table_.find(bounds);

    // 如果没有bounds这个key就新建
    // if (bounds_to_lookup_table_.empty()) {
        // 新建转换表
        // auto insertion_result = bounds_to_lookup_table_.emplace(
            // bounds, PrecomputeValueToBoundedFloat(0, unknown_result, lower_bound,
                                            //   upper_bound));
        // return insertion_result.first->second.get();
        // return PrecomputeValueToBoundedFloat(0, unknown_result, lower_bound, upper_bound).release();
        return std::move(PrecomputeValueToBoundedFloat(0, unknown_result, lower_bound, upper_bound).release());
    // } 
    // 如果存在就直接返回原始指针
    // else {
        // return lookup_table_iterator->second.get();
        // std::cout<<"some is ..."<<std::endl;
    // }
}
