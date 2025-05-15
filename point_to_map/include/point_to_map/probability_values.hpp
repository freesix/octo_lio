#pragma once

#include <cmath>
#include <vector>

#include "glog/logging.h"

inline float Clamp(float value, float lower, float upper){
    if(value > upper){
        return upper; 
    }
    if(value < lower){
        return lower; 
    }
    return value;
}

namespace{
// 为了避免浮点运算, 将[0~0.9]的浮点数转成[1~32767]之间的值
inline u_int16_t BoundedFloatToValue(const float float_value,
                                  const float lower_bound,
                                  const float upper_bound) {
    const int value =
        std::lround((Clamp(float_value, lower_bound, upper_bound) - lower_bound) *
                    (32766.f / (upper_bound - lower_bound))) + 1;
    DCHECK_GE(value, 1);
    DCHECK_LE(value, 32767);
    return value;
}
}

inline float Odds(float probability) {
    return probability / (1.f - probability);
}

inline float ProbabilityFromOdds(const float odds) {
    return odds / (odds + 1.f);
}

inline float ProbabilityToCorrespondenceCost(const float probability) {
    return 1.f - probability;
}

inline float CorrespondenceCostToProbability(const float correspondence_cost) {
    return 1.f - correspondence_cost;
}

constexpr float kMinProbability = 0.1f;                         // 0.1
constexpr float kMaxProbability = 1.f - kMinProbability;        // 0.9
constexpr float kMinCorrespondenceCost = 1.f - kMaxProbability; // 0.1
constexpr float kMaxCorrespondenceCost = 1.f - kMinProbability; // 0.9

inline float ClampProbability(const float probability) {
    return Clamp(probability, kMinProbability, kMaxProbability);
}

inline float ClampCorrespondenceCost(const float correspondence_cost) {
    return Clamp(correspondence_cost, kMinCorrespondenceCost, kMaxCorrespondenceCost);
}

constexpr u_int16_t kUnknownProbabilityValue = 0; // 0
constexpr u_int16_t kUnknownCorrespondenceValue = kUnknownProbabilityValue; // 0
constexpr u_int16_t kUpdateMarker = 1u << 15; // 32768

// 将浮点数correspondence_cost转成[1, 32767]范围内的 u_int16_t 整数
inline u_int16_t CorrespondenceCostToValue(const float correspondence_cost) {
    return BoundedFloatToValue(correspondence_cost, kMinCorrespondenceCost, kMaxCorrespondenceCost);
}

// 将浮点数probability转成[1, 32767]范围内的 u_int16_t 整数
inline u_int16_t ProbabilityToValue(const float probability) {
    return BoundedFloatToValue(probability, kMinProbability, kMaxProbability);
}

// c++11: extern c风格
extern const std::vector<float>* const kValueToProbability;
extern const std::vector<float>* const kValueToCorrespondenceCost;

inline float ValueToProbability(const u_int16_t value) {
    return (*kValueToProbability)[value];
}

inline float ValueToCorrespondenceCost(const u_int16_t value) {
  return (*kValueToCorrespondenceCost)[value];
}

// 测试用的函数
inline u_int16_t ProbabilityValueToCorrespondenceCostValue(u_int16_t probability_value) {
    if (probability_value == kUnknownProbabilityValue) {
        return kUnknownCorrespondenceValue;
    }
    bool update_carry = false;
    if (probability_value > kUpdateMarker) {
        probability_value -= kUpdateMarker;
        update_carry = true;
    }
    u_int16_t result = CorrespondenceCostToValue(
        ProbabilityToCorrespondenceCost(ValueToProbability(probability_value)));
    if(update_carry) result += kUpdateMarker;
    return result;
}

// 测试用的函数
inline u_int16_t CorrespondenceCostValueToProbabilityValue(
    u_int16_t correspondence_cost_value) {
    if(correspondence_cost_value == kUnknownCorrespondenceValue){
        return kUnknownProbabilityValue;
    }
    bool update_carry = false;
    if(correspondence_cost_value > kUpdateMarker) {
        correspondence_cost_value -= kUpdateMarker;
        update_carry = true;
    }
    u_int16_t result = ProbabilityToValue(CorrespondenceCostToProbability(
        ValueToCorrespondenceCost(correspondence_cost_value)));
    if(update_carry){
        result += kUpdateMarker;
    }
    return result;
}

std::vector<u_int16_t> ComputeLookupTableToApplyOdds(float odds);
std::vector<u_int16_t> ComputeLookupTableToApplyCorrespondenceCostOdds(float odds);

