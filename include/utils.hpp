#pragma once
#include <chrono>

namespace fem1d {

// Type alias for clarity
using TimePoint = std::chrono::high_resolution_clock::time_point;

// Get current time
inline TimePoint now() {
    return std::chrono::high_resolution_clock::now();
}

// Seconds between two time points
inline double elapsed_seconds(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace fem1d
