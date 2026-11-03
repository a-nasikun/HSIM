// Internal helper: wall-clock stopwatch.
#pragma once

#include <chrono>

namespace hsim::detail {

class Timer {
public:
	Timer() : start_(std::chrono::steady_clock::now()) {}
	double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count(); }

private:
	std::chrono::steady_clock::time_point start_;
};

} // namespace hsim::detail
