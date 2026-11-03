/**
 * @file log.h
 * @brief Run-time configurable progress output.
 *
 * The library never writes to the console unless asked to: the amount of output is selected with
 * hsim::Options::verbosity and the destination with hsim::Options::logSink.
 */
#pragma once

#include <functional>
#include <string>

namespace hsim {

/// Amount of progress output.
enum class Verbosity {
	Silent = 0, ///< no output (default)
	Summary = 1, ///< one line per stage of the algorithm
	Detailed = 2, ///< additionally one line per level and iteration
};

/// Receives every message whose level is enabled by hsim::Options::verbosity.
using LogSink = std::function<void(Verbosity level, const std::string& message)>;

/// Sink that writes each message as one line to std::clog.
[[nodiscard]] LogSink defaultLogSink();

} // namespace hsim
