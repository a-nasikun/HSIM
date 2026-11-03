// Internal helpers: formatted logging according to hsim::Options.
#pragma once

#include "hsim/options.h"

#include <sstream>
#include <string>
#include <utility>

namespace hsim::detail {

class Logger {
public:
	explicit Logger(const Options& options)
	    : verbosity_(options.verbosity), sink_(options.logSink ? options.logSink : defaultLogSink())
	{
	}

	bool enabled(Verbosity level) const
	{
		return level != Verbosity::Silent && static_cast<int>(level) <= static_cast<int>(verbosity_);
	}

	template <typename... Args> void operator()(Verbosity level, Args&&... args) const
	{
		if (!enabled(level)) return;
		std::ostringstream message;
		(message << ... << std::forward<Args>(args));
		sink_(level, message.str());
	}

private:
	Verbosity verbosity_;
	LogSink sink_;
};

} // namespace hsim::detail
