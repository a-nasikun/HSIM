#include "hsim/options.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace hsim {

void Options::validate() const
{
	if (!(levelRatio > 1.0)) throw std::invalid_argument("Options: levelRatio must be larger than 1");
	if (!(supportSize > 0.0)) throw std::invalid_argument("Options: supportSize must be positive");
	if (!(tolerance > 0.0)) throw std::invalid_argument("Options: tolerance must be positive");
	if (maxIterations < 1) throw std::invalid_argument("Options: maxIterations must be at least 1");
	if (!(shiftParameter >= 0.0 && shiftParameter < 1.0))
		throw std::invalid_argument("Options: shiftParameter must lie in [0, 1)");
	if (numThreads < 0) throw std::invalid_argument("Options: numThreads must be non-negative");
	if (!(areaFactor > 0.0)) throw std::invalid_argument("Options: areaFactor must be positive");
	if (denseBackend == DenseEigenBackend::CuSolver && !builtWithCuda())
		throw std::invalid_argument("Options: the CuSolver backend requires a build with HSIM_WITH_CUDA=ON");
}

int subspaceSize(int numModes)
{
	return std::max(2 * numModes, numModes + 8);
}

bool builtWithCuda()
{
#ifdef HSIM_WITH_CUDA
	return true;
#else
	return false;
#endif
}

LogSink defaultLogSink()
{
	return [](Verbosity, const std::string& message) { std::clog << message << '\n'; };
}

} // namespace hsim
