#include "hsim/sampling.h"

#include "hsim/mesh.h"
#include "hsim/options.h"

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>

namespace hsim {

std::vector<int> samplingLevelSizes(Eigen::Index numVertices, int numModes, double levelRatio)
{
	if (numModes < 1) throw std::invalid_argument("samplingLevelSizes: the number of modes must be positive");
	if (!(levelRatio > 1.0)) throw std::invalid_argument("samplingLevelSizes: the level ratio must be larger than 1");

	// The coarsest level must hold the q iterated vectors; levels are added until one has at most 2 q samples.
	const int minimalSize = subspaceSize(numModes);
	const int stopSize = 2 * minimalSize;

	std::vector<int> sizes;
	int numSamples = static_cast<int>(numVertices);
	do {
		numSamples = static_cast<int>(std::round(numSamples / levelRatio));
		if (numSamples < minimalSize) break;
		sizes.push_back(numSamples);
	} while (numSamples > stopSize);

	if (sizes.empty()) {
		throw std::invalid_argument("samplingLevelSizes: the mesh (" + std::to_string(numVertices) + " vertices) is too coarse to compute "
			+ std::to_string(numModes) + " eigenpairs with level ratio " + std::to_string(levelRatio) + "; at least "
			+ std::to_string(static_cast<long>(std::ceil(levelRatio * minimalSize))) + " vertices are needed");
	}
	return sizes;
}

namespace {
	struct QueueEntry {
		int vertex;
		double distance;
		bool operator>(const QueueEntry& other) const { return distance > other.distance; }
	};
	using MinQueue = std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>>;
}

void accumulateGraphDistance(const Eigen::MatrixXd& V, const Adjacency& adjacency, int source, Eigen::VectorXd& distance)
{
	MinQueue queue;
	distance(source) = 0.0;
	queue.push({ source, 0.0 });

	while (!queue.empty()) {
		const QueueEntry top = queue.top();
		queue.pop();
		for (const int neighbor : adjacency.neighbors(top.vertex)) {
			const Eigen::VectorXd a = V.block<1, 3>(top.vertex, 0);
			const Eigen::VectorXd b = V.block<1, 3>(neighbor, 0);
			const double candidate = top.distance + (b - a).norm();
			if (candidate < distance(neighbor)) {
				distance(neighbor) = candidate;
				queue.push({ neighbor, candidate });
			}
		}
	}
}

Samples farthestPointSampling(const Eigen::MatrixXd& V, const Adjacency& adjacency, const std::vector<int>& levelSizes)
{
	if (levelSizes.empty()) throw std::invalid_argument("farthestPointSampling: no levels");
	if (levelSizes.front() > V.rows()) throw std::invalid_argument("farthestPointSampling: more samples than vertices");

	const int numLevels = static_cast<int>(levelSizes.size());
	Samples result;
	result.vertices.resize(levelSizes.front());
	result.distance.resize(static_cast<size_t>(numLevels));

	Eigen::VectorXd D = Eigen::VectorXd::Constant(V.rows(), std::numeric_limits<double>::infinity());
	result.vertices(0) = 0;

	// Coarsest level first; the samples of a finer level extend those of the coarser one.
	int first = 1;
	for (int level = numLevels - 1; level >= 0; --level) {
		for (int i = first; i < levelSizes[static_cast<size_t>(level)]; ++i) {
			accumulateGraphDistance(V, adjacency, result.vertices(i - 1), D);
			Eigen::Index farthest = 0;
			D.maxCoeff(&farthest);
			result.vertices(i) = static_cast<int>(farthest);
		}
		result.distance[static_cast<size_t>(level)] = D;
		first = levelSizes[static_cast<size_t>(level)];
	}
	return result;
}

} // namespace hsim
