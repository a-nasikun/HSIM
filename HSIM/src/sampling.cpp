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
		throw std::invalid_argument(
		    "samplingLevelSizes: the mesh (" + std::to_string(numVertices) + " vertices) is too coarse to compute "
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
} // namespace

namespace {
/* Candidates for the farthest vertex: largest distance first, smallest vertex index among equal distances (the same
 * tie-breaking as Eigen's maxCoeff). Entries become stale when the distance of their vertex decreases. */
struct FarthestEntry {
	double distance;
	int vertex;
	bool operator<(const FarthestEntry& other) const
	{
		return distance < other.distance || (distance == other.distance && vertex > other.vertex);
	}
};
using MaxQueue = std::priority_queue<FarthestEntry>;

/* Dijkstra from `source` on the edge graph with precomputed edge lengths; lowers the entries of `distance`.
 * Every lowered entry is also recorded in `farthest` (if given). */
void accumulate(const Adjacency& adjacency, const std::vector<double>& lengths, int source, Eigen::VectorXd& distance,
                MaxQueue* farthest = nullptr)
{
	MinQueue queue;
	distance(source) = 0.0;
	if (farthest) farthest->push({0.0, source});
	queue.push({source, 0.0});

	while (!queue.empty()) {
		const QueueEntry top = queue.top();
		queue.pop();
		if (top.distance > distance(top.vertex)) continue; // stale entry, the vertex was settled with a smaller distance
		const Adjacency::Range neighbors = adjacency.neighbors(top.vertex);
		const double* length = lengths.data() + adjacency.offset(top.vertex);
		for (Eigen::Index j = 0; j < neighbors.size(); ++j) {
			const int neighbor = neighbors.first[j];
			const double candidate = top.distance + length[j];
			if (candidate < distance(neighbor)) {
				distance(neighbor) = candidate;
				if (farthest) farthest->push({candidate, neighbor});
				queue.push({neighbor, candidate});
			}
		}
	}
}
/* Vertex with the largest current distance (smallest index among ties). */
int farthestVertex(MaxQueue& farthest, const Eigen::VectorXd& distance)
{
	while (farthest.top().distance != distance(farthest.top().vertex))
		farthest.pop(); // discard stale entries
	return farthest.top().vertex;
}
} // namespace

void accumulateGraphDistance(const Eigen::MatrixXd& V, const Adjacency& adjacency, int source, Eigen::VectorXd& distance)
{
	accumulate(adjacency, adjacency.edgeLengths(V), source, distance);
}

Samples farthestPointSampling(const Eigen::MatrixXd& V, const Adjacency& adjacency, const std::vector<int>& levelSizes)
{
	if (levelSizes.empty()) throw std::invalid_argument("farthestPointSampling: no levels");
	if (levelSizes.front() > V.rows()) throw std::invalid_argument("farthestPointSampling: more samples than vertices");

	const int numLevels = static_cast<int>(levelSizes.size());
	Samples result;
	result.vertices.resize(levelSizes.front());
	result.distance.resize(static_cast<size_t>(numLevels));

	const std::vector<double> lengths = adjacency.edgeLengths(V);
	Eigen::VectorXd D = Eigen::VectorXd::Constant(V.rows(), std::numeric_limits<double>::infinity());
	result.vertices(0) = 0;

	// The farthest vertex is tracked with a lazy max-heap instead of scanning D after every Dijkstra.
	MaxQueue farthest;
	for (int v = 0; v < static_cast<int>(V.rows()); ++v)
		farthest.push({D(v), v});

	// Coarsest level first; the samples of a finer level extend those of the coarser one.
	int first = 1;
	for (int level = numLevels - 1; level >= 0; --level) {
		for (int i = first; i < levelSizes[static_cast<size_t>(level)]; ++i) {
			accumulate(adjacency, lengths, result.vertices(i - 1), D, &farthest);
			result.vertices(i) = farthestVertex(farthest, D);
		}
		result.distance[static_cast<size_t>(level)] = D;
		first = levelSizes[static_cast<size_t>(level)];
	}
	return result;
}

} // namespace hsim
