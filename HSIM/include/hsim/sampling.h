/**
 * @file sampling.h
 * @brief Nested farthest-point sampling that defines the degrees of freedom of the hierarchy.
 */
#pragma once

#include "common.h"

#include <vector>

namespace hsim {

class Adjacency;

/// Number of samples m_0 > m_1 > ... > m_{L-1} on the levels of the hierarchy.
///
/// Starting from the number of vertices n, the size is repeatedly divided by `levelRatio` (and rounded) as long as the
/// coarsest level can still hold the q = subspaceSize(numModes) iterated vectors; the construction stops once a level
/// has at most 2 q samples.
/// @throws std::invalid_argument if not even one level can be formed (mesh too coarse for numModes).
std::vector<int> samplingLevelSizes(Eigen::Index numVertices, int numModes, double levelRatio);

/// Result of the nested farthest-point sampling.
struct Samples {
	/// Sample vertices, ordered such that the first m_k entries are the samples of level k (nested sets).
	Eigen::VectorXi vertices;
	/// Geodesic distance field (graph distance along edges) to the sample set after level k was completed.
	std::vector<Eigen::VectorXd> distance;
};

/// Farthest-point sampling with respect to the graph (Dijkstra) distance along mesh edges.
/// The first sample is vertex 0, hence the result is deterministic. The coarsest level is sampled first.
Samples farthestPointSampling(const Eigen::MatrixXd& V, const Adjacency& adjacency, const std::vector<int>& levelSizes);

/// Single-source Dijkstra on the edge graph that lowers the entries of `distance` (multi-source accumulation:
/// on return distance(v) = min(distance(v), d(source, v))).
void accumulateGraphDistance(const Eigen::MatrixXd& V, const Adjacency& adjacency, int source, Eigen::VectorXd& distance);

} // namespace hsim
