#include "hsim/prolongation.h"

#include "hsim/mesh.h"
#include "hsim/options.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

#ifdef _OPENMP
#	include <omp.h>
#endif

namespace hsim {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

double supportRadius(double area, int numSamples, double supportSize, double areaFactor)
{
	return std::sqrt((areaFactor * area * supportSize) / (kPi * numSamples));
}

namespace {
struct QueueEntry {
	int vertex;
	double distance;
	bool operator>(const QueueEntry& other) const { return distance > other.distance; }
};
using MinQueue = std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>>;

/* Dijkstra from `source` that stops once the front has passed `radius`; on return D holds the exact graph distance
 * for every vertex with D <= radius. Every vertex whose entry of D was changed is appended to `touched` (possibly more
 * than once), so that the caller can visit and reset exactly the explored region. */
void truncatedDijkstra(const Adjacency& adjacency, const std::vector<double>& lengths, int source, double radius,
                       Eigen::VectorXd& D, std::vector<int>& touched)
{
	MinQueue queue;
	D(source) = 0.0;
	touched.push_back(source);
	queue.push({source, 0.0});

	while (!queue.empty()) {
		const QueueEntry top = queue.top();
		queue.pop();
		if (top.distance > D(top.vertex)) continue; // stale entry
		if (top.distance > radius) break; // every remaining vertex is farther than the support radius
		const Adjacency::Range neighbors = adjacency.neighbors(top.vertex);
		const double* length = lengths.data() + adjacency.offset(top.vertex);
		for (Eigen::Index j = 0; j < neighbors.size(); ++j) {
			const int neighbor = neighbors.first[j];
			const double candidate = top.distance + length[j];
			if (candidate < D(neighbor)) {
				D(neighbor) = candidate;
				touched.push_back(neighbor);
				queue.push({neighbor, candidate});
			}
		}
	}
}
} // namespace

SparseMatrix radialBasisLevel(const Eigen::MatrixXd& V, const Adjacency& adjacency, const Eigen::VectorXi& samples,
                              const std::vector<int>& levelSizes, int k, double radius, int numThreads)
{
	if (k < 0 || k >= static_cast<int>(levelSizes.size())) throw std::out_of_range("radialBasisLevel: level index out of range");

	const int numCenters = levelSizes[static_cast<size_t>(k)];
	const Eigen::Index numRows = (k == 0) ? V.rows() : levelSizes[static_cast<size_t>(k) - 1];

	// Row index of every vertex: the vertex itself on level 0, its index among the samples of level k-1 otherwise.
	std::vector<int> rowOf(static_cast<size_t>(V.rows()), -1);
	for (Eigen::Index row = 0; row < numRows; ++row)
		rowOf[static_cast<size_t>((k == 0) ? static_cast<int>(row) : samples(row))] = static_cast<int>(row);

	// Cubic polynomial phi(d) = a d^3 + b d^2 + 1 with phi(0) = 1, phi(r) = 0, phi'(0) = phi'(r) = 0.
	const double a = 2.0 / std::pow(radius, 3);
	const double b = -3.0 / std::pow(radius, 2);

#ifdef _OPENMP
	const int threads = (numThreads > 0) ? numThreads : omp_get_max_threads();
#else
	(void)numThreads;
	const int threads = 1;
#endif
	std::vector<std::vector<Eigen::Triplet<double>>> triplets(static_cast<size_t>(threads));
	const std::vector<double> lengths = adjacency.edgeLengths(V);

#ifdef _OPENMP
#	pragma omp parallel num_threads(threads)
#endif
	{
#ifdef _OPENMP
		const int tid = omp_get_thread_num();
#else
		const int tid = 0;
#endif
		auto& local = triplets[static_cast<size_t>(tid)];
		Eigen::VectorXd D = Eigen::VectorXd::Constant(V.rows(), std::numeric_limits<double>::infinity());
		std::vector<int> touched;

#ifdef _OPENMP
#	pragma omp for schedule(dynamic, 16)
#endif
		for (int col = 0; col < numCenters; ++col) {
			touched.clear();
			truncatedDijkstra(adjacency, lengths, samples(col), radius, D, touched);
			std::sort(touched.begin(), touched.end());
			touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
			for (const int vertex : touched) {
				const double d = D(vertex);
				const int row = rowOf[static_cast<size_t>(vertex)];
				if (row >= 0 && d <= radius) local.emplace_back(row, col, a * d * d * d + b * d * d + 1.0);
				D(vertex) = std::numeric_limits<double>::infinity();
			}
		}
	}

	std::vector<Eigen::Triplet<double>> all;
	size_t total = 0;
	for (const auto& t : triplets)
		total += t.size();
	all.reserve(total);
	for (const auto& t : triplets)
		all.insert(all.end(), t.begin(), t.end());

	SparseMatrix U(numRows, numCenters);
	U.setFromTriplets(all.begin(), all.end());
	return U;
}

void makePartitionOfUnity(SparseMatrix& U)
{
	Eigen::VectorXd rowSum = Eigen::VectorXd::Zero(U.rows());
	for (Eigen::Index c = 0; c < U.outerSize(); ++c)
		for (SparseMatrix::InnerIterator it(U, c); it; ++it)
			rowSum(it.row()) += it.value();

	Eigen::VectorXd scale(U.rows());
	for (Eigen::Index r = 0; r < U.rows(); ++r)
		scale(r) = (std::abs(rowSum(r)) < 10.0 * std::numeric_limits<double>::epsilon()) ? 1.0 : 1.0 / rowSum(r);

	for (Eigen::Index c = 0; c < U.outerSize(); ++c)
		for (SparseMatrix::InnerIterator it(U, c); it; ++it)
			it.valueRef() = scale(it.row()) * it.value();
}

std::vector<SparseMatrix> buildProlongation(const Eigen::MatrixXd& V, const Adjacency& adjacency, double area,
                                            const Eigen::VectorXi& samples, const std::vector<int>& levelSizes,
                                            const Options& options, std::vector<double>* radii)
{
	const int numLevels = static_cast<int>(levelSizes.size());
	std::vector<SparseMatrix> U(static_cast<size_t>(numLevels));
	if (radii) radii->resize(static_cast<size_t>(numLevels));

	for (int k = 0; k < numLevels; ++k) {
		const double r = supportRadius(area, levelSizes[static_cast<size_t>(k)], options.supportSize, options.areaFactor);
		if (radii) (*radii)[static_cast<size_t>(k)] = r;
		U[static_cast<size_t>(k)] = radialBasisLevel(V, adjacency, samples, levelSizes, k, r, options.numThreads);
		makePartitionOfUnity(U[static_cast<size_t>(k)]);
	}
	return U;
}

SparseMatrix composeProlongation(const std::vector<SparseMatrix>& U, Eigen::Index numVertices)
{
	SparseMatrix P(numVertices, numVertices);
	P.setIdentity();
	for (const auto& Uk : U)
		P = P * Uk;
	return P;
}

} // namespace hsim
