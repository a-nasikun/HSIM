// Unit tests of the level sizes and the farthest-point sampling.
#include <doctest/doctest.h>

#include "test_support.h"

#include <hsim/hsim.h>

#include <limits>
#include <set>

TEST_CASE("sampling: level sizes")
{
	using V = std::vector<int>;
	CHECK(hsim::samplingLevelSizes(137098, 100, 10.0) == V{13710, 1371});
	CHECK(hsim::samplingLevelSizes(241607, 100, 10.0) == V{24161, 2416, 242});
	CHECK(hsim::samplingLevelSizes(5001, 20, 10.0) == V{500, 50});
	CHECK(hsim::samplingLevelSizes(2001, 20, 10.0) == V{200});
	// q = max(2n, n + 8) = 12 for n = 4: the coarsest level holds at least q samples
	CHECK(hsim::samplingLevelSizes(2562, 4, 10.0) == V{256, 26});
	CHECK_THROWS_AS((void)hsim::samplingLevelSizes(100, 20, 10.0), std::invalid_argument);
	CHECK_THROWS_AS((void)hsim::samplingLevelSizes(1000, 0, 10.0), std::invalid_argument);
	CHECK_THROWS_AS((void)hsim::samplingLevelSizes(1000, 5, 1.0), std::invalid_argument);
}

namespace {
/* Reference: Bellman-Ford style relaxation until no change (slow but obviously correct). */
Eigen::VectorXd referenceDistance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int source)
{
	Eigen::VectorXd D = Eigen::VectorXd::Constant(V.rows(), std::numeric_limits<double>::infinity());
	D(source) = 0.0;
	for (bool changed = true; changed;) {
		changed = false;
		for (Eigen::Index f = 0; f < F.rows(); ++f)
			for (int a = 0; a < 3; ++a)
				for (int b = 0; b < 3; ++b) {
					const int u = F(f, a), v = F(f, b);
					const double candidate = D(u) + (V.row(u) - V.row(v)).norm();
					if (candidate < D(v) - 1e-15) {
						D(v) = candidate;
						changed = true;
					}
				}
	}
	return D;
}
} // namespace

TEST_CASE("sampling: graph distance agrees with a reference implementation")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(2, V, F);
	const hsim::Adjacency adjacency(F, V.rows());
	for (const int source : {0, 17, 101}) {
		Eigen::VectorXd D = Eigen::VectorXd::Constant(V.rows(), std::numeric_limits<double>::infinity());
		hsim::accumulateGraphDistance(V, adjacency, source, D);
		CHECK((D - referenceDistance(V, F, source)).cwiseAbs().maxCoeff() < 1e-12);
		// graph distance on the sphere is at least the Euclidean distance
		for (Eigen::Index v = 0; v < V.rows(); ++v)
			CHECK(D(v) >= (V.row(v) - V.row(source)).norm() - 1e-12);
	}
}

TEST_CASE("sampling: nested farthest-point samples")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(4, V, F);
	const hsim::Adjacency adjacency(F, V.rows());
	const std::vector<int> sizes = hsim::samplingLevelSizes(V.rows(), 4, 10.0);
	const hsim::Samples s = hsim::farthestPointSampling(V, adjacency, sizes);

	REQUIRE(s.vertices.size() == sizes.front());
	CHECK(s.vertices(0) == 0);
	const std::set<int> unique(s.vertices.data(), s.vertices.data() + s.vertices.size());
	CHECK(unique.size() == static_cast<size_t>(s.vertices.size()));

	// Deterministic
	const hsim::Samples again = hsim::farthestPointSampling(V, adjacency, sizes);
	CHECK(again.vertices == s.vertices);

	// Distance fields: zero at the (finished) samples, and non-increasing from coarse to fine levels
	REQUIRE(s.distance.size() == sizes.size());
	for (int i = 0; i < sizes.front() - 1; ++i)
		CHECK(s.distance[0](s.vertices(i)) == 0.0);
	CHECK((s.distance[0].array() <= s.distance[1].array()).all());
	// Farthest-point property: the next sample is the farthest vertex from the previous ones
	CHECK(s.distance[1].maxCoeff() >= s.distance[0].maxCoeff());
}
