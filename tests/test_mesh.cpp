// Unit tests of the mesh module: input validation, adjacency, discrete operators.
#include <doctest/doctest.h>

#include "test_support.h"

#include <hsim/hsim.h>

#include <cmath>
#include <set>

TEST_CASE("mesh: adjacency of the icosahedron")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(0, V, F);
	const hsim::Adjacency adjacency(F, V.rows());
	REQUIRE(adjacency.numVertices() == 12);

	for (Eigen::Index v = 0; v < V.rows(); ++v) {
		const auto neighbors = adjacency.neighbors(v);
		CHECK(neighbors.size() == 5); // every vertex of the icosahedron has valence 5
		CHECK(std::is_sorted(neighbors.begin(), neighbors.end()));
		for (const int w : neighbors) {
			CHECK(w != v);
			const auto back = adjacency.neighbors(w);
			CHECK(std::find(back.begin(), back.end(), static_cast<int>(v)) != back.end()); // symmetric
		}
	}

	const std::vector<double> lengths = adjacency.edgeLengths(V);
	for (Eigen::Index v = 0; v < V.rows(); ++v) {
		const auto neighbors = adjacency.neighbors(v);
		for (Eigen::Index j = 0; j < neighbors.size(); ++j)
			CHECK(lengths[adjacency.offset(v) + static_cast<size_t>(j)]
			      == doctest::Approx((V.row(neighbors.first[j]) - V.row(v)).norm()));
	}
}

TEST_CASE("mesh: discrete operators")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(3, V, F);
	const hsim::Operators op = hsim::assembleOperators(V, F);
	const Eigen::Index n = V.rows();

	CHECK(op.stiffness.rows() == n);
	CHECK(op.mass.rows() == n);
	// S is symmetric with constant functions in its kernel
	CHECK((hsim::SparseMatrix(op.stiffness.transpose()) - op.stiffness).norm() < 1e-12 * op.stiffness.norm());
	CHECK((op.stiffness * Eigen::VectorXd::Ones(n)).cwiseAbs().maxCoeff() < 1e-10);
	// S is positive semi-definite on a few random vectors
	for (int trial = 0; trial < 5; ++trial) {
		const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
		CHECK(x.dot(op.stiffness * x) >= -1e-12);
	}
	// M is diagonal and positive, the Voronoi areas sum to the surface area
	for (Eigen::Index c = 0; c < op.mass.outerSize(); ++c)
		for (hsim::SparseMatrix::InnerIterator it(op.mass, c); it; ++it) {
			CHECK(it.row() == it.col());
			CHECK(it.value() > 0.0);
		}
	CHECK(op.mass.diagonal().sum() == doctest::Approx(op.area).epsilon(1e-10));
	// A refined icosphere approximates the unit sphere (area 4 pi) from below
	CHECK(op.area < 4.0 * hsim_test::kPi);
	CHECK(op.area == doctest::Approx(4.0 * hsim_test::kPi).epsilon(0.01));
}

TEST_CASE("mesh: validation, normalisation and reading")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(1, V, F);
	CHECK_NOTHROW(hsim::validateMesh(V, F));

	Eigen::MatrixXd Vbad = V;
	Vbad(3, 1) = std::numeric_limits<double>::quiet_NaN();
	CHECK_THROWS_AS(hsim::validateMesh(Vbad, F), std::invalid_argument);
	CHECK_THROWS_AS(hsim::validateMesh(V, F.leftCols(2)), std::invalid_argument);
	Eigen::MatrixXi Fbad = F;
	Fbad(2, 2) = -1;
	CHECK_THROWS_AS(hsim::validateMesh(V, Fbad), std::invalid_argument);

	Eigen::MatrixXd W = 3.0 * V;
	W.col(0).array() += 7.0;
	hsim::normalizeMesh(W, 2.0);
	const Eigen::RowVectorXd lo = W.colwise().minCoeff(), hi = W.colwise().maxCoeff();
	CHECK((hi - lo).maxCoeff() == doctest::Approx(2.0));
	CHECK((0.5 * (hi + lo)).norm() == doctest::Approx(0.0));

	CHECK_THROWS_AS(hsim::readMesh(hsim_test::repoPath("Models/does_not_exist.obj"), V, F), std::runtime_error);
	CHECK_NOTHROW(hsim::readMesh(hsim_test::repoPath("Models/AIM894_Chinese_Dragon/894_Chinese_Dragon_2000.obj"), V, F));
	CHECK(V.rows() == 2001);
	CHECK(F.rows() == 3998);
}
