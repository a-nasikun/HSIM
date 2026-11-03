// Unit tests of the radial basis functions and the prolongation operators.
#include <doctest/doctest.h>

#include "test_support.h"

#include <hsim/hsim.h>

#include <cmath>

namespace {
struct Setup {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim::Adjacency adjacency;
	hsim::Operators operators;
	std::vector<int> sizes;
	hsim::Samples samples;

	Setup()
	{
		hsim_test::icosphere(4, V, F);
		adjacency = hsim::Adjacency(F, V.rows());
		operators = hsim::assembleOperators(V, F);
		sizes = hsim::samplingLevelSizes(V.rows(), 4, 10.0);
		samples = hsim::farthestPointSampling(V, adjacency, sizes);
	}
};
} // namespace

TEST_CASE("prolongation: support radius")
{
	// pi r^2 = areaFactor * supportSize * A / m
	const double r = hsim::supportRadius(2.0, 10, 7.5, 2.0);
	CHECK(hsim_test::kPi * r * r == doctest::Approx(2.0 * 7.5 * 2.0 / 10));
}

TEST_CASE("prolongation: radial basis functions")
{
	const Setup s;
	const double radius = hsim::supportRadius(s.operators.area, s.sizes[0], 7.5, 2.0);
	const hsim::SparseMatrix U = hsim::radialBasisLevel(s.V, s.adjacency, s.samples.vertices, s.sizes, 0, radius);
	REQUIRE(U.rows() == s.V.rows());
	REQUIRE(U.cols() == s.sizes[0]);

	for (Eigen::Index c = 0; c < U.outerSize(); ++c)
		for (hsim::SparseMatrix::InnerIterator it(U, c); it; ++it) {
			CHECK(it.value() >= -1e-14);
			CHECK(it.value() <= 1.0);
		}
	// value 1 at the centre of every basis function
	for (int j = 0; j < s.sizes[0]; ++j)
		CHECK(U.coeff(s.samples.vertices(j), j) == doctest::Approx(1.0));

	// the result does not depend on the number of threads
	const hsim::SparseMatrix U1 = hsim::radialBasisLevel(s.V, s.adjacency, s.samples.vertices, s.sizes, 0, radius, 1);
	CHECK((U - U1).norm() == 0.0);

	// level 1 maps the samples of level 1 to the samples of level 0
	const double radius1 = hsim::supportRadius(s.operators.area, s.sizes[1], 7.5, 2.0);
	const hsim::SparseMatrix U_1 = hsim::radialBasisLevel(s.V, s.adjacency, s.samples.vertices, s.sizes, 1, radius1);
	CHECK(U_1.rows() == s.sizes[0]);
	CHECK(U_1.cols() == s.sizes[1]);
	for (int j = 0; j < s.sizes[1]; ++j)
		CHECK(U_1.coeff(j, j) == doctest::Approx(1.0)); // samples are nested
}

TEST_CASE("prolongation: partition of unity")
{
	hsim::SparseMatrix U(3, 2);
	U.insert(0, 0) = 1.0;
	U.insert(0, 1) = 3.0;
	U.insert(2, 1) = 0.5;
	U.makeCompressed();
	hsim::makePartitionOfUnity(U);
	CHECK(U.coeff(0, 0) == doctest::Approx(0.25));
	CHECK(U.coeff(0, 1) == doctest::Approx(0.75));
	CHECK(U.coeff(1, 0) == 0.0); // empty row stays empty
	CHECK(U.coeff(2, 1) == doctest::Approx(1.0));
}

TEST_CASE("prolongation: hierarchy operators")
{
	const Setup s;
	const hsim::Options options;
	const hsim::Hierarchy h = hsim::buildHierarchy(s.V, s.adjacency, s.operators, 4, options);
	REQUIRE(h.numLevels() == 2);
	CHECK(h.basis.rows() == s.V.rows());
	CHECK(h.basis.cols() == s.sizes[1]);

	// Galerkin operators: S_1 = U_1^T U_0^T S U_0 U_1 = P^T S P
	const hsim::SparseMatrix PtSP = h.basis.transpose() * s.operators.stiffness * h.basis;
	CHECK((hsim::SparseMatrix(h.levelStiffness[1] - PtSP)).norm() < 1e-10 * PtSP.norm());

	// Coarse eigenpairs are M_1-orthonormal and ascending
	const Eigen::MatrixXd X = h.coarse.vectors;
	const Eigen::MatrixXd G = X.transpose() * (h.levelMass[1] * X);
	CHECK((G - Eigen::MatrixXd::Identity(G.rows(), G.cols())).cwiseAbs().maxCoeff() < 1e-8);
	for (Eigen::Index i = 1; i < h.coarse.values.size(); ++i)
		CHECK(h.coarse.values(i) >= h.coarse.values(i - 1));

	// basis functions of every level, evaluated at the vertices, form a partition of unity
	for (int level = 0; level < h.numLevels(); ++level) {
		Eigen::VectorXd sum = Eigen::VectorXd::Zero(s.V.rows());
		for (int j = 0; j < h.levelSizes[static_cast<size_t>(level)]; ++j)
			sum += hsim::basisFunction(h, level, j);
		CHECK((sum.array() - 1.0).abs().maxCoeff() < 1e-12);
	}
	CHECK_THROWS_AS((void)hsim::basisFunction(h, 2, 0), std::out_of_range);
	CHECK_THROWS_AS((void)hsim::basisFunction(h, 0, -1), std::out_of_range);
}
