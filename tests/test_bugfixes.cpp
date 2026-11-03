// Tests for the defects fixed in the 2026 refactoring (Phase 2) and for input validation.
#include <doctest/doctest.h>

#include "test_support.h"

#include <hsim/hsim.h>

TEST_CASE("the prolongation operator is a partition of unity")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(4, V, F);
	hsim::Solver solver;
	solver.setMesh(V, F);
	solver.buildHierarchy(4); // two levels
	REQUIRE(solver.hierarchy().numLevels() == 2);

	for (const hsim::SparseMatrix& U : solver.hierarchy().prolongation) {
		const Eigen::VectorXd rowSum = U * Eigen::VectorXd::Ones(U.cols());
		CHECK((rowSum.array() - 1.0).abs().maxCoeff() < 1e-12);
	}
	const hsim::SparseMatrix& P = solver.hierarchy().basis;
	const Eigen::VectorXd rowSum = P * Eigen::VectorXd::Ones(P.cols());
	CHECK((rowSum.array() - 1.0).abs().maxCoeff() < 1e-12);
}

TEST_CASE("few eigenpairs (numModes < 8) are computed correctly")
{
	// For numModes < 8 the subspace size is numModes + 8 > 2 numModes; the coarsest level must hold it.
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(4, V, F);
	hsim::Options options;
	options.tolerance = 1e-6;
	options.maxIterations = 30;
	options.shiftParameter = 0.25;
	const hsim::Result result = hsim::eigs(V, F, 4, options);
	REQUIRE(result.eigenvalues.size() == 4);
	for (int i = 1; i < 4; ++i)
		CHECK(result.eigenvalues(i) == doctest::Approx(2.0).epsilon(1e-3)); // l = 1 on the unit sphere
}

TEST_CASE("invalid input is reported by exceptions")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	SUBCASE("mesh too coarse for the number of modes")
	{
		hsim_test::icosphere(1, V, F); // 42 vertices
		hsim::Solver solver;
		solver.setMesh(V, F);
		CHECK_THROWS_AS(solver.buildHierarchy(20), std::invalid_argument);
	}

	SUBCASE("invalid meshes")
	{
		hsim::Solver solver;
		CHECK_THROWS_AS(solver.setMesh(Eigen::MatrixXd(), Eigen::MatrixXi()), std::invalid_argument);
		hsim_test::icosphere(1, V, F);
		Eigen::MatrixXi badF = F;
		badF(0, 0) = static_cast<int>(V.rows());
		CHECK_THROWS_AS(solver.setMesh(V, badF), std::invalid_argument);
		CHECK_THROWS_AS(solver.setMesh(V.leftCols(2), F), std::invalid_argument);
		CHECK_THROWS_AS((void)solver.solve(5), std::logic_error); // no mesh set
	}

	SUBCASE("invalid options")
	{
		hsim::Options options;
		options.shiftParameter = 1.5;
		CHECK_THROWS_AS(options.validate(), std::invalid_argument);
		CHECK_THROWS_AS(hsim::Solver{options}, std::invalid_argument);
		options = hsim::Options{};
		options.maxIterations = 0;
		CHECK_THROWS_AS(options.validate(), std::invalid_argument);
		options = hsim::Options{};
		options.levelRatio = 1.0;
		CHECK_THROWS_AS(options.validate(), std::invalid_argument);
		if (!hsim::builtWithCuda()) {
			options = hsim::Options{};
			options.denseBackend = hsim::DenseEigenBackend::CuSolver;
			CHECK_THROWS_AS(options.validate(), std::invalid_argument);
		}
	}

	SUBCASE("more modes than the hierarchy was built for")
	{
		hsim_test::icosphere(3, V, F); // 642 vertices
		hsim::Solver solver;
		solver.setMesh(V, F);
		solver.buildHierarchy(5);
		CHECK_THROWS_AS((void)hsim::hierarchicalSubspaceIteration(solver.operators(), solver.hierarchy(), 40, solver.options()),
		                std::invalid_argument);
		// The Solver rebuilds the hierarchy instead.
		const hsim::Result result = solver.solve(10);
		CHECK(result.eigenvalues.size() == 10);
		CHECK(solver.hierarchy().numModes == 10);
	}
}
