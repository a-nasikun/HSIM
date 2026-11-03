// Tests for the defects fixed in the 2026 refactoring (see the commit history of Phase 2).
#include <doctest/doctest.h>

#include "test_support.h"

#include "LBEigenSolver.h"

#include <igl/readOBJ.h>

TEST_CASE("a default-initialised solver gives the baseline result (members are initialised)")
{
	const hsim_test::Baseline b = hsim_test::readBaseline("dragon2000_n20");
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	REQUIRE(igl::readOBJ(hsim_test::repoPath(b.mesh), V, F));

	HSIM::LBEigenSolver solver;	// default-initialised (on the stack), not value-initialised
	CHECK_FALSE(solver.getClampBasis());
	Eigen::VectorXd lambda;
	Eigen::MatrixXd X;
	std::tie(lambda, X) = solver.eigs(V, F, b.numModes, b.tol, b.maxIter, b.shift);
	for (int i = 1; i < b.numModes; ++i)
		CHECK(lambda(i) == doctest::Approx(b.eigenvalues[i]).epsilon(1e-7));
}

TEST_CASE("the prolongation operator is a partition of unity")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(4, V, F);
	HSIM::LBEigenSolver solver;
	solver.setup(V, F, 4);
	solver.buildHierarchy(4);	// two levels
	REQUIRE(solver.getLevel() == 2);

	const Eigen::SparseMatrix<double> P = solver.getBasisMatrix();
	const Eigen::VectorXd rowSum = P * Eigen::VectorXd::Ones(P.cols());
	CHECK((rowSum.array() - 1.0).abs().maxCoeff() < 1e-12);
}

TEST_CASE("few eigenpairs (numModes < 8) are computed correctly")
{
	// For numModes < 8 the subspace size is numModes + 8 > 2 numModes; the coarsest level must hold it.
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim_test::icosphere(4, V, F);
	HSIM::LBEigenSolver solver;
	Eigen::VectorXd lambda;
	Eigen::MatrixXd X;
	std::tie(lambda, X) = solver.eigs(V, F, 4, 1e-6, 30, 0.25);
	REQUIRE(lambda.size() >= 4);
	for (int i = 1; i < 4; ++i)
		CHECK(lambda(i) == doctest::Approx(2.0).epsilon(1e-3));	// l = 1 on the unit sphere
}

TEST_CASE("invalid input is reported by exceptions")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	SUBCASE("mesh too coarse for the number of modes")
	{
		hsim_test::icosphere(1, V, F);	// 42 vertices
		HSIM::LBEigenSolver solver;
		solver.setup(V, F, 20);
		CHECK_THROWS_AS(solver.buildHierarchy(20), std::invalid_argument);
	}

	hsim_test::icosphere(3, V, F);	// 642 vertices
	HSIM::LBEigenSolver solver;
	solver.setup(V, F, 5);
	solver.buildHierarchy(5);

	SUBCASE("more modes than the hierarchy was built for")
	{
		CHECK_THROWS_AS(solver.performSubspaceIterationMethod(HSIM::LBEigenSolver::SIMType::SIM_MultiLevel, 50, 5, 1e-2, true,
			HSIM::LBEigenSolver::ShiftType::SHIFT_ZHAO), std::invalid_argument);
	}

	SUBCASE("shift parameter outside [0, 1)")
	{
		solver.zhaoShiftParam = 1.5;
		CHECK_THROWS_AS(solver.performSubspaceIterationMethod(HSIM::LBEigenSolver::SIMType::SIM_MultiLevel, 5, 5, 1e-2, true,
			HSIM::LBEigenSolver::ShiftType::SHIFT_ZHAO), std::invalid_argument);
	}

	SUBCASE("out-of-range eigenpair index")
	{
		solver.performSubspaceIterationMethod(HSIM::LBEigenSolver::SIMType::SIM_MultiLevel, 5, 5, 1e-2, true,
			HSIM::LBEigenSolver::ShiftType::SHIFT_ZHAO);
		CHECK_NOTHROW(solver.getEigenVectors(0));
		CHECK_THROWS_AS(solver.getEigenVectors(-1), std::out_of_range);
		CHECK_THROWS_AS(solver.getEigenVectors(10000), std::out_of_range);
		CHECK_THROWS_AS(solver.getEigenValueSIM(10000), std::out_of_range);
		CHECK_THROWS_AS(solver.getEigenValueReduced(-3), std::out_of_range);
		// The returned vector is a copy (the former implementation returned a dangling reference).
		const Eigen::VectorXd x = solver.getEigenVectors(1);
		CHECK(x.isApprox(solver.getEigenVectors().col(1)));
	}
}
