// Tests of the complete solver: interface behaviour, determinism, and agreement with an independent reference
// eigensolver (Spectra, shift-invert Lanczos).
#include <doctest/doctest.h>

#include "test_support.h"

#include <hsim/hsim.h>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <algorithm>

namespace {
void dragon2000(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	hsim::readMesh(hsim_test::repoPath("Models/AIM894_Chinese_Dragon/894_Chinese_Dragon_2000.obj"), V, F);
}

/* The `numModes` smallest eigenvalues of S x = lambda M x computed with Spectra (shift-invert around a small negative
 * shift, so that S - sigma M is positive definite). */
Eigen::VectorXd referenceEigenvalues(const hsim::Operators& op, int numModes)
{
	using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
	using BOpType = Spectra::SparseSymMatProd<double>;
	const double sigma = -1e-3 * op.stiffness.diagonal().mean() / op.mass.diagonal().mean();
	OpType opInv(op.stiffness, op.mass);
	BOpType opB(op.mass);
	Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> solver(opInv, opB, numModes, 3 * numModes,
	                                                                                      sigma);
	solver.init();
	solver.compute(Spectra::SortRule::LargestMagn, 2000, 1e-12);
	REQUIRE(solver.info() == Spectra::CompInfo::Successful);
	Eigen::VectorXd values = solver.eigenvalues();
	std::sort(values.data(), values.data() + values.size());
	return values;
}
} // namespace

TEST_CASE("solver: agrees with an independent reference eigensolver")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	dragon2000(V, F);
	const int numModes = 20;

	hsim::Options options;
	options.tolerance = 1e-8;
	options.maxIterations = 50;
	hsim::Solver solver(options);
	solver.setMesh(V, F);
	const hsim::Result result = solver.solve(numModes);

	const Eigen::VectorXd reference = referenceEigenvalues(solver.operators(), numModes);
	CHECK(std::abs(result.eigenvalues(0)) < 1e-12);
	for (int i = 1; i < numModes; ++i) {
		INFO("mode " << i);
		CHECK(result.eigenvalues(i) == doctest::Approx(reference(i)).epsilon(1e-9));
	}
}

TEST_CASE("solver: diagnostics are consistent")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	hsim::readMesh(hsim_test::repoPath("Models/AIM894_Chinese_Dragon/894_Chinese_Dragon_5000.obj"), V, F);
	hsim::Options options;
	options.tolerance = 1e-4;
	options.maxIterations = 20;
	const hsim::Result r = hsim::eigs(V, F, 20, options);
	const hsim::Diagnostics& d = r.diagnostics;

	REQUIRE(d.levelSizes.size() == 2);
	REQUIRE(d.levels.size() == 2);
	CHECK(d.levels[0].dimension == V.rows());
	CHECK(d.levels[1].dimension == d.levelSizes[0]);
	for (const auto& level : d.levels) {
		CHECK(level.iterations >= 1);
		CHECK(level.iterations <= options.maxIterations);
		CHECK(level.shift > 0.0);
	}
	CHECK(d.eigenvalueHistory.rows() == 20);
	CHECK(d.eigenvalueHistory.cols() == d.levels[0].iterations);
	CHECK(d.residualHistory.cols() == d.levels[0].iterations);
	CHECK(d.ritzCriterionHistory.cols() == d.levels[0].iterations);
	// the last column of the history is the result
	CHECK((d.eigenvalueHistory.col(d.eigenvalueHistory.cols() - 1) - r.eigenvalues).cwiseAbs().maxCoeff() == 0.0);
	CHECK(d.residualHistory.col(d.residualHistory.cols() - 1).tail(19).maxCoeff() < options.tolerance);
	CHECK(d.hierarchySeconds > 0.0);
	CHECK(d.iterationSeconds > 0.0);
}

TEST_CASE("solver: reuse, determinism and stage order")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	dragon2000(V, F);

	hsim::Options options;
	options.tolerance = 1e-6;
	options.maxIterations = 20;
	hsim::Solver solver(options);
	CHECK_THROWS_AS(solver.computeSamples(10), std::logic_error);
	solver.setMesh(V, F);
	CHECK_THROWS_AS(solver.computeProlongation(), std::invalid_argument);
	CHECK_THROWS_AS(solver.solveCoarseProblem(), std::invalid_argument);

	const hsim::Result first = solver.solve(10);
	CHECK(first.diagnostics.hierarchySeconds > 0.0);
	const hsim::Result second = solver.solve(10); // reuses the hierarchy
	CHECK(second.diagnostics.hierarchySeconds == 0.0);
	CHECK((first.eigenvalues - second.eigenvalues).cwiseAbs().maxCoeff() == 0.0);

	// the number of threads does not change the result beyond round-off
	options.numThreads = 1;
	const hsim::Result single = hsim::eigs(V, F, 10, options);
	CHECK((first.eigenvalues - single.eigenvalues).cwiseAbs().maxCoeff() < 1e-12 * first.eigenvalues.maxCoeff());

	// the verbosity options route messages to the sink
	int messages = 0;
	options.verbosity = hsim::Verbosity::Summary;
	options.logSink = [&](hsim::Verbosity, const std::string&) { ++messages; };
	(void)hsim::eigs(V, F, 10, options);
	CHECK(messages > 0);
}
