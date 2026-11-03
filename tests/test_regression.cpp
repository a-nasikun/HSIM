// Regression tests: the solver must reproduce the eigenvalues recorded from the original implementation.
#include <doctest/doctest.h>

#include "test_support.h"

#include <hsim/hsim.h>

namespace {

void loadMesh(const std::string& mesh, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	if (mesh.rfind("ico", 0) == 0)
		hsim_test::icosphere(std::stoi(mesh.substr(3)), V, F);
	else
		hsim::readMesh(hsim_test::repoPath(mesh), V, F);
}

hsim::Options optionsFor(const hsim_test::Baseline& b)
{
	hsim::Options options;
	options.tolerance = b.tol;
	options.maxIterations = b.maxIter;
	options.shiftParameter = b.shift;
	return options;
}

struct Run {
	hsim::Result result;
	hsim::Operators operators;
};

Run run(const hsim_test::Baseline& b)
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	loadMesh(b.mesh, V, F);
	REQUIRE(V.rows() == b.numVertices);

	hsim::Solver solver(optionsFor(b));
	solver.setMesh(V, F);
	Run r;
	r.result = solver.solve(b.numModes);
	r.operators = solver.operators();
	return r;
}

void checkAgainstBaseline(const std::string& name, double relTol)
{
	const hsim_test::Baseline b = hsim_test::readBaseline(name);
	const Run r = run(b);
	const hsim::Result& res = r.result;
	const int n = b.numModes;

	CHECK(res.diagnostics.levelSizes == b.levels);
	REQUIRE(res.eigenvalues.size() == n);
	REQUIRE(res.eigenvectors.cols() == n);
	REQUIRE(res.eigenvectors.rows() == b.numVertices);

	// The first eigenvalue of a closed surface is zero (constant eigenfunction).
	CHECK(std::abs(res.eigenvalues(0)) < 1e-10);
	for (int i = 1; i < n; ++i) {
		INFO("eigenvalue index " << i);
		CHECK(res.eigenvalues(i) == doctest::Approx(b.eigenvalues[static_cast<size_t>(i)]).epsilon(relTol));
	}

	// Every non-constant mode satisfies the requested tolerance.
	const Eigen::VectorXd residual
	    = hsim::relativeResiduals(res.eigenvectors, res.eigenvalues, r.operators.stiffness, r.operators.mass);
	for (int i = 1; i < n; ++i) {
		INFO("residual index " << i);
		CHECK(residual(i) < b.tol);
	}

	// Eigenvectors are M-orthonormal.
	const Eigen::MatrixXd G = res.eigenvectors.transpose() * (r.operators.mass * res.eigenvectors);
	CHECK((G - Eigen::MatrixXd::Identity(n, n)).cwiseAbs().maxCoeff() < 1e-8);
}

} // namespace

TEST_CASE("regression: Chinese dragon (2000 vertices), one level, 20 modes")
{
	checkAgainstBaseline("dragon2000_n20", 1e-10);
}

TEST_CASE("regression: Chinese dragon (5000 vertices), two levels, 20 modes")
{
	checkAgainstBaseline("dragon5000_n20", 1e-10);
}

TEST_CASE("regression: Chinese dragon (5000 vertices), GUI defaults tol=1e-2, maxIter=5")
{
	checkAgainstBaseline("dragon5000_gui", 1e-10);
}

TEST_CASE("regression: icosphere (2562 vertices), 25 modes")
{
	checkAgainstBaseline("sphere_n25", 1e-10);
}

TEST_CASE("physics: icosphere spectrum approximates l(l+1)")
{
	const hsim_test::Baseline b = hsim_test::readBaseline("sphere_n25");
	const Run r = run(b);
	// l = 1, 2, 3, 4 occupy indices [1,4), [4,9), [9,16), [16,25)
	int index = 1;
	for (int l = 1; l <= 4; ++l) {
		for (int m = 0; m < 2 * l + 1; ++m, ++index) {
			INFO("l = " << l << ", index " << index);
			CHECK(r.result.eigenvalues(index) == doctest::Approx(l * (l + 1.0)).epsilon(0.01));
		}
	}
}
