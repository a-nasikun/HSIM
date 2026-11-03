// Regression tests: the solver must reproduce the eigenvalues recorded from the original implementation.
#include <doctest/doctest.h>

#include "test_support.h"

#include "LBEigenSolver.h"

#include <igl/readOBJ.h>

#include <memory>

namespace {

struct RunResult {
	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;
	Eigen::SparseMatrix<double> S, M;
	std::vector<int> levels;
};

void loadMesh(const std::string& mesh, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	if (mesh.rfind("ico", 0) == 0) hsim_test::icosphere(std::stoi(mesh.substr(3)), V, F);
	else REQUIRE(igl::readOBJ(hsim_test::repoPath(mesh), V, F));
}

RunResult run(const hsim_test::Baseline& b)
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	loadMesh(b.mesh, V, F);
	REQUIRE(V.rows() == b.numVertices);

	auto solver = std::make_unique<HSIM::LBEigenSolver>();
	RunResult r;
	std::tie(r.eigenvalues, r.eigenvectors) = solver->eigs(V, F, b.numModes, b.tol, b.maxIter, b.shift);
	r.S = solver->getStiffnessMatrix();
	r.M = solver->getMassMatrix();
	r.levels = solver->getSubspaceSize();
	return r;
}

void checkAgainstBaseline(const std::string& name, double relTol)
{
	const hsim_test::Baseline b = hsim_test::readBaseline(name);
	const RunResult r = run(b);
	const int n = b.numModes;

	CHECK(r.levels == b.levels);
	REQUIRE(r.eigenvalues.size() >= n);
	REQUIRE(r.eigenvectors.cols() >= n);

	// The first eigenvalue of a closed surface is zero (constant eigenfunction).
	CHECK(std::abs(r.eigenvalues(0)) < 1e-10);
	for (int i = 1; i < n; ++i) {
		INFO("eigenvalue index " << i);
		CHECK(r.eigenvalues(i) == doctest::Approx(b.eigenvalues[i]).epsilon(relTol));
	}

	// Residual ||S x - lambda M x||_M / ||S x||_M of every non-constant mode is below the requested tolerance.
	for (int i = 1; i < n; ++i) {
		const Eigen::VectorXd x = r.eigenvectors.col(i);
		const Eigen::VectorXd Sx = r.S * x;
		const Eigen::VectorXd res = Sx - r.eigenvalues(i) * (r.M * x);
		INFO("residual index " << i);
		CHECK(std::sqrt(res.dot(r.M * res) / Sx.dot(r.M * Sx)) < b.tol);
	}

	// Eigenvectors are M-orthonormal.
	const Eigen::MatrixXd X = r.eigenvectors.leftCols(n);
	const Eigen::MatrixXd G = X.transpose() * (r.M * X);
	CHECK((G - Eigen::MatrixXd::Identity(n, n)).cwiseAbs().maxCoeff() < 1e-8);
}

} // namespace

TEST_CASE("regression: Chinese dragon (2000 vertices), one level, 20 modes")
{
	checkAgainstBaseline("dragon2000_n20", 1e-7);
}

TEST_CASE("regression: Chinese dragon (5000 vertices), two levels, 20 modes")
{
	checkAgainstBaseline("dragon5000_n20", 1e-7);
}

TEST_CASE("regression: Chinese dragon (5000 vertices), GUI defaults tol=1e-2, maxIter=5")
{
	checkAgainstBaseline("dragon5000_gui", 1e-7);
}

TEST_CASE("regression: icosphere (2562 vertices), 25 modes")
{
	checkAgainstBaseline("sphere_n25", 1e-7);
}

TEST_CASE("physics: icosphere spectrum approximates l(l+1)")
{
	const hsim_test::Baseline b = hsim_test::readBaseline("sphere_n25");
	const RunResult r = run(b);
	// l = 1, 2, 3, 4 occupy indices [1,4), [4,9), [9,16), [16,25)
	int index = 1;
	for (int l = 1; l <= 4; ++l) {
		for (int m = 0; m < 2 * l + 1; ++m, ++index) {
			INFO("l = " << l << ", index " << index);
			CHECK(r.eigenvalues(index) == doctest::Approx(l * (l + 1.0)).epsilon(0.01));
		}
	}
}
