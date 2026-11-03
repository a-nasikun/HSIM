// Unit tests of the dense generalized eigensolver.
#include <doctest/doctest.h>

#include <hsim/hsim.h>

TEST_CASE("dense eigensolver: known spectrum and M-orthonormality")
{
	// S = diag(4, 1, 9), M = diag(2, 1, 3): eigenvalues 1, 2, 3
	const Eigen::MatrixXd S = Eigen::Vector3d(4, 1, 9).asDiagonal();
	const Eigen::MatrixXd M = Eigen::Vector3d(2, 1, 3).asDiagonal();
	const hsim::DenseEigenpairs e = hsim::solveDenseGeneralized(S, M, hsim::DenseEigenBackend::Eigen);
	REQUIRE(e.success);
	CHECK(e.values(0) == doctest::Approx(1.0));
	CHECK(e.values(1) == doctest::Approx(2.0));
	CHECK(e.values(2) == doctest::Approx(3.0));

	// random symmetric S and SPD M
	const int n = 30;
	const Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
	const Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
	const Eigen::MatrixXd S2 = A + A.transpose();
	const Eigen::MatrixXd M2 = B * B.transpose() + n * Eigen::MatrixXd::Identity(n, n);
	const hsim::DenseEigenpairs f = hsim::solveDenseGeneralized(S2, M2);
	REQUIRE(f.success);
	CHECK((S2 * f.vectors - M2 * f.vectors * f.values.asDiagonal()).norm() < 1e-9 * S2.norm());
	CHECK((f.vectors.transpose() * M2 * f.vectors - Eigen::MatrixXd::Identity(n, n)).norm() < 1e-10);
	for (int i = 1; i < n; ++i)
		CHECK(f.values(i) >= f.values(i - 1));
}

TEST_CASE("dense eigensolver: errors")
{
	CHECK_THROWS_AS((void)hsim::solveDenseGeneralized(Eigen::MatrixXd::Identity(3, 3), Eigen::MatrixXd::Identity(2, 2)),
	                std::invalid_argument);
	// M not positive definite: reported as failure, not as an exception
	const Eigen::MatrixXd M = -Eigen::MatrixXd::Identity(3, 3);
	CHECK_FALSE(hsim::solveDenseGeneralized(Eigen::MatrixXd::Identity(3, 3), M, hsim::DenseEigenBackend::Eigen).success);
	if (!hsim::builtWithCuda())
		CHECK_THROWS_AS((void)hsim::solveDenseGeneralized(Eigen::MatrixXd::Identity(3, 3), Eigen::MatrixXd::Identity(3, 3),
		                                                  hsim::DenseEigenBackend::CuSolver),
		                std::invalid_argument);
}

TEST_CASE("dense eigensolver: CUDA and CPU backends agree" * doctest::skip(!hsim::builtWithCuda()))
{
	const int n = 50;
	const Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
	const Eigen::MatrixXd S = A + A.transpose();
	const Eigen::MatrixXd M = Eigen::MatrixXd::Identity(n, n) * 2.0;
	const hsim::DenseEigenpairs cpu = hsim::solveDenseGeneralized(S, M, hsim::DenseEigenBackend::Eigen);
	const hsim::DenseEigenpairs gpu = hsim::solveDenseGeneralized(S, M, hsim::DenseEigenBackend::CuSolver);
	REQUIRE(gpu.success);
	CHECK((cpu.values - gpu.values).cwiseAbs().maxCoeff() < 1e-10 * cpu.values.cwiseAbs().maxCoeff());
}
