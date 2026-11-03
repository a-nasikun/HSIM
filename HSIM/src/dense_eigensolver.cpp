#include "hsim/dense_eigensolver.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <stdexcept>

namespace hsim {

namespace detail {
#ifdef HSIM_WITH_CUDA
DenseEigenpairs solveDenseCuSolver(const Eigen::MatrixXd& S, const Eigen::MatrixXd& M); // dense_eigensolver_cuda.cpp
#endif

DenseEigenpairs solveDenseEigen(const Eigen::MatrixXd& S, const Eigen::MatrixXd& M)
{
	DenseEigenpairs result;
	// Eigen's generalized solver does not report a failed Cholesky factorization of M (it returns meaningless values),
	// hence the explicit check that M is positive definite (cuSOLVER reports this case through devInfo).
	if (Eigen::LLT<Eigen::MatrixXd>(M).info() != Eigen::Success) return result;
	const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(S, M); // lower triangles, Cholesky of M
	if (solver.info() != Eigen::Success) return result;
	result.values = solver.eigenvalues();
	result.vectors = solver.eigenvectors();
	result.success = result.values.allFinite() && result.vectors.allFinite();
	return result;
}
} // namespace detail

DenseEigenpairs solveDenseGeneralized(const Eigen::MatrixXd& S, const Eigen::MatrixXd& M, DenseEigenBackend backend)
{
	if (S.rows() != S.cols() || M.rows() != M.cols() || S.rows() != M.rows())
		throw std::invalid_argument("solveDenseGeneralized: S and M must be square matrices of equal size");

	switch (backend) {
	case DenseEigenBackend::Eigen:
		return detail::solveDenseEigen(S, M);
	case DenseEigenBackend::CuSolver:
#ifdef HSIM_WITH_CUDA
		return detail::solveDenseCuSolver(S, M);
#else
		throw std::invalid_argument("solveDenseGeneralized: the library was built without CUDA support");
#endif
	case DenseEigenBackend::Default:
	default:
#ifdef HSIM_WITH_CUDA
		return detail::solveDenseCuSolver(S, M);
#else
		return detail::solveDenseEigen(S, M);
#endif
	}
}

} // namespace hsim
