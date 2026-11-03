/**
 * @file dense_eigensolver.h
 * @brief Dense generalized symmetric-definite eigensolver with interchangeable backends.
 */
#pragma once

#include "options.h"

#include <Eigen/Core>

namespace hsim {

/// All eigenpairs of a dense pencil, eigenvalues in ascending order, eigenvectors M-orthonormal.
struct DenseEigenpairs {
	Eigen::VectorXd values;
	Eigen::MatrixXd vectors;
	bool success = false; ///< false if the solver reported an error or produced non-finite values
};

/// Solves S x = lambda M x for symmetric S and symmetric positive definite M (only the lower triangles are used).
/// @throws std::invalid_argument for inconsistent sizes or if the CuSolver backend is requested in a CPU-only build;
///         std::runtime_error for CUDA runtime errors.
[[nodiscard]] DenseEigenpairs solveDenseGeneralized(const Eigen::MatrixXd& S, const Eigen::MatrixXd& M,
                                                    DenseEigenBackend backend = DenseEigenBackend::Default);

} // namespace hsim
