/**
 * @file subspace_iteration.h
 * @brief The hierarchical subspace iteration: coarse-to-fine shifted subspace iteration with Rayleigh--Ritz steps.
 */
#pragma once

#include "hierarchy.h"

#include <vector>

namespace hsim {

/// Statistics of one level of the subspace iteration.
struct LevelDiagnostics {
	Eigen::Index dimension = 0; ///< number of degrees of freedom of the level (n on level 0)
	int iterations = 0;         ///< subspace iterations performed
	double shift = 0.0;         ///< spectral shift mu used on this level
	double seconds = 0.0;       ///< wall-clock time of this level (excluding coarser levels)
};

/// Information about a run, e.g. for convergence plots and timing tables.
struct Diagnostics {
	std::vector<int> levelSizes;          ///< m_0, ..., m_{L-1}
	std::vector<LevelDiagnostics> levels; ///< index k = level k (0 = finest, the mesh itself)
	double hierarchySeconds = 0.0;        ///< time for building the hierarchy (0 if it was reused)
	double iterationSeconds = 0.0;        ///< time for the subspace iteration on all levels

	/// Columns: iterations on the finest level. Rows: modes 0, ..., numModes-1.
	Eigen::MatrixXd eigenvalueHistory;    ///< Ritz values
	Eigen::MatrixXd residualHistory;      ///< relative residuals, see relativeResiduals()
	Eigen::MatrixXd ritzCriterionHistory; ///< sqrt|1 - lambda^2 / (q^T q + mu (2 lambda - mu))| of the Ritz vectors q
};

/// Output of the solver.
struct Result {
	Eigen::VectorXd eigenvalues;  ///< numModes smallest eigenvalues, ascending
	Eigen::MatrixXd eigenvectors; ///< corresponding eigenvectors (columns, vertex values), M-orthonormal
	Diagnostics diagnostics;
};

/// Relative residuals r_i = ||S x_i - lambda_i M x_i||_M / ||S x_i||_M of the columns of X.
/// (r_0 of the constant mode is meaningless since S x_0 = 0.)
Eigen::VectorXd relativeResiduals(const Eigen::MatrixXd& X, const Eigen::VectorXd& lambda, const SparseMatrix& S,
	const SparseMatrix& M);

/// Runs the subspace iteration from the coarsest to the finest level.
/// @throws std::invalid_argument if the hierarchy is incomplete or was built for fewer modes than requested;
///         std::runtime_error if a sparse factorization fails.
Result hierarchicalSubspaceIteration(const Operators& operators, const Hierarchy& hierarchy, int numModes,
	const Options& options);

} // namespace hsim
