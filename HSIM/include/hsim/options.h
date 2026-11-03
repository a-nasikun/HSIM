/**
 * @file options.h
 * @brief All user-adjustable parameters of the hierarchical subspace iteration method.
 *
 * The defaults reproduce the behaviour of the original implementation used for the paper
 * A. Nasikun and K. Hildebrandt, "The Hierarchical Subspace Iteration Method for Laplace--Beltrami
 * Eigenproblems", ACM Transactions on Graphics 41(2), 2022.
 */
#pragma once

#include "log.h"

namespace hsim {

/// Backend used for the small dense generalized eigenproblems (coarsest level and Rayleigh--Ritz steps).
enum class DenseEigenBackend {
	Default,  ///< cuSOLVER if the library was built with HSIM_WITH_CUDA, Eigen otherwise
	Eigen,    ///< Eigen::GeneralizedSelfAdjointEigenSolver on the CPU
	CuSolver, ///< cusolverDnDsygvd on the GPU (requires a build with HSIM_WITH_CUDA)
};

/// Parameters of the method. All members have defaults; call validate() to check a modified set.
struct Options {
	/* ---------------- Hierarchy construction ---------------- */

	/// Ratio between the number of degrees of freedom of two consecutive levels
	/// (level k has about n / levelRatio^(k+1) samples, n = number of vertices).
	double levelRatio = 10.0;

	/// Average number of samples of the same level that lie inside the support of one basis function.
	/// Determines the support radius r_k of the basis functions on level k by
	/// pi r_k^2 = areaFactor * supportSize * A / m_k (A: surface area, m_k: number of samples).
	double supportSize = 7.5;

	/* ---------------- Subspace iteration ---------------- */

	/// Convergence tolerance on the relative residual ||S x - lambda M x||_M / ||S x||_M of every sought eigenpair.
	double tolerance = 1e-2;

	/// Maximal number of subspace iterations per level.
	int maxIterations = 2;

	/// Position of the spectral shift: on every level the operator is shifted once by the Ritz value with
	/// index c + floor(shiftParameter * (numModes - c)), where c = 1 is the first non-trivial mode.
	/// Must lie in [0, 1). 0 shifts to the first non-zero eigenvalue.
	double shiftParameter = 0.3;

	/* ---------------- Linear algebra and parallelism ---------------- */

	/// Solver for the dense generalized eigenproblems.
	DenseEigenBackend denseBackend = DenseEigenBackend::Default;

	/// Number of OpenMP threads for the construction of the prolongation operators (0: OpenMP default).
	int numThreads = 0;

	/* ---------------- Output ---------------- */

	/// Amount of progress output.
	Verbosity verbosity = Verbosity::Silent;

	/// Destination of the progress output; empty means defaultLogSink().
	LogSink logSink;

	/* ---------------- Advanced ---------------- */

	/// Factor multiplying the surface area in the support-radius formula (see supportSize).
	/// The original implementation used the sum of the doubled triangle areas, i.e. 2 A; the default keeps
	/// this so that results are identical to the published ones.
	double areaFactor = 2.0;

	/// Throws std::invalid_argument if a parameter is out of range.
	void validate() const;
};

/// Number q of vectors in the iterated subspace when numModes eigenpairs are sought: q = max(2 numModes, numModes + 8).
int subspaceSize(int numModes);

/// True if the library was built with cuSOLVER support (CMake option HSIM_WITH_CUDA).
bool builtWithCuda();

} // namespace hsim
