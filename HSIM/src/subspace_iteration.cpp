#include "hsim/subspace_iteration.h"

#include "logger.h"
#include "timer.h"

#include <Eigen/SparseCholesky>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#	include <omp.h>
#endif

namespace hsim {

namespace {

/* Y = A X for sparse A and a dense block X, computed column by column (in parallel). Eigen's built-in product for a
 * column-major sparse matrix updates whole rows of Y, which is cache-unfriendly for blocks with many columns. */
Eigen::MatrixXd multiply(const SparseMatrix& A, const Eigen::MatrixXd& X, int numThreads = 0)
{
	Eigen::MatrixXd Y(A.rows(), X.cols());
#ifdef _OPENMP
	const int threads = (numThreads > 0) ? numThreads : omp_get_max_threads();
#	pragma omp parallel for schedule(static) num_threads(threads)
#else
	(void)numThreads;
#endif
	for (Eigen::Index j = 0; j < X.cols(); ++j)
		Y.col(j).noalias() = A * X.col(j);
	return Y;
}

} // namespace

Eigen::VectorXd relativeResiduals(const Eigen::MatrixXd& X, const Eigen::VectorXd& lambda, const SparseMatrix& S,
                                  const SparseMatrix& M)
{
	const Eigen::Index n = lambda.rows();
	const Eigen::MatrixXd SX = multiply(S, X.leftCols(n));
	const Eigen::MatrixXd R = SX - multiply(M, X.leftCols(n)) * lambda.asDiagonal();
	const Eigen::RowVectorXd numerator = R.cwiseProduct(multiply(M, R)).colwise().sum();
	const Eigen::RowVectorXd denominator = SX.cwiseProduct(multiply(M, SX)).colwise().sum();
	return numerator.cwiseQuotient(denominator).cwiseSqrt().transpose();
}

namespace {

/* sqrt|1 - lambda_i^2 / (q_i^T q_i + mu (2 lambda_i - mu))| for the Ritz coefficient vectors q_i (columns of Q). */
Eigen::VectorXd ritzCriterion(const Eigen::MatrixXd& Q, const Eigen::VectorXd& lambda, double mu)
{
	Eigen::VectorXd c(lambda.rows());
	for (Eigen::Index i = 0; i < lambda.rows(); ++i) {
		const double lambda2 = lambda(i) * lambda(i);
		double qTq = Q.col(i).transpose() * Q.col(i);
		qTq += mu * (2 * lambda(i) - mu);
		c(i) = std::sqrt(std::abs(1.0 - lambda2 / qTq));
	}
	return c;
}

struct History {
	Eigen::MatrixXd eigenvalues, residuals, ritz;
};

/* Shifted subspace iteration on one level, starting from the prolongated vectors X with Ritz values lambda.
 * The shift is chosen once per level from the incoming Ritz values. Index 0 (constant mode) is excluded from the
 * shift choice and the convergence test. */
LevelDiagnostics iterateLevel(const SparseMatrix& S, const SparseMatrix& M, bool finest, int numModes, const Options& options,
                              const detail::Logger& log, int level, Eigen::MatrixXd& X, Eigen::VectorXd& lambda, History* history)
{
	const detail::Timer timer;
	const int q = subspaceSize(numModes);
	const double tol = options.tolerance;
	const int maxIter = options.maxIterations;

	SparseMatrix Id(S.rows(), S.cols());
	Id.setIdentity();

	Eigen::SimplicialLDLT<SparseMatrix> factorization;
	Eigen::MatrixXd Xbar(X.rows(), q);
	Xbar.setZero();
	Eigen::MatrixXd Sproj, Mproj;

	const int firstNonTrivial = 1;
	int lastConvergedId = firstNonTrivial;
	double eps = 1e3 * std::numeric_limits<double>::epsilon();
	double mu = 0.0;
	int factorizations = 0;
	int iterations = 0;
	bool ritzSuccess = false;
	bool isConverged = false;

	do {
		/* Factorise the (shifted) operator: once with the shift, again with a tiny regularisation whenever the
		 * Rayleigh--Ritz step failed. */
		detail::Timer stage;
		if (!ritzSuccess) {
			if (factorizations == 0) {
				if (tol < 1e-9 && finest) {
					mu = 0.0; // very high accuracy requested: no spectral shift on the finest level
					factorization.compute(S + eps * Id);
				}
				else {
					const int shiftId
					    = lastConvergedId + static_cast<int>(std::floor(options.shiftParameter * (numModes - lastConvergedId)));
					mu = lambda(shiftId);
					factorization.compute(S - mu * M);
				}
			}
			else {
				eps *= 10;
				mu = eps;
				factorization.compute(S + mu * Id);
			}
			if (factorization.info() != Eigen::Success)
				throw std::runtime_error("hierarchicalSubspaceIteration: sparse LDLT factorization failed on level "
				                         + std::to_string(level));
			++factorizations;
			log(Verbosity::Detailed, "[hsim]   level ", level, ": factorization ", factorizations, " with shift ", mu, " in ",
			    stage.seconds(), " s");
		}

		/* Inverse iteration step  Xbar = (S - mu M)^{-1} M X */
		stage = detail::Timer();
		// The q solves are independent and share the (read-only) factorization.
		const Eigen::MatrixXd rhs = multiply(M, X, options.numThreads);
#ifdef _OPENMP
		const int threads = (options.numThreads > 0) ? options.numThreads : omp_get_max_threads();
#	pragma omp parallel for schedule(static) num_threads(threads)
#endif
		for (int j = 0; j < q; ++j)
			Xbar.col(j) = factorization.solve(rhs.col(j));

		const double solveSeconds = stage.seconds();

		/* Rayleigh--Ritz projection */
		stage = detail::Timer();
		Sproj.noalias() = Xbar.transpose() * multiply(S, Xbar, options.numThreads);
		Mproj.noalias() = Xbar.transpose() * multiply(M, Xbar, options.numThreads);
		const double projectionSeconds = stage.seconds();
		stage = detail::Timer();
		const DenseEigenpairs ritz = solveDenseGeneralized(Sproj, Mproj, options.denseBackend);
		const double ritzSeconds = stage.seconds();
		ritzSuccess = ritz.success;
		++iterations;

		if (!ritzSuccess) {
			log(Verbosity::Summary, "[hsim]   level ", level, ": Rayleigh-Ritz step failed in iteration ", iterations);
			if (iterations >= maxIter) isConverged = true;
			continue;
		}

		lambda = ritz.values;
		X = Xbar * ritz.vectors;

		/* Convergence check */
		stage = detail::Timer();
		const Eigen::VectorXd residual = relativeResiduals(X, lambda, S, M);
		log(Verbosity::Detailed, "[hsim]   level ", level, ", iteration ", iterations, ": solves ", solveSeconds,
		    " s, projection ", projectionSeconds, " s, Rayleigh-Ritz ", ritzSeconds, " s, residuals ", stage.seconds(), " s");
		if (finest && history) {
			history->eigenvalues.col(iterations - 1) = lambda.head(numModes);
			history->residuals.col(iterations - 1) = residual.head(numModes);
			history->ritz.col(iterations - 1) = ritzCriterion(ritz.vectors, lambda, mu).head(numModes);
		}
		log(Verbosity::Detailed, "[hsim]   level ", level, ", iteration ", iterations, ": residual of mode ", numModes - 1, " = ",
		    residual(numModes - 1));

		if (iterations >= maxIter || residual(numModes - 1) < tol) isConverged = true;

		for (int i = lastConvergedId; i < numModes; ++i) {
			if (residual(i) < tol)
				lastConvergedId = i;
			else
				break;
		}
		if (iterations < maxIter && lastConvergedId < numModes - 1) isConverged = false;
	} while (!isConverged);

	LevelDiagnostics d;
	d.dimension = S.rows();
	d.iterations = iterations;
	d.shift = mu;
	d.seconds = timer.seconds();
	return d;
}

} // namespace

Result hierarchicalSubspaceIteration(const Operators& operators, const Hierarchy& hierarchy, int numModes, const Options& options)
{
	options.validate();
	if (numModes < 1) throw std::invalid_argument("hierarchicalSubspaceIteration: the number of modes must be positive");
	if (!hierarchy.hasCoarseSolution())
		throw std::invalid_argument("hierarchicalSubspaceIteration: the hierarchy is incomplete (build it first)");
	const int q = subspaceSize(numModes);
	if (q > hierarchy.coarse.vectors.cols())
		throw std::invalid_argument("hierarchicalSubspaceIteration: the hierarchy was built for "
		                            + std::to_string(hierarchy.numModes) + " eigenpairs and cannot be used for "
		                            + std::to_string(numModes));

	const detail::Logger log(options);
	const detail::Timer timer;
	const int numLevels = hierarchy.numLevels();

	// Operators of the space iterated on level k: S itself on level 0, the Galerkin restriction to the samples of
	// level k-1 otherwise (cached in the hierarchy).
	auto stiffnessOn = [&](int k) -> const SparseMatrix& {
		return (k == 0) ? operators.stiffness : hierarchy.levelStiffness[static_cast<size_t>(k) - 1];
	};
	auto massOn = [&](int k) -> const SparseMatrix& {
		return (k == 0) ? operators.mass : hierarchy.levelMass[static_cast<size_t>(k) - 1];
	};

	Result result;
	Diagnostics& diag = result.diagnostics;
	diag.levelSizes = hierarchy.levelSizes;
	diag.levels.resize(static_cast<size_t>(numLevels));

	History history;
	history.eigenvalues = Eigen::MatrixXd::Zero(numModes, options.maxIterations);
	history.residuals = Eigen::MatrixXd::Zero(numModes, options.maxIterations);
	history.ritz = Eigen::MatrixXd::Zero(numModes, options.maxIterations);

	// Start with the eigenpairs of the coarsest level, then prolongate and iterate from coarse to fine.
	Eigen::MatrixXd X = hierarchy.coarse.vectors.leftCols(q);
	Eigen::VectorXd lambda = hierarchy.coarse.values.head(q);
	for (int k = numLevels - 1; k >= 0; --k) {
		X = hierarchy.prolongation[static_cast<size_t>(k)] * X;
		diag.levels[static_cast<size_t>(k)]
		    = iterateLevel(stiffnessOn(k), massOn(k), k == 0, numModes, options, log, k, X, lambda, &history);
		const LevelDiagnostics& d = diag.levels[static_cast<size_t>(k)];
		log(Verbosity::Summary, "[hsim] level ", k, " (", d.dimension, " DOFs): ", d.iterations, " iteration(s) in ", d.seconds,
		    " s");
	}

	const int finestIterations = diag.levels[0].iterations;
	diag.eigenvalueHistory = history.eigenvalues.leftCols(finestIterations);
	diag.residualHistory = history.residuals.leftCols(finestIterations);
	diag.ritzCriterionHistory = history.ritz.leftCols(finestIterations);
	diag.iterationSeconds = timer.seconds();

	result.eigenvalues = lambda.head(numModes);
	result.eigenvectors = X.leftCols(numModes);
	return result;
}

} // namespace hsim
