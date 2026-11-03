#include "hsim/hierarchy.h"

#include "hsim/prolongation.h"
#include "logger.h"
#include "timer.h"

#include <stdexcept>

namespace hsim {

void computeSamples(Hierarchy& hierarchy, const Eigen::MatrixXd& V, const Adjacency& adjacency, int numModes,
                    const Options& options)
{
	options.validate();
	const detail::Logger log(options);
	const detail::Timer timer;

	hierarchy = Hierarchy{};
	hierarchy.numModes = numModes;
	hierarchy.levelSizes = samplingLevelSizes(V.rows(), numModes, options.levelRatio);
	hierarchy.samples = farthestPointSampling(V, adjacency, hierarchy.levelSizes);

	log(Verbosity::Summary, "[hsim] sampling: ", hierarchy.numLevels(), " level(s) in ", timer.seconds(), " s");
	for (int k = 0; k < hierarchy.numLevels(); ++k)
		log(Verbosity::Detailed, "[hsim]   level ", k, ": ", hierarchy.levelSizes[static_cast<size_t>(k)], " samples");
}

void computeProlongation(Hierarchy& hierarchy, const Eigen::MatrixXd& V, const Adjacency& adjacency, const Operators& operators,
                         const Options& options)
{
	if (!hierarchy.hasSamples()) throw std::invalid_argument("computeProlongation: the samples have not been computed");
	options.validate();
	const detail::Logger log(options);
	const detail::Timer timer;

	hierarchy.prolongation = buildProlongation(V, adjacency, operators.area, hierarchy.samples.vertices, hierarchy.levelSizes,
	                                           options, &hierarchy.supportRadii);
	hierarchy.basis = composeProlongation(hierarchy.prolongation, V.rows());
	hierarchy.coarse = DenseEigenpairs{};

	log(Verbosity::Summary, "[hsim] prolongation: ", hierarchy.basis.nonZeros(), " non-zeros (",
	    static_cast<double>(hierarchy.basis.nonZeros()) / static_cast<double>(hierarchy.basis.rows()), " per vertex) in ",
	    timer.seconds(), " s");
	for (int k = 0; k < hierarchy.numLevels(); ++k) {
		const SparseMatrix& U = hierarchy.prolongation[static_cast<size_t>(k)];
		log(Verbosity::Detailed, "[hsim]   U_", k, ": ", U.rows(), " x ", U.cols(), ", support radius ",
		    hierarchy.supportRadii[static_cast<size_t>(k)], ", ",
		    static_cast<double>(U.nonZeros()) / static_cast<double>(U.rows()), " non-zeros per row");
	}
}

void solveCoarseProblem(Hierarchy& hierarchy, const Operators& operators, const Options& options)
{
	if (!hierarchy.hasProlongation()) throw std::invalid_argument("solveCoarseProblem: the prolongation has not been computed");
	options.validate();
	const detail::Logger log(options);
	const detail::Timer timer;

	const SparseMatrix& P = hierarchy.basis;
	const SparseMatrix Sc = P.transpose() * operators.stiffness * P;
	const SparseMatrix Mc = P.transpose() * operators.mass * P;
	const Eigen::MatrixXd S(Sc);
	const Eigen::MatrixXd M(Mc);

	// If the dense solver fails, retry with S + eps I for eps = 1e-15, 1e-14, ... (at most 9 retries).
	hierarchy.coarse = solveDenseGeneralized(S, M, options.denseBackend);
	double eps = 1e-16;
	for (int attempt = 1; !hierarchy.coarse.success && attempt < 10; ++attempt) {
		eps *= 10.0;
		Eigen::MatrixXd SReg = S;
		SReg.diagonal().array() += eps;
		log(Verbosity::Summary, "[hsim] coarse eigenproblem failed, retrying with regularisation ", eps);
		hierarchy.coarse = solveDenseGeneralized(SReg, M, options.denseBackend);
	}
	if (!hierarchy.coarse.success)
		throw std::runtime_error("solveCoarseProblem: the dense eigensolver failed on the coarsest level");

	log(Verbosity::Summary, "[hsim] coarse eigenproblem (", S.rows(), " x ", S.cols(), ") in ", timer.seconds(), " s");
}

Eigen::VectorXd basisFunction(const Hierarchy& hierarchy, int level, int index)
{
	if (!hierarchy.hasProlongation()) throw std::invalid_argument("basisFunction: the prolongation has not been computed");
	if (level < 0 || level >= hierarchy.numLevels()) throw std::out_of_range("basisFunction: level index out of range");
	const SparseMatrix& Uk = hierarchy.prolongation[static_cast<size_t>(level)];
	if (index < 0 || index >= Uk.cols()) throw std::out_of_range("basisFunction: basis function index out of range");

	Eigen::VectorXd f = Uk.col(index);
	for (int k = level - 1; k >= 0; --k)
		f = hierarchy.prolongation[static_cast<size_t>(k)] * f;
	return f;
}

Hierarchy buildHierarchy(const Eigen::MatrixXd& V, const Adjacency& adjacency, const Operators& operators, int numModes,
                         const Options& options)
{
	Hierarchy hierarchy;
	computeSamples(hierarchy, V, adjacency, numModes, options);
	computeProlongation(hierarchy, V, adjacency, operators, options);
	solveCoarseProblem(hierarchy, operators, options);
	return hierarchy;
}

} // namespace hsim
