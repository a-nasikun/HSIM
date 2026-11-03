#include "hsim/solver.h"

#include "logger.h"
#include "timer.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace hsim {

Result eigs(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes, const Options& options)
{
	Solver solver(options);
	solver.setMesh(V, F);
	return solver.solve(numModes);
}

Solver::Solver(Options options) : options_(std::move(options)) { options_.validate(); }

void Solver::setMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	const detail::Logger log(options_);
	const detail::Timer timer;
	operators_ = assembleOperators(V, F); // validates the mesh
	V_ = V;
	F_ = F;
	adjacency_ = Adjacency(F_, V_.rows());
	hierarchy_ = Hierarchy{};
	hierarchySeconds_ = 0.0;
	log(Verbosity::Summary, "[hsim] mesh with ", V_.rows(), " vertices and ", F_.rows(), " faces, operators assembled in ",
		timer.seconds(), " s");
}

void Solver::setOptions(const Options& options)
{
	options.validate();
	options_ = options;
}

void Solver::requireMesh(const char* stage) const
{
	if (!hasMesh()) throw std::logic_error(std::string("Solver::") + stage + ": no mesh has been set (call setMesh first)");
}

void Solver::computeSamples(int numModes)
{
	requireMesh("computeSamples");
	const detail::Timer timer;
	hsim::computeSamples(hierarchy_, V_, adjacency_, numModes, options_);
	hierarchySeconds_ = timer.seconds();
}

void Solver::computeProlongation()
{
	requireMesh("computeProlongation");
	const detail::Timer timer;
	hsim::computeProlongation(hierarchy_, V_, adjacency_, operators_, options_);
	hierarchySeconds_ += timer.seconds();
}

void Solver::solveCoarseProblem()
{
	requireMesh("solveCoarseProblem");
	const detail::Timer timer;
	hsim::solveCoarseProblem(hierarchy_, operators_, options_);
	hierarchySeconds_ += timer.seconds();
}

void Solver::buildHierarchy(int numModes)
{
	computeSamples(numModes);
	computeProlongation();
	solveCoarseProblem();
}

Result Solver::solve(int numModes)
{
	requireMesh("solve");
	double hierarchySeconds = 0.0;
	if (hierarchy_.numModes != numModes || !hierarchy_.hasSamples()) {
		buildHierarchy(numModes);
		hierarchySeconds = hierarchySeconds_;
	}
	else {
		if (!hierarchy_.hasProlongation()) computeProlongation();
		if (!hierarchy_.hasCoarseSolution()) solveCoarseProblem();
	}
	Result result = hierarchicalSubspaceIteration(operators_, hierarchy_, numModes, options_);
	result.diagnostics.hierarchySeconds = hierarchySeconds;
	return result;
}

} // namespace hsim
