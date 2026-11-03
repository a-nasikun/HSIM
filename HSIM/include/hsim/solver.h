/**
 * @file solver.h
 * @brief High-level interface: hsim::eigs() for one-shot use, hsim::Solver for reuse and step-by-step use.
 */
#pragma once

#include "subspace_iteration.h"

namespace hsim {

/// Computes the numModes smallest eigenpairs of the Laplace--Beltrami operator of the triangle mesh (V, F).
///
/// Example:
/// @code
///   hsim::Options options;
///   options.tolerance = 1e-4;
///   hsim::Result result = hsim::eigs(V, F, 100, options);
/// @endcode
Result eigs(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes, const Options& options = {});

/// Stateful solver that keeps the operators and the hierarchy of one mesh.
///
/// Typical use: setMesh(), then solve(). The individual stages (computeSamples(), computeProlongation(),
/// solveCoarseProblem()) can also be run one at a time, e.g. for visualisation.
class Solver {
public:
	explicit Solver(Options options = {});

	/// Sets the mesh, assembles the operators and the adjacency, and discards any previous hierarchy.
	void setMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

	/// Changes the options. Hierarchy-related options take effect when the next hierarchy stage is computed.
	void setOptions(const Options& options);

	/// Builds the complete hierarchy for numModes eigenpairs.
	void buildHierarchy(int numModes);

	/// Computes numModes eigenpairs. The hierarchy is (re)built if it is missing, incomplete, or was built for a
	/// different number of modes.
	Result solve(int numModes);

	/* Individual stages of buildHierarchy() */
	void computeSamples(int numModes);
	void computeProlongation();
	void solveCoarseProblem();

	/* Accessors */
	bool hasMesh() const { return V_.rows() > 0; }
	const Options& options() const { return options_; }
	const Eigen::MatrixXd& vertices() const { return V_; }
	const Eigen::MatrixXi& faces() const { return F_; }
	const Operators& operators() const { return operators_; }
	const Adjacency& adjacency() const { return adjacency_; }
	const Hierarchy& hierarchy() const { return hierarchy_; }

private:
	void requireMesh(const char* stage) const;

	Options options_;
	Eigen::MatrixXd V_;
	Eigen::MatrixXi F_;
	Operators operators_;
	Adjacency adjacency_;
	Hierarchy hierarchy_;
	double hierarchySeconds_ = 0.0;
};

} // namespace hsim
