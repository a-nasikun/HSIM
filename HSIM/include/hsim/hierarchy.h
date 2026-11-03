/**
 * @file hierarchy.h
 * @brief The multilevel hierarchy: samples, prolongation operators and the coarsest-level eigenpairs.
 */
#pragma once

#include "dense_eigensolver.h"
#include "mesh.h"
#include "sampling.h"

#include <vector>

namespace hsim {

/// Everything that is computed once per mesh and number of sought modes.
struct Hierarchy {
	int numModes = 0; ///< number of eigenpairs the hierarchy was built for
	std::vector<int> levelSizes; ///< m_0 > ... > m_{L-1}
	Samples samples; ///< nested farthest-point samples
	std::vector<double> supportRadii; ///< r_0, ..., r_{L-1}
	std::vector<SparseMatrix> prolongation; ///< U_0, ..., U_{L-1}
	SparseMatrix basis; ///< P = U_0 ... U_{L-1}
	DenseEigenpairs coarse; ///< all eigenpairs of (P^T S P, P^T M P)

	int numLevels() const { return static_cast<int>(levelSizes.size()); }
	bool hasSamples() const { return samples.vertices.size() > 0; }
	bool hasProlongation() const { return !prolongation.empty() && basis.size() > 0; }
	bool hasCoarseSolution() const { return coarse.success; }
};

/// Step 1: level sizes and nested samples.
void computeSamples(Hierarchy& hierarchy, const Eigen::MatrixXd& V, const Adjacency& adjacency, int numModes,
                    const Options& options);

/// Step 2: prolongation operators (requires step 1).
void computeProlongation(Hierarchy& hierarchy, const Eigen::MatrixXd& V, const Adjacency& adjacency, const Operators& operators,
                         const Options& options);

/// Step 3: eigenpairs of the problem restricted to the coarsest level (requires step 2). If the dense solver fails,
/// it is retried with an increasingly regularised stiffness matrix.
/// @throws std::runtime_error if all attempts fail.
void solveCoarseProblem(Hierarchy& hierarchy, const Operators& operators, const Options& options);

/// Vertex values of basis function `index` of level `level`, i.e. U_0 ... U_{level-1} U_level e_index.
/// @throws std::out_of_range for invalid indices, std::invalid_argument if the prolongation is missing.
[[nodiscard]] Eigen::VectorXd basisFunction(const Hierarchy& hierarchy, int level, int index);

/// Steps 1-3.
[[nodiscard]] Hierarchy buildHierarchy(const Eigen::MatrixXd& V, const Adjacency& adjacency, const Operators& operators,
                                       int numModes, const Options& options);

} // namespace hsim
