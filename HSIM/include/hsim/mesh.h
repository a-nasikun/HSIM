/**
 * @file mesh.h
 * @brief Triangle-mesh input, connectivity, and the discrete Laplace--Beltrami operator.
 */
#pragma once

#include "common.h"

#include <string>
#include <vector>

namespace hsim {

/// Reads a triangle mesh (OBJ, OFF, PLY, STL, ...; polygons are triangulated).
/// @throws std::runtime_error if the file cannot be read or contains no triangles.
void readMesh(const std::string& path, Eigen::MatrixXd& V, Eigen::MatrixXi& F);

/// Checks that (V, F) is a non-empty triangle mesh in R^3 with finite coordinates and valid indices.
/// @throws std::invalid_argument otherwise.
void validateMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

/// Translates and uniformly scales V such that its bounding box is centred at the origin and its longest side has
/// length `diameter`.
void normalizeMesh(Eigen::MatrixXd& V, double diameter = 1.0);

/// Vertex-to-vertex adjacency of a mesh in compressed (CSR) form; the neighbours of every vertex are sorted.
class Adjacency {
public:
	/// Read-only view of the neighbours of one vertex.
	struct Range {
		const int* first;
		const int* last;
		const int* begin() const { return first; }
		const int* end() const { return last; }
		Eigen::Index size() const { return last - first; }
	};

	Adjacency() = default;

	/// Builds the adjacency of the vertices 0, ..., numVertices-1 from the faces F (any polygon size).
	Adjacency(const Eigen::MatrixXi& F, Eigen::Index numVertices);

	Eigen::Index numVertices() const { return static_cast<Eigen::Index>(offsets_.empty() ? 0 : offsets_.size() - 1); }
	/// Position of the first neighbour of `vertex` in the flat neighbour array (see edgeLengths()).
	size_t offset(Eigen::Index vertex) const { return static_cast<size_t>(offsets_[static_cast<size_t>(vertex)]); }

	/// Euclidean length of every adjacency entry, in the order of the flat neighbour array: the length of the edge to the
	/// j-th neighbour of v is edgeLengths(V)[offset(v) + j].
	std::vector<double> edgeLengths(const Eigen::MatrixXd& V) const;

	Range neighbors(Eigen::Index vertex) const
	{
		return {neighbors_.data() + offsets_[static_cast<size_t>(vertex)],
		        neighbors_.data() + offsets_[static_cast<size_t>(vertex) + 1]};
	}

private:
	std::vector<int> offsets_;
	std::vector<int> neighbors_;
};

/// Discretisation of the Laplace--Beltrami eigenproblem S x = lambda M x with linear finite elements.
struct Operators {
	SparseMatrix stiffness; ///< S: cotangent Laplacian, symmetric positive semi-definite
	SparseMatrix mass; ///< M: lumped (Voronoi) mass matrix, diagonal positive definite
	double area = 0.0; ///< total surface area
};

/// Assembles the stiffness and mass matrices of a triangle mesh.
[[nodiscard]] Operators assembleOperators(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

} // namespace hsim
