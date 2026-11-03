#include "hsim/mesh.h"

#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/read_triangle_mesh.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace hsim {

void readMesh(const std::string& path, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	V.resize(0, 0);
	F.resize(0, 0);
	if (!igl::read_triangle_mesh(path, V, F)) throw std::runtime_error("readMesh: cannot read the mesh file '" + path + "'");
	if (V.rows() == 0 || F.rows() == 0) throw std::runtime_error("readMesh: the mesh file '" + path + "' contains no triangles");
}

void validateMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	if (V.rows() == 0 || F.rows() == 0) throw std::invalid_argument("validateMesh: the mesh is empty");
	if (V.cols() != 3) throw std::invalid_argument("validateMesh: vertex coordinates must have 3 columns");
	if (F.cols() != 3) throw std::invalid_argument("validateMesh: only triangle meshes (3 indices per face) are supported");
	if (!V.allFinite()) throw std::invalid_argument("validateMesh: vertex coordinates must be finite");
	if (F.minCoeff() < 0 || F.maxCoeff() >= V.rows()) throw std::invalid_argument("validateMesh: face index out of range");
}

void normalizeMesh(Eigen::MatrixXd& V, double diameter)
{
	if (V.rows() == 0) return;
	const Eigen::RowVectorXd minV = V.colwise().minCoeff();
	const Eigen::RowVectorXd maxV = V.colwise().maxCoeff();
	const double longestSide = (maxV - minV).maxCoeff();
	if (!(longestSide > 0.0)) throw std::invalid_argument("normalizeMesh: degenerate bounding box");
	const Eigen::RowVectorXd centre = 0.5 * (minV + maxV);
	V = ((V.rowwise() - centre) * (diameter / longestSide)).eval();
}

Adjacency::Adjacency(const Eigen::MatrixXi& F, Eigen::Index numVertices)
{
	// Every pair of vertices of a face is adjacent (for triangles: the three edges).
	std::vector<std::pair<int, int>> edges;
	edges.reserve(static_cast<size_t>(F.rows() * F.cols() * (F.cols() - 1)));
	for (Eigen::Index f = 0; f < F.rows(); ++f)
		for (Eigen::Index j = 0; j < F.cols(); ++j)
			for (Eigen::Index k = j + 1; k < F.cols(); ++k) {
				edges.emplace_back(F(f, j), F(f, k));
				edges.emplace_back(F(f, k), F(f, j));
			}
	std::sort(edges.begin(), edges.end());
	edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

	offsets_.assign(static_cast<size_t>(numVertices) + 1, 0);
	neighbors_.resize(edges.size());
	for (size_t e = 0; e < edges.size(); ++e) {
		++offsets_[static_cast<size_t>(edges[e].first) + 1];
		neighbors_[e] = edges[e].second;
	}
	for (size_t v = 0; v < static_cast<size_t>(numVertices); ++v)
		offsets_[v + 1] += offsets_[v];
}

std::vector<double> Adjacency::edgeLengths(const Eigen::MatrixXd& V) const
{
	std::vector<double> lengths(neighbors_.size());
	for (Eigen::Index v = 0; v < numVertices(); ++v)
		for (size_t e = offset(v); e < offset(v + 1); ++e)
			lengths[e] = (V.row(neighbors_[e]) - V.row(v)).norm();
	return lengths;
}

Operators assembleOperators(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	validateMesh(V, F);
	Operators op;
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, op.mass);
	igl::cotmatrix(V, F, op.stiffness);
	op.stiffness = -op.stiffness; // libigl's cotangent matrix is negative semi-definite
	Eigen::VectorXd doubleArea;
	igl::doublearea(V, F, doubleArea);
	op.area = doubleArea.sum() / 2.0;
	return op;
}

} // namespace hsim
