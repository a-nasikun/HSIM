/**
 * @file prolongation.h
 * @brief Prolongation operators between consecutive levels of the hierarchy.
 *
 * The basis function of sample i on level k is the radial function phi(d) = 2 (d/r_k)^3 - 3 (d/r_k)^2 + 1 of the
 * graph distance d to the sample, supported on d <= r_k. U_k contains these functions evaluated at the degrees of
 * freedom of level k-1 (the vertices for k = 0); its rows are normalised to sum to one (partition of unity).
 */
#pragma once

#include "common.h"

#include <vector>

namespace hsim {

class Adjacency;
struct Options;

/// Support radius r with pi r^2 = areaFactor * supportSize * area / numSamples.
[[nodiscard]] double supportRadius(double area, int numSamples, double supportSize, double areaFactor);

/// Prolongation operator U_k of level k: an (n x m_0) matrix for k = 0 and an (m_{k-1} x m_k) matrix otherwise.
/// The rows are not yet normalised (see makePartitionOfUnity()).
/// @param samples    nested sample vertices (Samples::vertices)
/// @param levelSizes m_0, ..., m_{L-1}
/// @param radius     support radius r_k
/// @param numThreads OpenMP threads (0: default)
[[nodiscard]] SparseMatrix radialBasisLevel(const Eigen::MatrixXd& V, const Adjacency& adjacency, const Eigen::VectorXi& samples,
                                            const std::vector<int>& levelSizes, int k, double radius, int numThreads = 0);

/// Scales every row of U to unit sum. Rows whose sum vanishes (|sum| < 10 eps) are left unchanged.
void makePartitionOfUnity(SparseMatrix& U);

/// All prolongation operators U_0, ..., U_{L-1} (normalised) and their support radii.
[[nodiscard]] std::vector<SparseMatrix> buildProlongation(const Eigen::MatrixXd& V, const Adjacency& adjacency, double area,
                                                          const Eigen::VectorXi& samples, const std::vector<int>& levelSizes,
                                                          const Options& options, std::vector<double>* radii = nullptr);

/// The product P = U_0 U_1 ... U_{L-1} (n x m_{L-1}) mapping coarsest-level coefficients to vertex values.
[[nodiscard]] SparseMatrix composeProlongation(const std::vector<SparseMatrix>& U, Eigen::Index numVertices);

} // namespace hsim
