/**
 * @file common.h
 * @brief Basic types shared by all HSIM modules.
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace hsim {

/// Sparse matrix type used throughout the library (column-major, double precision).
using SparseMatrix = Eigen::SparseMatrix<double>;

} // namespace hsim
