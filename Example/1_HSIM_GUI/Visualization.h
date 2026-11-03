// Visualisation helpers of the GUI example (libigl viewer).
#pragma once

#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Core>

#include <vector>

/// Colours the mesh by a scalar per-vertex field (jet colour map, optionally normalised to [min, max]).
void showScalarField(igl::opengl::glfw::Viewer& viewer, const Eigen::VectorXd& field, bool normalize = true);

/// Shows the nested samples as points, one colour per level.
void showSamples(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::VectorXi& samples,
	const std::vector<int>& levelSizes);

/// Removes all points.
void clearPoints(igl::opengl::glfw::Viewer& viewer);
