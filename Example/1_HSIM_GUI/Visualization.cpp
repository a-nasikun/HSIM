#include "Visualization.h"

#include <igl/jet.h>

void showScalarField(igl::opengl::glfw::Viewer& viewer, const Eigen::VectorXd& field, bool normalize)
{
	Eigen::MatrixXd colors;
	igl::jet(field, normalize, colors);
	viewer.data().set_colors(colors);
}

void showSamples(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::VectorXi& samples,
	const std::vector<int>& levelSizes)
{
	const int numLevels = static_cast<int>(levelSizes.size());
	if (numLevels == 0) return;

	// Qualitative palette for up to 11 levels, jet colour map beyond.
	Eigen::MatrixXd palette;
	if (numLevels <= 11) {
		palette.resize(11, 3);
		palette << 228, 26, 28, 55, 126, 184, 77, 225, 74, 152, 78, 163, 255, 127, 0, 255, 255, 51, 166, 86, 40, 247, 129, 191,
			153, 153, 153, 10, 10, 10, 178, 223, 138;
		palette /= 255.0;
	}
	else {
		igl::jet(Eigen::VectorXd::LinSpaced(numLevels, 0, numLevels - 1), true, palette);
	}

	clearPoints(viewer);
	// Samples of the coarsest level get the first colour, those added on level k the colour of index k.
	for (int i = 0; i < levelSizes.back(); ++i) viewer.data().add_points(V.row(samples(i)), palette.row(0));
	for (int k = numLevels - 1; k > 0; --k)
		for (int i = levelSizes[static_cast<size_t>(k)]; i < levelSizes[static_cast<size_t>(k) - 1]; ++i)
			viewer.data().add_points(V.row(samples(i)), palette.row(k));
}

void clearPoints(igl::opengl::glfw::Viewer& viewer) { viewer.data().points.resize(0, 6); }
