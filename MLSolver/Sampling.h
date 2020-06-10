#ifndef SAMPLING_H
#define SAMPLING_H

#include <igl/opengl/glfw/Viewer.h>

#include "Utility.h"

using namespace std;

namespace MuLES
{
	/* Enum for different sampling method */
	enum SamplingType {
		Sample_Farthest_Point = 0,
		Sample_Random = 1
	};

	void constructRandomSample(Eigen::VectorXi &Sample, const Eigen::MatrixXd &V, int n);
	void ComputeDijkstra(const Eigen::MatrixXd &V, const int Si, const vector<set<int>> &AdM, Eigen::VectorXd &D);
	void constructFarthestPointSample(const Eigen::MatrixXd &V, const std::vector<std::set<int>> &AdM, const int &numSamples, Eigen::VectorXi &Samples);
	void constructFarthestPointSample(const Eigen::MatrixXd &V, const std::vector<std::set<int>> &AdM, Eigen::VectorXi &Samples, vector<int>& SubspaceSize, vector<Eigen::VectorXd>& geoDist);
}
#endif // !SAMPLING_H

