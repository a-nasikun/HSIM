#include "Basis.h"

#include <set>
#include <queue>
#include <vector>
#include <iostream>

#include <omp.h>			// OpenMP, parallelizing basis construction

#include <Eigen/Eigen>

/* [DIJKSTRA-based BASIS] */
// Computing Dijkstra Distance (sing input, D initialized inside, stops when reaching a certain distance, 
// and used for Basis Construction), COMPACT Version
void ComputeDijkstraCompact(const Eigen::MatrixXd &V, const int s1, const vector<set<int>> &AdM,
	Eigen::VectorXd &D, const double a, const double b, const double nDist, const int col, const int sampleSize, int probType,
	vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter)
{
	//vector<VertexPair> container;
	//container.reserve(V.rows());
	priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistPQueue;
	
	int elCounter = 1;

	// Computing distance for initial sample points S
	D(s1) = 0.0f;
	VertexPair vp{ s1,D(s1) };
	DistPQueue.push(vp);

	double distFromCenter = numeric_limits<double>::infinity();

	// For other vertices in mesh
	do {
		if (DistPQueue.size() == 0) break;
		VertexPair vp1 = DistPQueue.top();
		distFromCenter = vp1.distance;
		DistPQueue.pop();

		// Updating the distance for neighbors of vertex of lowest distance in priority queue
		auto const& elem = AdM[vp1.vId];
		for (auto it = elem.begin(); it != elem.end(); ++it) {
			double dist;
			/* Regular Dikjstra */
			VtoVDist(V.row(vp1.vId), V.row(*it), dist);
			double tempDist = vp1.distance + dist;

			/* Correct distance using Euclidean Distance */
			//double tempDist = (V.row(s1) - V.row(*it)).norm();

			/* updating the distance */
			if (tempDist < D(*it)) {
				D(*it) = tempDist;
				VertexPair vp2{ *it,tempDist };
				DistPQueue.push(vp2);
			}
		}
	} while (distFromCenter <= nDist);

	// Constructing a sparse matrix based on the triplet

	for (int i = 0; i < D.size(); i++) {
		double aVal = D(i);
		if (aVal <= nDist) {
			/* Linear Mapping */
			//double val = 1.0 - aVal/nDist;
			//UTriplet.push_back(Eigen::Triplet<double>(i, col, val));
	
			/* Polynomial mapping (3rd Order) */
			double val = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;					// face
			if(probType==0)
				UTriplet.push_back(Eigen::Triplet<double>(i, col, val));

			else if (probType == 1)												// tet-mesh
			{
				for (int k = 0; k < 3; k++)
					UTriplet.push_back(Eigen::Triplet<double>(3 * i + k, 3 * col + k, val));
			}
		}
	
		// For the next samples, reset to infinity
		D(i) = numeric_limits<double>::infinity();
	}
}

void ComputeDijkstraCompact(const Eigen::MatrixXd &V, const int s1, const vector<set<int>> &AdM,
	Eigen::VectorXd &D, const double a, const double b, const double nDist, const int col, int level, Eigen::VectorXi Sample, const vector<int>& SubspaceSize, int probType,
	vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter)
{
	priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistPQueue;

	int elCounter = 1;

	// Computing distance for initial sample points S
	D(s1) = 0.0f;
	VertexPair vp{ s1,D(s1) };
	DistPQueue.push(vp);

	double distFromCenter = numeric_limits<double>::infinity();

	// For other vertices in mesh
	do {
		if (DistPQueue.size() == 0) break;
		VertexPair vp1 = DistPQueue.top();
		distFromCenter = vp1.distance;
		DistPQueue.pop();

		// Updating the distance for neighbors of vertex of lowest distance in priority queue
		auto const& elem = AdM[vp1.vId];
		for (auto it = elem.begin(); it != elem.end(); ++it) {
			double dist;

			/* Regular dijkstra */
			//VtoVDist(V.row(vp1.vId), V.row(*it), dist);
			dist = (V.row(*it)- V.row(vp1.vId)).norm();
			double tempDist = vp1.distance + dist;

			/* Correct distance using Euclidean Distance */
			//double tempDist = (V.row(s1) - V.row(*it)).norm();

			/* updating the distance */
			if (tempDist < D(*it)) {
				D(*it) = tempDist;
				VertexPair vp2{ *it,tempDist };
				DistPQueue.push(vp2);
			}
		}
	} while (distFromCenter <= nDist);

	for (int i = 0; i < SubspaceSize[level-1]; i++) {
		double aVal = D(Sample(i));
		if (aVal <= nDist) {
			/* Linear Mapping */
			//double val = 1.0 - aVal/nDist;
			//UTriplet.push_back(Eigen::Triplet<double>(i, col, val));

			/* Polynomial mapping (3rd Order) */
			double val = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;
			if (probType == 0) {
				UTriplet.push_back(Eigen::Triplet<double>(i, col, val));
			}
			else if (probType == 1) {
				for (int k = 0; k < 3; k++) {
					UTriplet.push_back(Eigen::Triplet<double>(3 * i + k, 3 * col + k, val));
				}
			}
		}
	}

	D.setConstant(numeric_limits<double>::infinity());

}