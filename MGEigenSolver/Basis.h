#ifndef BASIS_H
#define BASIS_H

#include "Utility.h"

using namespace std;

/* [DIJKSTRA-based BASIS] */
/* Computing Dijkstra Distance (single input, distance vector D initialized inside, stops when reaching a certain distance **/
void ComputeDijkstraCompact(const Eigen::MatrixXd &V, const int s1, const vector<set<int>> &AdM,
	Eigen::VectorXd &D, const double a, const double b, const double nDist, const int col, const int sampleSize, int probType,
	bool isClamped, int axis, double offset,
	vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter);	

/* [DIJKSTRA-based BASIS] 
** Computing Dijkstra Distance (single input, distance vector D initialized inside, stops when reaching a certain distance.
** This is for the coarse levels (i.e. level 1 on ward */
void ComputeDijkstraCompact(const Eigen::MatrixXd &V, const int s1, const vector<set<int>> &AdM,
	Eigen::VectorXd &D, const double a, const double b, const double nDist, const int col, int level, Eigen::VectorXi Sample, const vector<int>& SubspaceSize, int probType,
	bool isClamped, int axis, double offset,
	vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter);

#endif // !BASIS_H

