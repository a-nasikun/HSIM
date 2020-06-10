#include "Sampling.h"

#include <set>
#include <queue>
#include <time.h>
#include <random>
#include <chrono>
#include <iostream>

/* Construct random n number of samples */
void constructRandomSample(Eigen::VectorXi &Sample, const Eigen::MatrixXd &V, int n)
{
	std::set<int>						sample;
	std::random_device					rd;
	std::default_random_engine			generator(rd());
	std::uniform_int_distribution<long> distribution(0, V.rows());

	do {
		int newSample = distribution(generator);
		sample.insert(newSample);
	} while (sample.size() < n);

	Sample.resize(n);

	int i = 0;
	for (auto s : sample) {
		Sample(i++) = s;
	}
}

/* Construct Voxel-based sampling */
void constructVoxelSample(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const int &n, Eigen::VectorXi &Sample)
{
	// Initialization of computing boundary of each box	
	Eigen::Vector3d minV(V.col(0).minCoeff(), V.col(1).minCoeff(), V.col(2).minCoeff());
	Eigen::Vector3d maxV(V.col(0).maxCoeff(), V.col(1).maxCoeff(), V.col(2).maxCoeff());
	Eigen::Vector3d range, boxDist;

	range = maxV - minV;
	boxDist = range / (double)n;
	set<int>		locSample;


	// Constructing tensor (n x n x n x (box-members)) for sample candidates
	vector<int>							v1;
	vector<vector<int>>					v2;
	vector<vector<vector<int>>>			v3;
	vector<vector<vector<vector<int>>>> v4;

	// Creating empty vectors
	for (int i = 0; i < n; i++) { v2.push_back(v1); }
	for (int i = 0; i < n; i++) { v3.push_back(v2); }
	for (int i = 0; i < n; i++) { v4.push_back(v3); }


	//Put each vertex to its corresponding box
	for (int i = 0; i < V.rows(); i++) {
		int xId, yId, zId;
		xId = (int)abs(floor((V(i, 0) - minV(0)) / boxDist(0)));
		yId = (int)abs(floor((V(i, 1) - minV(1)) / boxDist(1)));
		zId = (int)abs(floor((V(i, 2) - minV(2)) / boxDist(2)));
		if (xId >(n - 1)) xId = n - 1;
		if (yId >(n - 1)) yId = n - 1;
		if (zId > (n - 1)) zId = n - 1;
		v4[xId][yId][zId].push_back(i);
	}

	// Populate the boxes of samples
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				// Only cares for non-empty boxes
				if (v4[i][j][k].size() > 0) {
					Eigen::Vector3d boxMin(minV(0) + i*boxDist(0), minV(1) + j*boxDist(1), minV(2) + k*boxDist(2));

					double	lambda = 3.0;
					double	minVal = numeric_limits<double>::infinity();
					int		minDistId;

					// Selecting vertex that closest to the center of each box/container;
					for (int l = 0; l < v4[i][j][k].size(); l++) {
						double	dist = (V.row(v4[i][j][k][l]) - (boxMin+boxDist/2.0).transpose()).norm();
						//VtoVDist(V.row(v4[i][j][k][l]), (boxMin + boxDist / 2.0), dist);
						//double pDist = lambda * exp(-lambda*dist);
						if (dist < minVal) {
							minVal = dist;
							minDistId = v4[i][j][k][l];
						}
					}
					locSample.insert(minDistId);
				}
			}
		}
	}
	Sample.resize(locSample.size());
	int i = 0;
	for (int s : locSample) {
		Sample(i++) = s;
		//viewer.data.add_points(V.row(s), Eigen::Vector3d(1.0, 0.0, 0.0).transpose());
	}
	printf("%d-box produces %d samples.\n", n, locSample.size());
}

void ComputeDijkstra(const Eigen::MatrixXd &V, const int Si, const vector<set<int>> &AdM, Eigen::VectorXd &D)
{
	// Define containers for (1) visited nodes and (2) priority queue
	priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistPQueue;
	int insertCounter = 0;

	// Computing distance for initial sample points S
	Eigen::VectorXd vRow1 = V.block<1, 3>(Si, 0);
	D(Si) = 0.0f;
	VertexPair vp{ Si,D(Si) };
	DistPQueue.push(vp);
	insertCounter++;

	double distFromCenter = numeric_limits<double>::infinity();
	// For other vertices in mesh
	while (DistPQueue.size()>0)
	{
		VertexPair vp1 = DistPQueue.top();
		Eigen::VectorXd vRow1 = V.block<1, 3>(vp1.vId, 0);
		DistPQueue.pop();

		for (set<int>::iterator it = AdM[vp1.vId].begin(); it != AdM[vp1.vId].end(); ++it) {
			double dist;
			Eigen::VectorXd vRow2 = V.block<1, 3>(*it, 0);
			dist = (vRow2 - vRow1).norm();
			//VtoVDist(vRow1, vRow2, dist);
			double tempDist = vp1.distance + dist;

			//double tempDist = (V.row(Si) - V.row(*it)).norm();

			if (tempDist < D(*it)) {
				D(*it) = tempDist;
				distFromCenter = tempDist;
				VertexPair vp2{ *it,tempDist };
				DistPQueue.push(vp2);
				insertCounter++;
			}
		}
	}
}

void constructFarthestPointSample(const Eigen::MatrixXd &V, const std::vector<std::set<int>> &AdM, const int &numSamples, Eigen::VectorXi &Samples)
{
	Samples.resize(numSamples);
	Eigen::VectorXd D;
	//Eigen::VectorXi Si;
	D.resize(V.rows());

	// Initialize the value of D
	for (int i = 0; i < V.rows(); i++) {
		D(i) = numeric_limits<double>::infinity();
	}

	srand(time(NULL));
	Samples(0) = rand() % V.rows();
	//Si = Samples.head(1);

	ComputeDijkstra(V, Samples(0), AdM, D);
	Eigen::VectorXi::Index maxIndex1;
	D.maxCoeff(&maxIndex1);
	Samples(0) = maxIndex1;

	for (int i = 1; i < numSamples; i++) {
		Eigen::VectorXi::Index maxIndex;
		ComputeDijkstra(V, Samples(i - 1), AdM, D);
		D.maxCoeff(&maxIndex);
		Samples(i) = maxIndex;
	}
}

void constructFarthestPointSample(const Eigen::MatrixXd &V, const std::vector<std::set<int>> &AdM, Eigen::VectorXi &Samples, vector<int>& SubspaceSize, vector<Eigen::VectorXd>& geoDist)
{
	const int level = SubspaceSize.size();
	Samples.resize(SubspaceSize[0]);
	Eigen::VectorXd D;
	D.resize(V.rows());

	// Initialize the value of D
	D.setConstant(numeric_limits<double>::infinity());

	/* Random number generator */
	std::random_device rd;								// Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd());								// Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> dist(0, V.rows());	// From 0 to F.rows()-1
	//Samples(0) = dist(gen);	
	Samples(0) = 0;
	
	Eigen::VectorXi::Index maxIndex;

	/* Samples on the coarsest level */
	printf("Sample %d: %d \n", level-1, SubspaceSize[level-1]);
	for (int i = 1; i < SubspaceSize[level-1]; ++i) {
		ComputeDijkstra(V, Samples(i-1), AdM, D);
		D.maxCoeff(&maxIndex);
		Samples(i) = maxIndex;
	}
	geoDist[level-1] = D; 
	printf("Range of D[%d]: %.2f -> %.2f \n", level - 1, D.minCoeff(), D.maxCoeff());

	/* For samples on coarser to finer level*/	
	for (int j = (level-2); j >= 0; --j)
	{
		printf("Sample %d: %d [%d-%d].\n", j, SubspaceSize[j]-SubspaceSize[j+1], SubspaceSize[j], SubspaceSize[j+1]);
		for (int i = SubspaceSize[j+1]; i < SubspaceSize[j]; ++i) {
			ComputeDijkstra(V, Samples(i-1), AdM, D);
			D.maxCoeff(&maxIndex);
			Samples(i) = maxIndex;
		}
		geoDist[j] = D;
		printf("Range of D[%d]: %.2f -> %.2f \n", j, D.minCoeff(), D.maxCoeff());
	}
}