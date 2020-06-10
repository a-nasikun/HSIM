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