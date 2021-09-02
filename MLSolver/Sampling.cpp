#include "Sampling.h"

#include <set>
#include <queue>
#include <time.h>
#include <random>
#include <chrono>
#include <iostream>

namespace HSIM
{
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
		while (DistPQueue.size() > 0)
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
		Samples(0) = dist(gen);	
		//Samples(0) = 0;

		Eigen::VectorXi::Index maxIndex;

		/* Samples on the coarsest level */
		printf("____> Sample %d: %d \n", level, SubspaceSize[level - 1]);
		for (int i = 1; i < SubspaceSize[level - 1]; ++i) {
			ComputeDijkstra(V, Samples(i - 1), AdM, D);
			D.maxCoeff(&maxIndex);
			Samples(i) = maxIndex;
		}
		geoDist[level - 1] = D;
		//printf("Range of D[%d]: %.2f -> %.2f \n", level - 1, D.minCoeff(), D.maxCoeff());

		/* For samples on coarser to finer level*/
		for (int j = (level - 2); j >= 0; --j)
		{
			printf("____> Sample %d: %d.\n", j+1, SubspaceSize[j]);
			for (int i = SubspaceSize[j + 1]; i < SubspaceSize[j]; ++i) {
				ComputeDijkstra(V, Samples(i - 1), AdM, D);
				D.maxCoeff(&maxIndex);
				Samples(i) = maxIndex;
			}
			geoDist[j] = D;
			//printf("Range of D[%d]: %.2f -> %.2f \n", j, D.minCoeff(), D.maxCoeff());
		}

		printf("____> Sample %d: %d \n", 0, V.rows());
	}

	void constructPoissonDiskSample(const Eigen::MatrixXd& V, const std::vector<std::set<int>>& AdM, const int& sampleSize, const double& avgEdgeLength, Eigen::VectorXi& Sample)
	{
		int nBox;
		printf("We have expected samples of %d \n", sampleSize);
		if (sampleSize < 1 || sampleSize>V.rows()) {
			cout << "Error! Sample cannot be 0 or larger than the number of vertices." << endl;
			return;
		}
		else if (sampleSize <= 300) {
			nBox = 8;
		}
		else if (sampleSize > 300 && sampleSize <= 500) {
			nBox = 10;
		}
		else if (sampleSize > 500 && sampleSize <= 2000) {
			nBox = 13;
		}
		else if (sampleSize > 2000 && sampleSize <= 3000) {
			nBox = 15;
		}
		else if (sampleSize > 3000 && sampleSize <= 6000) {
			nBox = 18;
		}
		else if (sampleSize > 6000 && sampleSize <= 10000) {
			nBox = 23;
		}
		else if (sampleSize > 10000 && sampleSize <= 30000) {
			nBox = 30;
		}
		else if (sampleSize > 30000 && sampleSize <= 75000) {
			nBox = 40;
		}
		else {
			nBox = 55;
			cout << "Warning! Are you sure to have this huge amount of sampling?." << endl;
			cout << "Our approach won't have much benefit with this large sampling, though." << endl;
		}
		/* Initialization of computing boundary of each box	 */
		Eigen::Vector3d minV(V.col(0).minCoeff(), V.col(1).minCoeff(), V.col(2).minCoeff());
		Eigen::Vector3d maxV(V.col(0).maxCoeff(), V.col(1).maxCoeff(), V.col(2).maxCoeff());
		Eigen::Vector3d range, boxDist;
		Eigen::Vector3i nn;

		range = maxV - minV;
		double minRange = range.minCoeff();

		for (int i = 0; i < 3; i++) {
			boxDist(i) = minRange / (double)nBox;
			nn(i) = (int)ceil(range(i) / boxDist(i));
		}

		//boxDist = range / (double)n;

		set<int>		locSample;
		const int		SamplePerBox = 10;
		//double			radius = 0.50 * 1.0 / 3.0 * (boxDist(0) + boxDist(1) + boxDist(2));
		double			radius = sqrt((double)V.rows() / (2.5 * (double)sampleSize)) * avgEdgeLength;

		/* Container for boxes */
		priority_queue<BoxStruct, std::vector<BoxStruct>, std::greater<BoxStruct>>	UnvisitedBoxQueue;
		list<int>																	VisitedEmptyBoxList;

		/* Constructing tensor (n x n x n x (box-members)) for sample candidates */
		list<int>							v1;
		vector<list<int>>					v2;
		vector<vector<list<int>>>			v3;
		vector<vector<vector<list<int>>>>	v4;

		/* Instatiating the tensor with empty elements */
		v2.reserve(nn(2));	v3.reserve(nn(1));	v4.reserve(nn(0));
		for (int i = 0; i < nn(2); i++) { v2.push_back(v1); }
		for (int i = 0; i < nn(1); i++) { v3.push_back(v2); }
		for (int i = 0; i < nn(0); i++) { v4.push_back(v3); }

		/* Put each vertex to its corresponding box */
		for (int vId = 0; vId < V.rows(); vId++) {
			int xId, yId, zId;
			xId = (int)abs(floor((V(vId, 0) - minV(0)) / boxDist(0)));
			yId = (int)abs(floor((V(vId, 1) - minV(1)) / boxDist(1)));
			zId = (int)abs(floor((V(vId, 2) - minV(2)) / boxDist(2)));
			if (xId > (nn(0) - 1)) xId = nn(0) - 1;
			if (yId > (nn(1) - 1)) yId = nn(1) - 1;
			if (zId > (nn(2) - 1)) zId = nn(2) - 1;

			v4[xId][yId][zId].push_back(vId);
		}

		// Randomizer for large number (LONG)
		std::random_device					rd;
		std::default_random_engine			generator(rd());
		int									trimCounter = 0;

		/* FOR TESTING ONLY*/
		int poolCounter = 0;
		/* END OF TESTING */

		// Populating Boxes with 10 random vertices
		for (int iIdx = 0; iIdx < nn(0); iIdx++) {
			for (int jIdx = 0; jIdx < nn(1); jIdx++) {
				for (int kIdx = 0; kIdx < nn(2); kIdx++) {
					Eigen::Vector3d		boxMin(minV(0) + iIdx * boxDist(0), minV(1) + jIdx * boxDist(1), minV(2) + kIdx * boxDist(2));

					/* IF there are more samples on a box then required, select randomly 10 of them; */
					int boxSize = v4[iIdx][jIdx][kIdx].size();
					if (boxSize == 0) {
						// ID(a,b,c) = a*(y*z) + b*(z) + c
						VisitedEmptyBoxList.push_back(iIdx * nn(2) * nn(1) + jIdx * nn(2) + kIdx);
					}
					else
					{	// Work on NON-EMPTY boxes only
						double randDist = (double)(rand() % 100) / (double)10;
						BoxStruct curBox{ iIdx * nn(2) * nn(1) + jIdx * nn(2) + kIdx, randDist };
						UnvisitedBoxQueue.push(curBox);

						/* If there are more elements than required, randomly discard the rest */
						if (boxSize > SamplePerBox) {
							set<int> vertexToDel;
							srand(time(NULL));
							do {
								std::uniform_int_distribution<long> distribution(0, boxSize);
								int vSel = distribution(generator);
								vertexToDel.insert(vertexToDel.end(), vSel);
							} while (vertexToDel.size() <= (boxSize - SamplePerBox));

							int l = 0;
							for (list<int>::iterator it = v4[iIdx][jIdx][kIdx].begin(); it != v4[iIdx][jIdx][kIdx].end(); ) {
								if (vertexToDel.find(l) != vertexToDel.end()) {
									it = v4[iIdx][jIdx][kIdx].erase(it);
								}
								else {
									++it;
								}
								l++;
							}
							trimCounter++;
						} // End of vertex(-ices) deletion
					}
					/* FOR TESTING ONLY*/
					poolCounter += v4[iIdx][jIdx][kIdx].size();
					/* END OF TESTING */
				}
			}
		} // End of Populating the boxes with samples POOL

		//printf("Has set up %d samplepool. \n", poolCounter);

		/* Getting samples from POOL */
		while (VisitedEmptyBoxList.size() < nn(0) * nn(1) * nn(2)) {
			/* Randomly select a certain box */
			BoxStruct box1 = UnvisitedBoxQueue.top();
			UnvisitedBoxQueue.pop();

			/* Determining in which box is this point located */
			int b_xId, b_yId, b_zId;
			b_xId = round(box1.id / (nn(1) * nn(2)));
			b_yId = round((box1.id - b_xId * nn(1) * nn(2)) / nn(2));
			b_zId = box1.id % nn(2);
			if (v4[b_xId][b_yId][b_zId].size() < 1) continue;

			/* Picking a random sample from a box */
			srand(time(NULL));
			int randV1 = rand() % v4[b_xId][b_yId][b_zId].size();
			int boxSample;
			int locCounter = 0;
			for (list<int>::iterator it = v4[b_xId][b_yId][b_zId].begin(); it != v4[b_xId][b_yId][b_zId].end(); ++it) {
				if (randV1 == locCounter) boxSample = *it;
				locCounter++;
			}
			locSample.insert(locSample.end(), boxSample);

			/* Iterating through neigboring boxes */
			for (int i = b_xId - 1; i <= b_xId + 1; i++) {
				if (i < 0 || i >(nn(0) - 1)) continue;
				for (int j = b_yId - 1; j <= b_yId + 1; j++) {
					if (j < 0 || j >(nn(1) - 1)) continue;
					for (int k = b_zId - 1; k <= b_zId + 1; k++) {
						if (k < 0 || k >(nn(2) - 1)) continue;
						if (v4[i][j][k].size() > 0) {
							/* Removing samples closer than a certain radius */
							for (list<int>::iterator it = v4[i][j][k].begin(); it != v4[i][j][k].end(); ) {
								double distAB = (V.row(boxSample) - V.row(*it)).norm();
								//VtoVDist(V.row(boxSample), V.row(*it), distAB);

								if (distAB <= radius) {
									it = v4[i][j][k].erase(it);
								}
								else {
									++it;
								}
							} /* End of Removing samples closer than a certain radius */

							  /* Put the empty boxes to its list */
							if (v4[i][j][k].size() < 1) {
								VisitedEmptyBoxList.push_back(i * nn(1) * nn(2) + j * nn(2) + k);
							}
							else
							{
								/* If the original box is not empty, put it back to queue with new priority */
								if (i == b_xId && j == b_yId && k == b_zId) {
									double nRandDist = (double)(rand() % 100) / (double)10;
									// ID = a*(y*z) + b*(z) + c.
									BoxStruct curBox{ i * nn(1) * nn(2) + j * nn(2) + k, nRandDist };
									UnvisitedBoxQueue.push(curBox);
								}
							}
						}
					} /* End of iteration on each box */
				}
			} /* End of Iterating through neigboring boxes */
		}
		/* END of Iterating to get All samples */

		/* Passing sample to caller */
		Sample.resize(locSample.size());

		int sId = 0;
		for (int samp : locSample) {
			Sample(sId++) = samp;
		}
		printf("The exact sample is %d \n", Sample.rows());
		//printf("	Sample: %d from %d pool (%.1f percent deletion) (boxes=%d).\n", Sample.size(), poolCounter, 
		//			100.0*(double)(poolCounter - Sample.size()) / (double)poolCounter, nn(0)*nn(1)*nn(2));
	}
}