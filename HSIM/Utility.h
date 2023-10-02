#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <set>
#include <vector>

#include <Eigen/Eigen>

using namespace std;

namespace HSIM {
	/* Data structure for Priority Queue */
	struct VertexPair {
		int		vId;
		double	distance;
		bool	operator> (const VertexPair& ref) const { return distance > ref.distance; }
		bool	operator< (const VertexPair& ref) const { return distance < ref.distance; }
	};


	/* Create a list of neighbors of every vertex */
	void initiateAdM(Eigen::MatrixXd& V, Eigen::MatrixXi& F, vector<set<int>>& AM);
	/* Precompute the distance between a vertex and its neighbor(s), for a bit of speed up */
	void initiateDistTableSpM(Eigen::MatrixXd& V, const vector<set<int>>& AM, Eigen::SparseMatrix<double>& DistanceTable);

	/* Compute distance between two points */
	void VtoVDist(const Eigen::VectorXd& V1, const Eigen::VectorXd& V2, double& dist);
	/* Rotate the edge*/
	void rotateVector2D(const Eigen::RowVector2d& e, const double& angle, Eigen::RowVector2d& e2);
	/* Get the angle between two edges */
	double VtoVangle(const Eigen::Vector3d& V1, const Eigen::Vector3d& V2, const Eigen::Vector3d& P);
	/* Compute Areas of surface*/
	double computeSurfaceArea(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
	/* Check accuracy*/
	double checkAccuracy(const Eigen::SparseMatrix<double>& M, const Eigen::VectorXd& ref, const Eigen::VectorXd& approx);

	void generateRandomMatrix(int m, int n, Eigen::MatrixXd& Mat);
	void generateBatheInitVectors(int m, int n, const Eigen::SparseMatrix<double> M, Eigen::MatrixXd& Mat);
	void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio = 1.0);
	void printVector(const Eigen::VectorXd& V, bool isTranspose = false, int numDigit = 16);

	template <typename DerivedM>
	void writeDenseMatrixToFile(const Eigen::PlainObjectBase<DerivedM>& M, const string& filename, bool overwrite = false)
	{
		ofstream myfile;
		if (overwrite)
			myfile.open(filename.c_str(), std::ofstream::out);						// overwrite existing content
		else
			myfile.open(filename.c_str(), std::ofstream::out | std::ofstream::app);	// append to the end of the text

		if (myfile.is_open()) {
			cout << "Writing Matrix file to text ";
			printf("of %dx%d size \n", M.rows(), M.cols());
			myfile << "#Cols=" << M.cols() << "\n";

			for (int i = 0; i < M.rows(); i++) {
				for (int j = 0; j < M.cols(); j++) {
					myfile << M(i, j) << "\t";
				}
				myfile << "\n";
			}
			cout << "The file writing is done\n";
			myfile.close();
		}
		else cout << "Unable to open file";
	}

}
#endif // !UTILITY_H

