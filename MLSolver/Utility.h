#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <set>
#include <vector>

#include <Eigen/Eigen>

using namespace std;
namespace MuLES
{
	/* Data structure for Priority Queue */
	struct VertexPair {
		int		vId;
		double	distance;
		bool	operator> (const VertexPair &ref) const { return distance > ref.distance; }
		bool	operator< (const VertexPair &ref) const { return distance < ref.distance; }
	};

	void generateRandomMatrix(int m, int n, Eigen::MatrixXd& Mat);
	void generateBatheInitVectors(int m, int n, const Eigen::SparseMatrix<double> M, Eigen::MatrixXd& Mat);
	void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio = 1.0);
	void printVector(const Eigen::VectorXd& V, bool isTranspose = false, int numDigit = 16);

	void checkOrthonormality(const Eigen::MatrixXd& EigVect, const Eigen::SparseMatrix<double>& M, Eigen::MatrixXd &OrthMat);

	template <typename DerivedM>
	void writeDenseMatrixToFile(const Eigen::PlainObjectBase<DerivedM>& M, const string& filename, bool overwrite = false);

	template <typename DerivedM>
	void writeDenseMatrixToFile(const Eigen::PlainObjectBase<DerivedM>& M, const string& filename, bool overwrite)
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

