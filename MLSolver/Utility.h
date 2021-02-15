#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <set>
#include <vector>

#include <Eigen/Eigen>

using namespace std;
namespace HSIM
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
	void generateBatheInitVectors(int m, int n, const Eigen::SparseMatrix<double> S, const Eigen::SparseMatrix<double> M, Eigen::MatrixXd& Mat);
	void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio = 1.0);
	void projectToSphere(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double radius = 1.0);
	void printVector(const Eigen::VectorXd& V, bool isTranspose = false, int numDigit = 16);

	void checkOrthonormality(const Eigen::MatrixXd& EigVect, const Eigen::SparseMatrix<double>& M, Eigen::MatrixXd &OrthMat);
	void WriteSparseMatrixToMatlab(const Eigen::SparseMatrix<double>& M, const string& filename);

	template <typename DerivedM>
	void writeDenseMatrixToFile(const Eigen::PlainObjectBase<DerivedM>& M, const string& filename, bool overwrite = false);

	template <typename DerivedM>
	void writeDenseMatrixToFile(const Eigen::PlainObjectBase<DerivedM>& M, const string& filename, bool overwrite)
	{
		ofstream myfile;		
		myfile.clear();
		myfile.close();

		if (overwrite)
			myfile.open(filename.c_str(), std::ofstream::out);						// overwrite existing content
		else
			myfile.open(filename.c_str(), std::ofstream::out | std::ofstream::app);	// append to the end of the text

		//ofstream myfile(filename.c_str(), std::ofstream::out | std::ofstream::app);

		if (myfile.is_open()) {
			//cout << "Writing Matrix file to text ";
			//printf("of %dx%d size \n", M.rows(), M.cols());
			myfile << "#Cols=" << M.cols() << "\n";

			for (int i = 0; i < M.rows(); i++) {
				for (int j = 0; j < M.cols(); j++) {
					myfile << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << M(i, j) << "\t";
				}
				myfile << "\n";
			}
			//cout << "The file writing is done\n";
			myfile.clear();
			myfile.close();
			//delete myfile;
		}
		else cout << "Unable to open file";
	}
}

#endif // !UTILITY_H

