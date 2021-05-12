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

	//template <typename DerivedM>
	//void writeDenseMatrixToFile(const Eigen::PlainObjectBase<DerivedM>& M, const string& filename, bool overwrite = false);

	template <typename DerivedM>
	void writeDenseMatrixToFile(const Eigen::PlainObjectBase<DerivedM>& M, const string& filename, bool overwrite = true)
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
			///myfile << "#Cols=" << M.cols() << "\n";

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

	template <typename DerivedM>
	void loadDenseMatrixFromFile(const string &filename,  Eigen::PlainObjectBase<DerivedM>&M)
	{
		ifstream readFile;
		readFile.open(filename);
		std::cout << "Filename: " << filename << std::endl;
		std:: string line; // , word;
		std::vector<std::vector<double>> matrix;
		matrix.reserve(10);
		int iCounter = 0;
		int colCounter = 0;

		//if (readFile.is_open()) {
		//std::cout << "Trying to read the file \n";
		while (!readFile.eof()) {
			//printf("Line number %d \n", iCounter + 1);
			getline(readFile, line);
			if (line.empty()) continue;
			std::istringstream iss(line);
			colCounter = 0;

			for (std::string word; iss >> word;) {
				//while(std::getline(iss, word, ' '))
				double vi = stod(word);
				//std::cout << vi << " \t";
				if (iCounter == 0) {
					std::vector<double> emptyVector;
					matrix.push_back(emptyVector);
					matrix[colCounter].reserve(10000);					
				}
				matrix[colCounter].push_back(vi);
				colCounter++;
			}
			iCounter++;	
		}

		//std::cout << "Convert to matrix \n";
		M.resize(iCounter, colCounter);
		for (int i = 0; i < iCounter; ++i) {
			for (int j = 0; j < colCounter; ++j) {
				M(i, j) = matrix[j][i];
			}
		}

		//std::cout << "Clearning things out \n";
		for (int i = 0; i < colCounter; ++i) {
			matrix[i].clear();
			matrix[i].shrink_to_fit();
		}
		matrix.clear();
		matrix.shrink_to_fit();
		readFile.close();

		//v = Eigen::VectorXd(vect.data());
		//printf("Matrix has the size of (%dx%d) \n", M.rows(), M.cols());

		//else {
		//	printf("Cannot access the file! \n");
		//}
	}

	void WriteVectorToFile(const Eigen::VectorXd& v, const string& filename);
	void ReadVectorFromFile(const string& filename, Eigen::VectorXd& v);
	void writeEigenDenseMatrixToBinary(Eigen::MatrixXd M, const std::string& filename);
	void readEigenDenseMatrixFromBinary(const std::string& filename, Eigen::MatrixXd& M);

}

#endif // !UTILITY_H

