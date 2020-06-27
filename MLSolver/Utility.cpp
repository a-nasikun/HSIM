#include "Utility.h"

#include <igl/doublearea.h>

#include <random>
#include <fstream>
#include <igl/matlab/matlabinterface.h>

#include <chrono>				// For quite precise timing, needed (perhaps) in most sub-task

using namespace std;

namespace MuLES
{

	/* Create a list of neighbors of every vertex *
	** Input:
	**		V: matrix of vertices
	**		F: matrix of faces (e.g. triangles) OR tetrahedron
	** Output:
	**		AM: Adjacency matrix */
	void generateRandomMatrix(int m, int n, Eigen::MatrixXd& Mat)
	{
		Eigen::MatrixXd MTemp(m, n);
		Eigen::MatrixXd Id(m, n); Id.setIdentity();
		Mat.resize(m, n);

		/* Random number generator */
		std::random_device rd;								// Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd());								// Standard mersenne_twister_engine seeded with rd()

		std::uniform_int_distribution<> dist(0, m);			// From 0 to F.rows()-1

		/* Filling out each element by random numbers */
		for (int j = 0; j < n; j++) {						// column
			for (int i = 0; i < m; i++) {					// row
				double randNum = (double)dist(gen);
				MTemp(i, j) = 2.0 * (randNum / (double)m - 0.5);
			}
			//M.col(j).normalize();		
		}

		/* Use as it is */
		Mat = MTemp;

		/* Regular normalization */
		//for (int j = 0; j < n; j++) {
		//	M.col(j) = MTemp.col(j).normalized();
		//}

		/* QR orthonormalized*/
		//cout << "Householder QR :";
		//Eigen::HouseholderQR<Eigen::MatrixXd> qr(MTemp);
		//M = qr.householderQ()*Id;	
		//cout << "ColPivHouseholder QR :";
		//Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(MTemp);
		//cout << "its rank:" << qr.rank() << endl;
		//M = qr.householderQ()*Id;

		//cout << "Block: \n" << Mat.topLeftCorner(20, min(n, 10)) << endl;
	}

	void generateBatheInitVectors(int m, int n, const Eigen::SparseMatrix<double> M, Eigen::MatrixXd& Mat)
	{
		/* Initiating the Bathe's proposed inite vectors */
		Mat.resize(m, n);
		Eigen::VectorXd e(m); e.setZero();
		Mat.col(0) = M.diagonal();

		for (int i = 1; i < (n - 1); ++i) {
			e(i) = 1.0;
			Mat.col(i) = e;
			e(i) = 0.0;
		}

		/* Last column **
		** Filling out each element by random numbers */
		std::random_device rd;								// Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd());								// Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<> dist(0, m);			// From 0 to F.rows()-1

		for (int i = 0; i < m; ++i) {					// row
			double randNum = (double)dist(gen);
			Mat(i, n - 1) = 2.0 * (randNum / (double)m - 0.5);
		}
		double mNorm = Mat.col(n - 1).transpose()*M*Mat.col(n - 1);
		mNorm = (double) 1.0 / sqrt(mNorm);
		Mat.col(n - 1) = mNorm * Mat.col(n - 1);
	}

	void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio)
	{
		cout << "> Scaling the mesh to have a unit length \n";
		Eigen::Vector3d minV;
		Eigen::Vector3d maxV;
		Eigen::Vector3d length;
		Eigen::MatrixXd MatSubs(V.rows(), V.cols());
		Eigen::MatrixXd MatAdd(V.rows(), V.cols());
		double maxVal;
		double scalingRatio = scaleRatio; //dia-meter, not radius


		/* Get the min and max coefficients */
		for (int i = 0; i < V.cols(); i++)
		{
			minV(i) = V.col(i).minCoeff();
			maxV(i) = V.col(i).maxCoeff();
			length(i) = maxV(i) - minV(i);
			MatSubs.col(i).setConstant(minV(i));
		}

		maxVal = length.maxCoeff();

		/* Translate to the Origin */
		V = V - MatSubs;

		/* Scale w.r.t the longest axis */
		V = V * (scalingRatio / maxVal);

		for (int i = 0; i < V.cols(); i++) {
			maxV(i) = V.col(i).maxCoeff();
			MatAdd.col(i).setConstant(0.5*maxV(i));
		}

		/* Translate s.t. the center is in the Origin */
		V = V - MatAdd;

		for (int i = 0; i < V.cols(); i++)
		{
			minV(i) = V.col(i).minCoeff();
			maxV(i) = V.col(i).maxCoeff();
			length(i) = maxV(i) - minV(i);
		}
		maxVal = length.maxCoeff();
		cout << "____after scaling: " << maxVal << endl;
	}

	void printVector(const Eigen::VectorXd& V, bool isTranspose, int numDigit)
	{
		std::string text = "%." + to_string(numDigit) + "lg";
		if (isTranspose) text += "\t";
		else text += "\n";

		for (int i = 0; i < V.rows(); ++i) {
			std::printf(text.c_str(), V(i));
		}

		std::cout << std::endl;
	}

	void checkOrthonormality(const Eigen::MatrixXd& EigVect, const Eigen::SparseMatrix<double>& M, Eigen::MatrixXd &OrthMat)
	{
		Engine* engine;
		igl::matlab::mlinit(&engine);

		OrthMat = EigVect.transpose()*M*EigVect;
		igl::matlab::mlsetmatrix(&engine, "M", OrthMat);
		igl::matlab::mleval(&engine, "clf;");
		igl::matlab::mleval(&engine, "imagesc(M);");

		srand(time(NULL));
		int k = rand() % 1000;
		if(k%2==0)
			igl::matlab::mleval(&engine, "colormap(gca,jet(256));");
		else 
			igl::matlab::mleval(&engine, "colormap(gca,parula(256));");
		igl::matlab::mleval(&engine, "colorbar(gca);");

		//igl::matlab::mleval(&engine, "hold on; n=size(M,1); for i = 1:n    plot([.5,n+.5],[i-.5,i-.5],'k-');    plot([i-.5,i-.5],[.5,n+.5],'k-'); end");
		//igl::matlab::mleval(&engine, "hold on;");
		//igl::matlab::mleval(&engine, "for i = 1:size(M,1)+1");
		//igl::matlab::mleval(&engine, "plot([.5, size(M,1)+0.5], [i - .5, i - .5], 'k-');");
		//igl::matlab::mleval(&engine, "plot([i - .5, i - .5], [.5, size(M,1)+0.5], 'k-');");
		//igl::matlab::mleval(&engine, "end");
	}
}