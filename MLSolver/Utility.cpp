#include "Utility.h"

#include <igl/doublearea.h>

#include <random>
#include <fstream>
#include <igl/matlab/matlabinterface.h>

/* [MATLAB] */
#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>
#include <tuple>
#include "engine.h"
#include "mex.h"

#include <chrono>				// For quite precise timing, needed (perhaps) in most sub-task

using namespace std;

namespace HSIM
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

	void generateBatheInitVectors(int m, int n, const Eigen::SparseMatrix<double> S, const Eigen::SparseMatrix<double> M, Eigen::MatrixXd& Mat)
	{
		/* Initiating the Bathe's proposed inite vectors */
		Mat.resize(m, n);
		Eigen::VectorXd e(m); e.setZero();
		Mat.col(0) = M.diagonal();

		Eigen::VectorXd ratios(M.rows()); 
		ratios.setZero();

		for (int i = 0; i < M.rows(); ++i) {
			ratios(i) = S.coeff(i, i) / M.coeff(i, i);
		}
		//std::cout << "S[i,i]/M[i,i] ratios \n" << ratios.transpose() << std::endl;
		Eigen::VectorXd ratios_sorted = ratios;

		//std::sort(std::begin(ratios.data()), std::end(ratios.data()), [](int lhs, int rhs) {return rhs > lhs});
		std::sort(ratios_sorted.data(), ratios_sorted.data() + ratios_sorted.size());
		//std::cout << "S[i,i]/M[i,i] ratios \n" << ratios_sorted.transpose() << std::endl;

		double up_bound = ratios_sorted(n-3);
		Eigen::VectorXi smallRatioID(n - 2);
		int idCounter = 0;
		for (int i = 0; i < ratios.size(); ++i) {
			if (ratios(i) <= up_bound) {
				smallRatioID(idCounter) = i;
				idCounter++;
			}
			if (idCounter >= n - 2) break;
		}
		std::cout << "Entries with minimum ratio of Sii/Mii are \n" << smallRatioID.transpose() << std::endl; 

		for (int i = 0; i < smallRatioID.size(); ++i) {
			e(smallRatioID(i)) = 1.0;
			Mat.col(i + 1) = e;
			e.setZero();
		}


		//for (int i = 1; i < (n - 1); ++i) {
		//	e(i) = 1.0;
		//	Mat.col(i) = e;
		//	e(i) = 0.0;
		//}

		/* Last column **
		** Filling out each element by random numbers */
		std::random_device rd;								// Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd());								// Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<> dist(0, m);			// From 0 to F.rows()-1

		for (int i = 0; i < m; ++i) {					// row
			double randNum = (double)dist(gen);
			Mat(i, n - 1) = 2.0 * (randNum / (double)m - 0.5);
		}
		double mNorm = Mat.col(n - 1).transpose() * M * Mat.col(n - 1);
		mNorm = (double)1.0 / sqrt(mNorm);
		Mat.col(n - 1) = mNorm * Mat.col(n - 1);
	}

	void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio)
	{
		//cout << "> Scaling the mesh to have a unit length \n";
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
		//cout << "____after scaling: " << maxVal << endl;
	}

	void printVector(const Eigen::VectorXd& V, bool isTranspose, int numDigit)
	{
		std::string text = "%." + to_string(numDigit) + "lg";
		if (isTranspose) text += "\t";
		else text += "\n";

		for (int i = 0; i < V.rows(); ++i) {
			printf(text.c_str(), V(i));
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

	void WriteSparseMatrixToMatlab(const Eigen::SparseMatrix<double>& M, const string& filename)
	{
		printf("Saving matrix with the size of M=%dx%d\n", M.rows(), M.cols());

		using namespace matlab::engine;
		Engine *ep;
		mxArray *MM = NULL, *MS = NULL, *result = NULL, *eigVecResult, *nEigs;

		const int NNZ_M = M.nonZeros();
		int nnzMCounter = 0;

		double	*srm = (double*)malloc(NNZ_M * sizeof(double));
		mwIndex *irm = (mwIndex*)malloc(NNZ_M * sizeof(mwIndex));
		mwIndex *jcm = (mwIndex*)malloc((M.cols() + 1) * sizeof(mwIndex));

		MM = mxCreateSparse(M.rows(), M.cols(), NNZ_M, mxREAL);
		srm = mxGetPr(MM);
		irm = mxGetIr(MM);
		jcm = mxGetJc(MM);

		// Getting matrix M
		jcm[0] = nnzMCounter;
		for (int i = 0; i < M.outerSize(); i++) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it) {
				srm[nnzMCounter] = it.value();
				irm[nnzMCounter] = it.row();
				nnzMCounter++;
			}
			jcm[i + 1] = nnzMCounter;
		}

		// Start Matlab Engine
		ep = engOpen(NULL);
		if (!(ep = engOpen(""))) {
			fprintf(stderr, "\nCan't start MATLAB engine\n");
			cout << "CANNOT START MATLAB " << endl;
		}
		else {
			cout << "MATLAB STARTS. OH YEAH!!!" << endl;
		}

		engPutVariable(ep, "MF", MM);
		// LAPTOP
		//engEvalString(ep, "save('F:/PROGRAMMING/Localized Vector Fields/Build/ForChristopher/Armadillo/Basis','Basis');");
		//engEvalString(ep, "save('E:/Local Programming/Localized Bases for Vector Fields/LocalFields_build/ForChristopher/Armadillo/Basis','Basis');");

		// WORKSTATION
		//engEvalString(ep, "save('E:/Local Programming/Localized Bases for Vector Fields/LocalFields_build/ForChristopher/Armadillo/Basis','Basis');");
		string saveFile = "save('" + filename + "','MF');";
		engEvalString(ep, saveFile.c_str());

	}
}