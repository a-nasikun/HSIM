#ifndef FAST_SPECTRUM_H
#define FAST_SPECTRUM_H

#include "Basis.h"
#include "Utility.h"
#include "Sampling.h"
#include "EigenSolver.h"
#include "Visualization.h"

/* [LIBIGL HEADERS] */
#include <igl/jet.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/isolines.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/barycenter.h>
#include <igl/avg_edge_length.h>
#include <igl/opengl/glfw/Viewer.h>

using namespace std;

class FastSpectrum
{
public:
	/* [MAIN FUNCTIONS CALLED BY USER FOR ABSTRACTION] */

	// <Constructor> INPUT: String of mesh-location & number of samples/subpsace, OUTPUT: approximated lifted eigenvectors & eigenvalues)
	void computeEigenPairs(const string &meshFile, const int &numOfSamples, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals);
	// <Constructor> INPUT: String of mesh-location & number of samples/subpsace, OUTPUT: Basis & reduced eigenvectors & eigenvalues )
	void computeEigenPairs(const string &meshFile, const int &numOfSamples, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals);
	// <Constructor> INPUT: String of mesh-location & number of samples/subpsace & type of sampling, OUTPUT: approximated lifted eigenvectors & eigenvalues)
	void computeEigenPairs(const string &meshFile, const int &numOfSamples, SamplingType sampleType, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals);
	// <Constructor> INPUT: String of mesh-location & number of samples/subpsace & type of sampling, OUTPUT: Basis & reduced eigenvectors & eigenvalues)
	void computeEigenPairs(const string &meshFile, const int &numOfSamples, SamplingType sampleType, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals);
	// <Constructor> INPUT: V and F matrices & number of samples/subpsace, OUTPUT: approximated lifted eigenvectors & eigenvalues)
	void computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals);
	// <Constructor> INPUT: V and F matrices & number of samples/subpsace, OUTPUT: OUTPUT: Basis & reduced eigenvectors & eigenvalues )
	void computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals);
	// <Constructor> INPUT: V and F matrices & number of samples/subpsace & type of sampling, OUTPUT: approximated lifted eigenvectors & eigenvalues)
	void computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, SamplingType sampleType, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals);
	// <Constructor> INPUT: V and F matrices & number of samples / subpsace & type of sampling, OUTPUT : Basis & reduced eigenvectors & eigenvalues)
	void computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, SamplingType sampleType, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals);
	
	void setInitialSystem(int n);
	void constructLaplacianMatrix();
	void constructSampleHierarchy(Eigen::VectorXd& n);
	void constructSample();
	void constructBasis();
	void constructRestrictedProblem();
	void solveRestrictedProblem();

	/* [SUPPLEMENTARY FUNCTIONS] */
	void readMesh(const string &meshFile, Eigen::MatrixXd &V, Eigen::MatrixXi &F);
	void constructBasisFunctionsLevel0();
	void constructBasisFunctionsLevelK(int k);
	void formPartitionOfUnity();
	void formPartitionOfUnityMultiLevel();
	void computeEigenPair();
	void normalizeReducedEigVects();

	/* [GETTER AND SETTER]*/
	void getV(Eigen::MatrixXd &V);
	void getF(Eigen::MatrixXi &F);
	void getVF(Eigen::MatrixXd &V, Eigen::MatrixXi &F);
	void getAvgEdgeLength(double &avgEdgeLength);
	void getMassMatrix(Eigen::SparseMatrix<double> &M);
	void getStiffnessMatrix(Eigen::SparseMatrix<double> &S);
	void getLaplacian(Eigen::SparseMatrix<double> &S, Eigen::SparseMatrix<double> &M);
	void getSamples(Eigen::VectorXi &Sample);
	void getSampleSize(int &sampleSize);
	void getFunctionBasis(Eigen::SparseMatrix<double> &U);
	void getReducedEigVals(Eigen::VectorXd &reducedEigVals);
	void getReducedEigVects(Eigen::MatrixXd &reducedEigVects);
	void getApproxEigVects(Eigen::MatrixXd &approxEigVects);
	void getReducedStiffnessMatrix(Eigen::SparseMatrix<double> &Sbar);
	void getReducedMassMatrix(Eigen::SparseMatrix<double> &Mbar);
	void getReducedLaplacian(Eigen::SparseMatrix<double> &Sbar, Eigen::SparseMatrix<double> &Mbar);

	void setV(const Eigen::MatrixXd &V);
	void setF(const Eigen::MatrixXi &F);
	void setMesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
	void setMesh(const string &meshFile);
	void setSample(const int &sampleSize, SamplingType sampleType);
	void setSampleNumber(const int &sampleSize);
	void setSampleType(SamplingType sampleType);

	/* Heat Method*/
	void solveHeatMethod(igl::opengl::glfw::Viewer &viewer);
	void solveHeatEquation(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void normalizeGradient(Eigen::VectorXd& grad);
	void solvePoissonEquation(const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void visualizeHeatGeodesics(igl::opengl::glfw::Viewer &viewer, const Eigen::VectorXd& functions);

	/* Reduced (1-level) Heat Method */	
	void solveHeatMethodRed(igl::opengl::glfw::Viewer &viewer);
	void prepareRedMatrix();
	void solveHeatEquationRed(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void normalizeGradientRed(Eigen::VectorXd& grad);
	void solvePoissonEquationRed(const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void visualizeHeatGeodesicsRed(igl::opengl::glfw::Viewer &viewer, const Eigen::VectorXd& functions);


	void testGaussSeidel();

public:
	/* [CLASS VARIABLES] */
	Eigen::SparseMatrix<double>		S, M, Mf, MfInv, G,  Basis, DistanceTableSpM;
	Eigen::SparseMatrix<double>		S_, M_, GTMf_;
	Eigen::MatrixXd					V, reducedEigVects;
	Eigen::MatrixXi					F;
	Eigen::VectorXd					reducedEigVals;
	Eigen::VectorXi					Sample;
	vector<set<int>>				AdM;
	vector<double>					maxNeighDist;
	double							avgEdgeLength, surfaceArea;
	int								sampleSize;
	SamplingType					sampleType = Sample_Farthest_Point;

	/* [MULTI-GRID STUFF]*/
	int									level;
	vector<int>							SubspaceSize;
	vector<Eigen::SparseMatrix<double>> U;
	vector<Eigen::SparseMatrix<double>> Sbar, Mbar;

	
};

#endif // !FAST_SPECTRUM_H

