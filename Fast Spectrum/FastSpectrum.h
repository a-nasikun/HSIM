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
	void constructBasisCoarseOnly();
	void constructNonSmoothBasis();
	void constructRestrictedProblem();
	void solveRestrictedProblem();

	/* [SUPPLEMENTARY FUNCTIONS] */
	void readMesh(const string &meshFile, Eigen::MatrixXd &V, Eigen::MatrixXi &F);
	void constructBasisFunctionsLevel0();
	void constructBasisFunctionsLevelK(int k);	
	void constructBasisFunctionsCoarseOnly();
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
	void solveHeatMethodRedMultiGridNonSmooth(igl::opengl::glfw::Viewer &viewer);
	void prepareRedMatrix();
	void solveHeatEquationRed(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void solveHeatEquationRedMultiGridNonSmooth(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void normalizeGradientRed(Eigen::VectorXd& grad);
	void solvePoissonEquationRed(const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void visualizeHeatGeodesicsRed(igl::opengl::glfw::Viewer &viewer, const Eigen::VectorXd& functions);
	

	void multiGridSolver(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b, Eigen::VectorXd& x);

	/* Supporting functions for non-smooth basis */
	void BuildLocalClusterLevel0();
	void BuildLocalClusterLevelK(int k);
	void visualizeClusterOfSamplesLevel0(igl::opengl::glfw::Viewer &viewer);
	void visualizeClusterOfSamplesLevelK(igl::opengl::glfw::Viewer &viewer, int k);
	void visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k);

	/* Reference eigenproblem */
	void computeRefEigenProblem(int numEig, const string& filename);

	void testGaussSeidel();

public:
	/* [CLASS VARIABLES] */
	Eigen::SparseMatrix<double>		S, M, Mf, MfInv, G,  Basis, DistanceTableSpM;
	Eigen::SparseMatrix<double>		S_, M_, GTMf_;									// reduced stiffness, mass, and divergent matrix, respectively
	Eigen::MatrixXd					V;
	Eigen::MatrixXd					reducedEigVects, refEigVects;
	Eigen::MatrixXi					F;												// face matrix
	Eigen::VectorXd					reducedEigVals, refEigVals;			
	Eigen::VectorXi					Sample, SampleMap;								// id of the samples | for a certain vertex sample, get the id of the sample
	vector<set<int>>				AdM;											// adjacency matrix, vertex to neighboring vertices
	vector<double>					maxNeighDist;									// geodesic distance for sampling, to control the sparsity
	double							avgEdgeLength, surfaceArea;						// average edge length and total surface area (from the mesh)
	int								sampleSize;										// sample of the finest level sample, i.e. level 0
	SamplingType					sampleType = Sample_Farthest_Point;				// we only use Farthest point sampling on this scenario

	/* [MULTI-GRID STUFF]*/
	int									level;										// the total number of hierarchy/levels on multigrid
	vector<int>							SubspaceSize;								// subspace/sample size on each level, from level 0 to level n-1
	vector<Eigen::SparseMatrix<double>> U;											// basis matrix on each level
	vector<Eigen::SparseMatrix<double>> Sbar, Mbar;									// reduced system matrices (stiffness + mass matrix) on each level
	vector<Eigen::VectorXd>				geoDist;									// resulting geodesic distance on each level, stored at the last index on each level
	vector<Eigen::VectorXi>				clusters;									// the index of sample point [level k] closest to a certain vertex/sample [level k-1]
	Eigen::VectorXi						sampleLevel;								// to indicate if a vertex is picked as a sample; if yes, at which highest level k

	/* HEAT METHOD */
	Eigen::VectorXd						inputHeat, outputHeat;						// input and output on heat method
	Eigen::VectorXd						grad;										// gradient direction on heat method
	Eigen::VectorXd						heatDistance;								// the resulting geodesic distance based on heat method
	Eigen::VectorXd						inputHeatRed, outputHeatRed;
	Eigen::VectorXd						gradRed;
	Eigen::VectorXd						heatDistanceRed;

	
};

#endif // !FAST_SPECTRUM_H

