#ifndef FAST_SPECTRUM_H
#define FAST_SPECTRUM_H

#include "Basis.h"
#include "Utility.h"
#include "Sampling.h"
#include "EigenSolver.h"
#include "Visualization.h"

#include <igl/opengl/glfw/Viewer.h>

using namespace std;

enum class SIM_Type {SIM_MultiLevel, SIM_1level_WarmStart, SIM_1level_Random, SIM_1level_Bathe};
enum class Problem_Class {Laplace_Beltrami, Linear_Elasticity};

class MLEigSolver
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
	void constructSampleHierarchy(Eigen::VectorXd& n);		// Creating hierarchies when constructing the samples (multi-level), using farthest point sampling
	void constructSample();									// Creating samples
	void constructBasis();									// Constructing multi-level basis
	void constructBasisWithClamping(int axis, double clampSelection, double support);						// Constructing multi-level basis with clamping on one side
	void constructBasisCoarseOnly();						// Constructing 1-level basis (similar to our SGP)
	void constructNonSmoothBasis();							// 1/0-weighted function, from SIGGRAPH Asia paper
	void constructRestrictedProblem();
	void solveRestrictedProblem();

	/* [SUPPLEMENTARY FUNCTIONS] */
	void readMesh(const string &meshFile, Eigen::MatrixXd &V, Eigen::MatrixXi &F);
	void constructBasisFunctionsLevel0(int axis, double offset, double support);
	void constructBasisFunctionsLevelK(int k, int axis, double offset, double support);
	void constructBasisFunctionsCoarseOnly(int axis, double offset, double support);
	void formPartitionOfUnity();
	void formPartitionOfUnityMultiLevel();
	void formPartitionOfUnityCombinatorialMultiLevel();
	void computeEigenPair();
	void normalizeReducedEigVects();
	void computeTempBasis(int k);
	void visualizeBasis(igl::opengl::glfw::Viewer &viewer, int k);

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

	/* For Volumetric Data */
	void createTetrahedralMesh();
	void computeVolume();
	void constructVolumetricMassMatrix();
	void constructHessianMatrixForLinearElasticity(double lame1, double lame2);
	void constructHessianMatrixForLinearElasticityMM(double m1l1, double m1l2, double m2l1, double m2l2, Eigen::VectorXd& vColor);
	void constructBasisFunctionsHessianLevel0(int axis, double offset, double support);
	void constructBasisFunctionsHessianLevelK(int k, int axis, double offset, double support);
	void constructBasisFunctionsHessianCoarseOnly(int axis, double offset, double support);


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
	void solveHeatEquationRed_CoarseOnly(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);	
	void solveHeatEquationRedMultiGridNonSmooth(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void normalizeGradientRed(Eigen::VectorXd& grad);
	void solvePoissonEquationRed(const Eigen::VectorXd& input, Eigen::VectorXd& output);	
	void solvePoissonEquationRed_CoarseOnly(const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void visualizeHeatGeodesicsRed(igl::opengl::glfw::Viewer &viewer, const Eigen::VectorXd& functions);

	/* Heat Diffusion */
	void multiLevelHeatDiffusion(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void buildMultiLevelRHS(Eigen::VectorXd& y0, vector<Eigen::VectorXd>& y);
	void solveHeatDiffusionLevelK(int k, vector<Eigen::VectorXd>& rhs, vector<Eigen::VectorXd>& x);
	void multiLevelHeatDiffusion(igl::opengl::glfw::Viewer &viewer, double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output);
	void buildMultiLevelRHS(igl::opengl::glfw::Viewer &viewer, double tau, const Eigen::VectorXd& y0, vector<Eigen::VectorXd>& y);
	void solveHeatDiffusionLevelK(igl::opengl::glfw::Viewer &viewer, double tau, int k, vector<Eigen::VectorXd>& rhs, vector<Eigen::VectorXd>& x);
	void buildMultiLevelRHSFineOnly(igl::opengl::glfw::Viewer &viewer, double tau, const Eigen::VectorXd& y0, vector<Eigen::VectorXd>& y);
	void solveHeatDiffusionLevelKFineOnly(igl::opengl::glfw::Viewer &viewer, double tau, int k, vector<Eigen::VectorXd>& rhs, vector<Eigen::VectorXd>& x);
	void postSmoothing(igl::opengl::glfw::Viewer &viewer, double tau, const Eigen::VectorXd& y0, Eigen::VectorXd& y);
	void diffusionVCycle(igl::opengl::glfw::Viewer &viewer, const Eigen::SparseMatrix<double>& L1, double tau, int lvl_now, int lvl_goal, Eigen::VectorXd& vin, Eigen::VectorXd& vout);
	void solveHeatDiffsuionVCycle(igl::opengl::glfw::Viewer &viewer, double tau, vector<Eigen::VectorXd>& rhs, vector<Eigen::VectorXd>& x);

	void measureAccuracyOfReducedSolver(Eigen::VectorXd& ref, Eigen::VectorXd& red);
	void mapFunction(Eigen::VectorXd& func);
	void rearrangeGradient3D(Eigen::SparseMatrix<double>& Grad3D);

	void multiGridSolver2LevelHardCode(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b, Eigen::VectorXd& x);
	void multiGridSolver(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b, Eigen::VectorXd& x);
	void multiGridSolverLevelK(Eigen::SparseMatrix<double>& A, int k, Eigen::VectorXd& b, Eigen::VectorXd& x0, Eigen::VectorXd& x);
	void initiateMultigrid(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b, Eigen::VectorXd& x);
	void multiGridVCycle(Eigen::SparseMatrix<double>& A, Eigen::SparseMatrix<double>& Mv, Eigen::VectorXd& b, Eigen::VectorXd& x, int k);
	void multiGridFCycle(Eigen::SparseMatrix<double>& A, Eigen::SparseMatrix<double>& Mv, Eigen::VectorXd& b, Eigen::VectorXd& x, int k);
	void multiGridWCycle(Eigen::SparseMatrix<double>& A, Eigen::SparseMatrix<double>& Mv, Eigen::VectorXd& b, Eigen::VectorXd& x, int k);
	
	void testGaussSeidelSolver(igl::opengl::glfw::Viewer &viewer);

	/* Supporting functions for non-smooth basis */
	void BuildLocalClusterLevel0();
	void BuildLocalClusterLevelK(int k);
	void visualizeClusterOfSamplesLevel0(igl::opengl::glfw::Viewer &viewer);
	void visualizeClusterOfSamplesLevelK(igl::opengl::glfw::Viewer &viewer, int k);
	void visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k);

	/* Reference eigenproblem */
	void computeRefEigenProblem(int numEig, const string& filename);
	void computeSelectedRefEigenProblem(int axis, double clampSelection, int numEig, const string& filename);


	void testGaussSeidel();
	void testConstantFuncsProjection();

	/* Subspace Iteration Method */
	void SubspaceIterMethod(igl::opengl::glfw::Viewer &viewer);
	void simpleSubspaceIterMethod(igl::opengl::glfw::Viewer &viewer, SIM_Type type, double tol, int numModes, int maxIter);
	void simpleSubspaceIterMethod(igl::opengl::glfw::Viewer &viewer, SIM_Type type, double tol, int width, int max_iter, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals);
	void simpleSubspaceIterMethodDiscardConverged(igl::opengl::glfw::Viewer &viewer, int width, int max_iter, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals);
	void SubspaceIterMethodShift(igl::opengl::glfw::Viewer &viewer);
	void SubspaceIterMethodBlockShift(igl::opengl::glfw::Viewer &viewer, int init, int width, double myu, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals);
	void MultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, double tol, int numEigs, int k, int maxIter, Eigen::MatrixXd& XL, Eigen::VectorXd& AL);


	void generateBatheInitVectors(int m, int n, Eigen::MatrixXd& Xk);
	void generateSampleBaseInitVectors(int m, Eigen::MatrixXd& Xk);
	void checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, Eigen::VectorXd& convRate);
	void checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, Eigen::SparseMatrix<double>& MK, Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate);
	void checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, Eigen::VectorXd& convRate);

	
public:
	/* [CLASS VARIABLES] */
	Problem_Class					problemType = Problem_Class::Laplace_Beltrami;	// Selected problem, e.g. either laplace--beltrami or hessian of elasticity
	Eigen::SparseMatrix<double>		S, M, Minv, Mf, MfInv, G;						// Stiffness, Mass and inverse mass matrix for triangle-mesh
	Eigen::SparseMatrix<double>		ST, MT, MTinv, MThNeg;							// Stiffness, Mass and inverse mass matrix, and H^{-0.5} for tet-mesh
	Eigen::SparseMatrix<double>		Basis;											// combined basis matrix (Basis = U0 * Uk * ....  U_{k-1} *  Uk)
	Eigen::SparseMatrix<double>		DistanceTableSpM;								// stored vertex-to-vertex distance table
	Eigen::SparseMatrix<double>		S_, M_, GTMf_;									// reduced stiffness, mass, and divergent matrix, respectively
	Eigen::MatrixXd					V;												// vertices position (triangles)
	Eigen::MatrixXi					F;												// face matrix (indices of each vertex for each face)
	Eigen::MatrixXd					TV, TB;											// rest-shape and barycentric positions of the tetaderal
	Eigen::MatrixXi					TT, TF;											// tetrahedron and face matrix (indices of each vertex for each face)
	Eigen::VectorXd					volume;											// volume of each tetrahedral mesh// tetahedral and faces index
	Eigen::MatrixXd					reducedEigVects, refEigVects;	
	Eigen::VectorXd					reducedEigVals, refEigVals;			
	Eigen::VectorXi					Sample, SampleMap;								// id of the samples | for a certain vertex sample, provided that it's a sample vertex, get its 'sample id' 
	vector<set<int>>				AdM;											// adjacency matrix, vertex to neighboring vertices
	vector<double>					maxNeighDist;									// geodesic distance for sampling, to control the sparsity
	double							avgEdgeLength, surfaceArea;						// average edge length and total surface area (from the mesh)
	int								sampleSize;										// sample of the finest level sample, i.e. level 0
	int								heatSource;
	SamplingType					sampleType = Sample_Farthest_Point;				// we only use Farthest point sampling on this scenario
	bool							isClampedBasis;

	/* [SUBSPACE ITERATION METHOD] */
	Eigen::MatrixXd					SIMeigVects;									// matrix storing eigenvectors, after Subspace Iteration Method (SIM)
	Eigen::VectorXd					SIMeigVals;										// vector stroing eigenvalues,  after Subspace Iteration Method (SIM)
	Eigen::MatrixXd					SIMeigValsStack;								// Stack of eigenvalues, from multiple iterations
	Eigen::MatrixXd					convergenceStacks;								// Stack of convergence rate, from multiple iterations

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
	double								tauScale;

	
};

#endif // !FAST_SPECTRUM_H

