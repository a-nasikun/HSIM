#ifndef ML_EIGEN_SOLVER
#define ML_EIGEN_SOLVER

#include "Basis.h"
#include "Utility.h"
#include "Sampling.h"
#include "EigenSolver.h"
#include "Visualization.h"

#include <igl/opengl/glfw/Viewer.h>

using namespace std;


//enum class Problem_Class { Laplace_Beltrami, Linear_Elasticity };

class MLEigenSolver
{
public:
	enum class EigenVectorType	{ EIGEN_REFERENCE, EIGEN_REDUCED, EIGEN_SIM };
	enum class SIMType			{ SIM_MultiLevel, SIM_1level_WarmStart, SIM_1level_Random, SIM_1level_Bathe };
	enum class EigenSolverType	{ EIG_SOLVER_MATLAB, EIG_SOLVER_CUDA, EIG_SOLVER_SPECTRA };
	enum class ShiftType		{ SHIFT_BATHE, SHIFT_ITOH_WILSON, SHIFT_GONG, SHIFT_ZHAO };

	void readMesh(const string &meshFile);
	void setInitialSystem(int n);
	void initiateAdM();
	void virtual constructMassMatrix();
	void virtual constructStiffnessMatrix();

	/* Construction of the subspace and/or hierarchy */
	void constructSampleHierarchy(const Eigen::VectorXd& samples);	// Constructing multi-level samples
	void virtual constructBasis(double support, int axis=0, double offset = 0.0);											// Constructing multi-level basis
	void virtual constructBasisCoarseOnly(double support, int axis = 0, double offset = 0.0);
	void virtual constructBasisNonSmooth(double support, int axis = 0, double offset = 0.0);
	void virtual constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support);	
	void virtual constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int level, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter);
	void formPartitionOfUnityMultiLevel();
	void formPartitionOfUnity();
	void solveRestrictedProblem();
	void virtual constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis = 0, double offset = 0.0);

	/* Subspace Iteration methods */
	void virtual performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType=ShiftType::SHIFT_BATHE,  int axis = 0, double offset = 0.0);
	void virtual performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false);
	void virtual performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	void virtual performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol=1e-4, bool isWithShift = false);
	void virtual performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	void virtual performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	void checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const;
	void checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, bool isWithShift, double mu, Eigen::VectorXd& convRate) const;

	/* Computing the reference eigenproblem */
	void computeRefEigenProblem(int numEigs, const EigenSolverType& solverType);
	void computeRefEigenProblemWithSelector(int numEigs, const EigenSolverType& solverType, int axis=0, double offset=0.0);

	/* Visualization tools */
	void visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k) const;
	void virtual visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, double scale=1.0) const;
	void virtual visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType,  double scale=1.0) const;
	void visualizeClusterOfSamplesLevel0(igl::opengl::glfw::Viewer &viewer);
	void visualizeClusterOfSamplesLevelK(igl::opengl::glfw::Viewer &viewer, int k);

	/* Some setters and getters */
	Eigen::MatrixXd getV() const;
	Eigen::MatrixXi getF() const;
	void setV(const Eigen::MatrixXd& Vin);
	void setF(const Eigen::MatrixXi& Fin);
	Eigen::SparseMatrix<double> getStiffnessMatrix() const;
	Eigen::SparseMatrix<double> getMassMatrix() const;
	Eigen::SparseMatrix<double> getBasisMatrix() const;
	bool getClampBasis() const;
	void setClampBasis(bool isClamped);
	int getLevel() const; 
	int getSampleSize() const; 
	const Eigen::VectorXi& getSample() const;
	const vector<int>& getSubspaceSize() const;

	const Eigen::MatrixXd& getEigenVectorsRef() const;
	const Eigen::VectorXd& getEigenVectorsRef(int id) const;
	const Eigen::VectorXd& getEigenValuesRef() const;
	const double getEigenValueRef(int id) const;
	const Eigen::MatrixXd& getEigenVectorsReduced() const;
	const Eigen::VectorXd& getEigenVectorsReduced(int id) const;
	const Eigen::VectorXd& getEigenValuesReduced() const;
	const double getEigenValueReduced(int id) const;
	const Eigen::MatrixXd& getEigenVectorsSIM() const;
	const Eigen::VectorXd& getEigenVectorsSIM(int id) const;
	const Eigen::VectorXd& getEigenValuesSIM() const;
	const double getEigenValueSIM(int id) const;

	void LBSTest_pos(igl::opengl::glfw::Viewer &viewer, int numBasis, int id, double trans, int transAxis, double angle, int rotAxis);
	void LBSTest_vel(igl::opengl::glfw::Viewer &viewer, int numBasis, int id, double trans, int transAxis, double angle, int rotAxis);

protected:
	/* Mesh-related variables */
	Eigen::MatrixXd						V;					// Coordinates of each vertices
	Eigen::MatrixXi						F;					// Vertices ID of each face (functions) or tetrahedron (elastics);

	/* System matrices */
	Eigen::SparseMatrix<double>			S;					// stiffness matrix
	Eigen::SparseMatrix<double>			M;					// mass matrix
	Eigen::SparseMatrix<double>			Minv, MhNeg;		// inverse and ^(1/2) of the mass matrix, respectively
	Eigen::SparseMatrix<double>			G;					// gradient matrix

	/* Utility matrices, vectors, and scalar */
	Eigen::SparseMatrix<double>			SelMatrix;			// selector matrix for eigen computation
	Eigen::VectorXi						Sample;				// the ID of vertices selected as samples 
	Eigen::VectorXi						SampleMap;			// For a certain vertex ID (provided that it's a sample vertex) get its 'sample id' 								
	int									sampleSize;			// sample of the finest level sample, i.e. level 0
	vector<set<int>>					AdM;				// adjacency matrix, vertex to neighboring vertices
	vector<double>						maxNeighDist;		// geodesic distance for sampling, to control the sparsity

	/* Subspace and spectral-related matrices, vectors, and variables */
	Eigen::SparseMatrix<double>			Basis;				// Basis matrix (U[0]*U[1]*... *U[k])
	bool								isClampedBasis;		// toogling if the basis is a clamped version
	Eigen::MatrixXd						EigVectsRef;		// reference eigenvectors
	Eigen::MatrixXd						EigVectsReduced;	// approximated eigenvectors
	Eigen::MatrixXd						EigVectsReducedNonLifted;	// approximated eigenvectors, not Lifted
	Eigen::VectorXd						EigValsRef;			// reference eigenvalues
	Eigen::VectorXd						EigValsReduced;		// approximated eigenvalues
	Eigen::MatrixXd						EigVectsSIM;		// matrix storing eigenvectors, after Subspace Iteration Method (SIM)
	Eigen::VectorXd						EigValsSIM;			// vector stroing eigenvalues,  after Subspace Iteration Method (SIM)
	Eigen::MatrixXd						EigValsSIMStack;	// Stack of eigenvalues, from multiple iterations
	Eigen::MatrixXd						convergence2Stacks;	// Stack of convergence rate (convergence criteria2), from multiple iterations
	Eigen::MatrixXd						convergence3Stacks;	// Stack of convergence rate (convergence criteria2), from multiple iterations
	Eigen::VectorXd						timingPerLevel;		// Timing (in secs) for every level (in multi-level)
	Eigen::VectorXi						itersPerLevel;		// Number of iterations per every level (in multi-level)

	/* [MULTI-GRID STUFF]*/
	int									level;				// the total number of hierarchy/levels on multigrid
	vector<int>							SubspaceSize;		// subspace/sample size on each level, from level 0 to level n-1
	vector<Eigen::SparseMatrix<double>> U;					// basis matrix on each level
	vector<Eigen::SparseMatrix<double>> Sbar, Mbar;			// reduced system matrices (stiffness + mass matrix) on each level
	vector<Eigen::VectorXd>				geoDist;			// resulting geodesic distance on each level, stored at the last index on each level
	vector<Eigen::VectorXi>				clusters;			// the index of sample point [level k] closest to a certain vertex/sample [level k-1]
	Eigen::VectorXi						sampleLevel;		// to indicate if a vertex is picked as a sample; if yes, at which highest level k




};

#endif // !ML_EIGEN_SOLVER
