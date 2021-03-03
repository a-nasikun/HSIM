#ifndef ML_EIGEN_SOLVER_H
#define ML_EIGEN_SOLVER_H

//#include "Basis.h"
#include "Utility.h"
#include "Sampling.h"
#include "EigenSolver.h"
#include "Visualization.h"

#include <igl/opengl/glfw/Viewer.h>

/* Temporary OpenMesh functionality */
#include<OpenMesh/Core/IO/MeshIO.hh>
#include<OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

namespace HSIM
{

	typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;

	/**
	*  @brief The base class for our multilevel eigenproblem. 
	*  @details
	*	This is the implementation of our hierarchical subspace iteration method (SIM) for geometry processing applications.
	*	In this work, we work specifically with `Laplace--Beltrami Eigenproblem` that is widely adopted for various tasks in geometry processing.
	*	The method proceeds in the following steps:
	*   <ol>
	*		<li> Set up the required matrix for `Laplace--Beltrami Eigenproblem`.
	*		<li> Construct the hierarchy.
	*		<li> Use the hierarchy to solve the `Laplace--Beltrami Eigenproblem` using the *subspace iteration method*.
	*   </ol>
	*
	*/
	class MLEigenSolver
	{
	public:
		MyMesh openMesh;
	public:
		enum class EigenVectorType { EIGEN_REFERENCE, EIGEN_REDUCED, EIGEN_SIM };
		enum class SIMType { SIM_MultiLevel, SIM_1level_WarmStart, SIM_1level_Random, SIM_1level_Bathe };
		enum class EigenSolverType { EIG_SOLVER_MATLAB, EIG_SOLVER_CUDA, EIG_SOLVER_SPECTRA };
		enum class ShiftType { SHIFT_BATHE, SHIFT_ITOH_WILSON, SHIFT_GONG, SHIFT_ZHAO };
		enum class BoundaryCondition {BOUNDARY_NONE, BOUNDARY_DIRICHLET, BOUNDARY_NEUMANN};

		void virtual doEdgeFlip(unsigned int n);
		void virtual checkBoundary();


		void readMesh(const string &meshFile);
		void setInitialSystem(int n);
		void initiateAdM();
		void virtual constructMassMatrix();
		void virtual constructStiffnessMatrix();

		/* Construction of the subspace and/or hierarchy */
		void setSamplingLevelsConstantRate(Eigen::VectorXi& sampSize, int numEigen = 50, double levelRatio = 16.0);
		void setSamplingLevelsAuto(Eigen::VectorXi& sampSize, int numEigen = 50, double levelRatio = 16.0, bool isFixedLowBound=true);
		void setSamplingLevelsSameRate(Eigen::VectorXi& sampSize, int numEigen = 50, double levelRatio = 16.0, bool isFixedLowBound = true);
		void setSamplingLevelFixedNum(Eigen::VectorXi& sampSize, int numEigen = 50, double levelRatio = 16.0, bool isFixedLowBound = true);
		void setSamplingLevelFixedLevels(Eigen::VectorXi& sampSize, int numEigen = 50, int numLevels = 3, bool isFixedLowBound = true);
		void constructSampleHierarchy(const Eigen::VectorXi& samples);	// Constructing multi-level samples
		void virtual constructBasis(double support, int axis = 0, double offset = 0.0);											// Constructing multi-level basis
		void virtual constructBasisCoarseOnly(double support, int axis = 0, double offset = 0.0);
		void virtual constructBasisNonSmooth(double support, int axis = 0, double offset = 0.0);
		void virtual constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support);
		void virtual constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int level, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet);
		void formPartitionOfUnityMultiLevel();
		void formPartitionOfUnity();
		void solveRestrictedProblem();
		void virtual constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis = 0, double offset = 0.0);

		/* Subspace Iteration methods */
		void virtual performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter = 50, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0);
		void virtual performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 20, double tol = 1e-4, bool isWithShift = false);
		void virtual performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 20, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void virtual performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 20, double tol = 1e-4, bool isWithShift = false);
		void virtual performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 20, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void virtual performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 20, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void virtual performMultiLevelSIM_withShiftOnceConverged(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 20, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void virtual SIMinCPU(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol = 1e-4, int maxIter = 20, int k=0);
		void virtual SIMinGPU(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol = 1e-4, int maxIter = 20, int k=0);
		void virtual solve(int numEigs, int maxIter = 50, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0, double levelRatio=10.0, double basisSupport=7.5);
		void virtual solve(int numEigs, double tol = 1e-4, int maxIter = 50, double levelRatio = 10.0, double basisSupport = 7.5, bool isWithShift = true);
		void checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, int num, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& MKinv, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const;
		void checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::MatrixXd& M, int num, bool isWithShift, double mu, Eigen::VectorXd& convRate) const;
		int sturmCheck(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::VectorXd& eigvals);

		void virtual solveBMG(int numEigs, double tol = 1e-4, int maxIter = 50, double levelRatio = 10.0, double basisSupport = 7.5, bool isWithShift = true);
		void virtual solveInversePowerMethod(unsigned int numEigs=50, double tol = 1e-4, int maxIter = 50, bool isWithShift = true);
		void virtual BMGiter(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol = 1e-4, int maxIter = 20, int k = 0);

		/* Computing the reference eigenproblem */
		void computeRefEigenProblem(int numEigs, const EigenSolverType& solverType, int maxIters = 1000, double tol = 1e-10);
		void computeRefEigenProblemWithSelector(int numEigs, const EigenSolverType& solverType, int axis = 0, double offset = 0.0);

		/* Visualization tools */
		void visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k) const;
		void virtual visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k = 0, double scale = 1.0) const;
		void virtual visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale = 1.0) const;
		void visualizeClusterOfSamplesLevel0(igl::opengl::glfw::Viewer &viewer);
		void visualizeClusterOfSamplesLevelK(igl::opengl::glfw::Viewer &viewer, int k);
		void visualizeHierarchy(igl::opengl::glfw::Viewer &viewer, int k=0, int sourceID=0);
		void visualizeHierarchy(igl::opengl::glfw::Viewer &viewer, int levels = 0, int sample1 = 0, int sample2=0);

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
		void setLevel(int n);
		int getSampleSize() const;
		const Eigen::VectorXi& getSample() const;
		const vector<int>& getSubspaceSize() const;
		void setZhaoShiftParam(const double shiftParam);
		double getZhaoShiftParam() const;
		void setQParam(const double value);
		double getQParam() const;
		void setNotIterateOverConvergedFlag(bool flag);
		bool getNotIterateOverConvergedFlag() const;
		void setParallelSolverFlag(bool flag=true);
		bool getParallelSolverFlag() const;
		const BoundaryCondition getBoundaryType() const;
		void setBoundaryType(const BoundaryCondition& boundary);


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

		void cublasGEMMtest();
		void cusparseCSRMMtest();		// old GPU (cuda 8.1?)
		void cusparseSpMMtest();		// modern GPU (CUDA 11.0)
		void cusparseDcsrm2(const Eigen::SparseMatrix<double> SM, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
		void cusolverSP(const Eigen::SparseMatrix<double> SM, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
		void cuSolverCholTest(Eigen::SparseMatrix<double> SM, Eigen::MatrixXd& B, Eigen::MatrixXd& X);
		void cuLinearSolverTest(Eigen::SparseMatrix<double> SM, Eigen::MatrixXd& B, Eigen::MatrixXd& X);
		void cusparseLinearSolverLLTest(Eigen::SparseMatrix<double> SM, Eigen::MatrixXd& B, Eigen::MatrixXd& X);
		void cuSparseProjection(Eigen::SparseMatrix<double>& Basis, Eigen::SparseMatrix<double>& Mat, Eigen::SparseMatrix<double>& MatRed);


	protected:
		/* Mesh-related variables */
		Eigen::MatrixXd						V;							//!< Coordinates of each vertex
		Eigen::MatrixXi						F;							//!< Vertices ID of each face (triangle);

		/* System matrices */
		Eigen::SparseMatrix<double>			S;							//!< Stiffness matrix
		Eigen::SparseMatrix<double>			M;							//!< Mass matrix
		Eigen::SparseMatrix<double>			Minv;						//!< Inverse  of the mass matrix
		Eigen::SparseMatrix<double>			MhNeg, MhPos;						//!< 'Square root' of the mass matrix (M^0.5)
		Eigen::SparseMatrix<double>			G;							//!< Gradient matrix (Each 3 rows corresponds to the x-y-z components of 1 vertex)

		/* Utility matrices, vectors, and scalar */
		Eigen::SparseMatrix<double>			SelMatrix;					//!< Selector matrix for eigen computation
		Eigen::VectorXi						Sample;						//!< Global ID of vertices selected as samples 
		Eigen::VectorXi						SampleMap;					//!< For a certain vertex ID (provided that it's a sample vertex) get its 'sample id' 								
		int									sampleSize;					//!< Total sample size (i.e. right after the finest level sample, level 1)
		std::vector<set<int>>				AdM;						//!< Adjacency matrix (vertex to neighboring vertices)
		std::vector<double>					maxNeighDist;				//!< Geodesic distance for sampling (to control the sparsity), for each level

		/* Subspace and spectral-related matrices, vectors, and variables */
		Eigen::SparseMatrix<double>			Basis;						//!< Basis matrix (U[0]*U[1]*... *U[k])
		bool								isClampedBasis;				//!< Toogle/Flag if the basis is a clamped version
		bool								hasBoundary;				//!< Toogle/Flag if the mesh has a boundary
	public: std::vector<bool>				boundaryVertexCheck;		//!< Check whether a vertex is in a boundary/not
	protected: 
		Eigen::MatrixXd						EigVectsRef;				//!< Reference eigenvectors (from either Matlab, SpectrA, or Manifold Harmonics in Graphite)
		Eigen::MatrixXd						EigVectsReduced;			//!< Approximated eigenvectors (lifted to reference space)
		Eigen::MatrixXd						EigVectsReducedNonLifted;	//!< approximated eigenvectors in the subspace(not Lifted)
		Eigen::VectorXd						EigValsRef;					//!< Reference eigenvalues (from either Matlab, SpectrA, or Manifold Harmonics in Graphite)
		Eigen::VectorXd						EigValsReduced;				//!< Approximated eigenvalues
		Eigen::MatrixXd						EigVectsSIM;				//!< Eigenvectors computed using our multi-level  Subspace Iteration Method (SIM)
		Eigen::VectorXd						EigValsSIM;					//!< Eigenvalues computed using our multi-level Subspace Iteration Method (SIM)

		/* Utility matrix/vector, for performance check */
		Eigen::MatrixXd						EigValsSIMStack;			//!< Stack of eigenvalues, from multiple iterations
		Eigen::MatrixXd						convergence2Stacks;			//!< Stack of convergence rate (convergence criteria2), from multiple iterations
		Eigen::MatrixXd						convergence3Stacks;			//!< Stack of convergence rate (convergence criteria2), from multiple iterations
		Eigen::VectorXd						timingPerLevel;				//!< Timing (in secs) for every level (in multi-level)
		Eigen::VectorXi						itersPerLevel;				//!< Number of iterations per every level (in multi-level)
		
		/* MULTI-GRID STUFF*/
		int											level;				//!< the total number of hierarchy/levels on multigrid
		std::vector<int>							SubspaceSize;		//!< subspace/sample size on each level, from level 0 to level n-1
		std::vector<Eigen::SparseMatrix<double>>	U;					//!< basis matrix on each level
		std::vector<Eigen::SparseMatrix<double>>	Sbar, Mbar;			//!< reduced system matrices (stiffness + mass matrix) on each level
		std::vector<Eigen::VectorXd>				geoDist;			//!< resulting geodesic distance on each level, stored at the last index on each level
		std::vector<Eigen::VectorXi>				clusters;			//!< the index of sample point [level k] closest to a certain vertex/sample [level k-1]
		Eigen::VectorXi								sampleLevel;		//!< to indicate if a vertex is picked as a sample; if yes, at which highest level k

		/* CLASS/OBJECT FLAG */
		bool isConvergedVectsNotUsed = true;
		bool isParallelSolverEnabled = true;
		double zhaoShiftParam = 0.25;
		double qParam = 1.5;

		void virtual subspaceIterationMethodHelper(int k, Eigen::SparseMatrix<double>& SK, Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd XbarK, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int solverStatus, int numEigs, int q, int& iterCount, int maxIter, double tol, Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& factorization, bool isWithShift, bool isConvergedNotUsed = true, int lastConvergedId = 0);
	public:
		// TEst variable
		bool isMultipleSolve = true;
		bool isSIMinGPU = false;
		int hierarchyMode = 2;				// 0: constant (i.e. 10), 1: auto (the more complicated one), 2: same ratio (1 adjusted to be like 0); 3:fixed levels
		int convergedCriteria = 2;			// 2: more expensive: relative L-2 norm of residual && 3: cheaper a^2/qTq
		bool isSampleBounded = true;
		BoundaryCondition boundaryType = BoundaryCondition::BOUNDARY_NONE;
	};
}

#endif // !ML_EIGEN_SOLVER_H

