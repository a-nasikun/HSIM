#ifndef ML_EIGEN_SOLVER
#define ML_EIGEN_SOLVER

//#include "Basis.h"
#include "Utility.h"
#include "Sampling.h"
#include "EigenSolver.h"
#include "Visualization.h"

#include <igl/opengl/glfw/Viewer.h>
namespace MuLES
{

	class MLEigenSolver
	{
	public:
		enum class EigenVectorType { EIGEN_REFERENCE, EIGEN_REDUCED, EIGEN_SIM };
		enum class SIMType { SIM_MultiLevel, SIM_1level_WarmStart, SIM_1level_Random, SIM_1level_Bathe };
		enum class EigenSolverType { EIG_SOLVER_MATLAB, EIG_SOLVER_CUDA, EIG_SOLVER_SPECTRA };
		enum class ShiftType { SHIFT_BATHE, SHIFT_ITOH_WILSON, SHIFT_GONG, SHIFT_ZHAO };

		void readMesh(const string &meshFile);
		void setInitialSystem(int n);
		void initiateAdM();
		void virtual constructMassMatrix();
		void virtual constructStiffnessMatrix();

		/* Construction of the subspace and/or hierarchy */
		void setSamplingLevels(Eigen::VectorXd& sampSize, int numEigen = 50, double levelRatio = 10.0);
		void constructSampleHierarchy(const Eigen::VectorXd& samples);	// Constructing multi-level samples
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
		void virtual performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0);
		void virtual performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false);
		void virtual performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void virtual performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false);
		void virtual performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void virtual performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const;
		void checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, bool isWithShift, double mu, Eigen::VectorXd& convRate) const;

		/* Computing the reference eigenproblem */
		void computeRefEigenProblem(int numEigs, const EigenSolverType& solverType, int maxIters = 1000, double tol = 1e-10);
		void computeRefEigenProblemWithSelector(int numEigs, const EigenSolverType& solverType, int axis = 0, double offset = 0.0);

		/* Visualization tools */
		void visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k) const;
		void virtual visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k = 0, double scale = 1.0) const;
		void virtual visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale = 1.0) const;
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
		void setZhaoShiftParam(const double shiftParam);
		double getZhaoShiftParam() const;

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


	protected:
		/* Mesh-related variables */
		Eigen::MatrixXd						V;							//!< Coordinates of each vertex
		Eigen::MatrixXi						F;							//!< Vertices ID of each face (triangle);

		/* System matrices */
		Eigen::SparseMatrix<double>			S;							//!< Stiffness matrix
		Eigen::SparseMatrix<double>			M;							//!< Mass matrix
		Eigen::SparseMatrix<double>			Minv;						//!< Inverse  of the mass matrix
		Eigen::SparseMatrix<double>			MhNeg;						//!< 'Square root' of the mass matrix (M^0.5)
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


	public:
		// TEst variable
		double zhaoShiftParam;
	};
}

#endif // !ML_EIGEN_SOLVER

