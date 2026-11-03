#ifndef ML_SOLVER_FUNCTIONS
#define ML_SOLVER_FUNCTIONS

#include "Utility.h"
#include "Sampling.h"
#include "EigenSolver.h"

#include <stdexcept>
#include <tuple>

using namespace std;

namespace HSIM {
	class LBEigenSolver
	{
	public:
		LBEigenSolver() = default;
		virtual ~LBEigenSolver() = default;

		enum class EigenVectorType { EIGEN_REFERENCE, EIGEN_REDUCED, EIGEN_SIM };
		enum class SIMType { SIM_MultiLevel, SIM_1level_WarmStart, SIM_1level_Random, SIM_1level_Bathe };
		enum class EigenSolverType { EIG_SOLVER_MATLAB, EIG_SOLVER_CUDA, EIG_SOLVER_SPECTRA };
		enum class ShiftType { SHIFT_BATHE, SHIFT_ITOH_WILSON, SHIFT_GONG, SHIFT_ZHAO };

		void readMesh(const string& meshFile);
		void setup(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes);
		void setInitialSystem(int n);
		void initiateAdM();
		void computeArea();
		void virtual constructMassMatrix();
		void virtual constructStiffnessMatrix();

		/* Main functions for user */
		tuple<Eigen::VectorXd, Eigen::MatrixXd> eigs(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes, double tol = 0.01, int maxIters = 2, double shiftParam = 0.3);
		void buildHierarchy();
		void buildHierarchy(int numModes);
		void buildHierarchy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes);


		/* Construction of the subspace and/or hierarchy */
		void setSamplingLevels(Eigen::VectorXd& sampSize, int numEigen = 50, double levelRatio = 10.0);
		void constructSampleHierarchy(const Eigen::VectorXd& samples);	// Constructing multi-level samples
		void virtual constructBasis(double support, int axis = 0, double offset = 0.0);											// Constructing multi-level basis
		void virtual constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support);
		void virtual constructDijkstraDisk(const int s1, Eigen::VectorXd& D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>>& UTriplet, int& elementCounter);
		void virtual constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis = 0, double offset = 0.0);
		void formPartitionOfUnityMultiLevel();
		void solveRestrictedProblem();

		/* Subspace Iteration methods */
		void performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0);
		void performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
		void checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const;
		void checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, bool isWithShift, double mu, Eigen::VectorXd& convRate) const;

		/* Applications [need to be deleted later] */
		void constructDiffusionTensor(const int numEigs, double t, const Eigen::MatrixXd& EigVects, const Eigen::VectorXd& EigVals, std::vector<Eigen::VectorXd>& DiffusionTensor);

		/* Visualization Tools*/
		void visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer& viewer, int k) const;
		void virtual visualizeBasis(igl::opengl::glfw::Viewer& viewer, int id, int k = 0, double scale = 1.0) const;
		void virtual visualizeEigenvector(igl::opengl::glfw::Viewer& viewer, int id, const EigenVectorType& eigType, double scale = 1.0) const;

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
		
		const Eigen::MatrixXd& getEigenVectorsReduced() const;
		Eigen::VectorXd getEigenVectorsReduced(int id) const;		// throws std::out_of_range
		const Eigen::VectorXd& getEigenValuesReduced() const;
		double getEigenValueReduced(int id) const;					// throws std::out_of_range
		const Eigen::MatrixXd& getEigenVectors() const;
		Eigen::VectorXd getEigenVectors(int id) const;				// throws std::out_of_range
		const Eigen::VectorXd& getEigenValues() const;
		double getEigenValueSIM(int id) const;						// throws std::out_of_range

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
		int									sampleSize = 0;		// sample of the finest level sample, i.e. level 0
		vector<set<int>>					AdM;				// adjacency matrix, vertex to neighboring vertices
		vector<double>						maxNeighDist;		// geodesic distance for sampling, to control the sparsity

		/* Subspace and spectral-related matrices, vectors, and variables */
		Eigen::SparseMatrix<double>			Basis;				// Basis matrix (U[0]*U[1]*... *U[k])
		bool								isClampedBasis = false;	// toogling if the basis is a clamped version
		Eigen::MatrixXd						EigVectsReduced;	// approximated eigenvectors
		Eigen::MatrixXd						EigVectsReducedNonLifted;	// approximated eigenvectors, not Lifted
		Eigen::VectorXd						EigValsReduced;		// approximated eigenvalues
		Eigen::MatrixXd						EigVectsSIM;		// matrix storing eigenvectors, after Subspace Iteration Method (SIM)
		Eigen::VectorXd						EigValsSIM;			// vector stroing eigenvalues,  after Subspace Iteration Method (SIM)
		Eigen::MatrixXd						EigValsSIMStack;	// Stack of eigenvalues, from multiple iterations
		Eigen::MatrixXd						convergence2Stacks;	// Stack of convergence rate (convergence criteria2), from multiple iterations
		Eigen::MatrixXd						convergence3Stacks;	// Stack of convergence rate (convergence criteria2), from multiple iterations
		Eigen::VectorXd						timingPerLevel;		// Timing (in secs) for every level (in multi-level)
		Eigen::VectorXi						itersPerLevel;		// Number of iterations per every level (in multi-level)

		/* [MULTI-GRID STUFF]*/
		int									level = 0;			// the total number of hierarchy/levels on multigrid
		int									numModesHierarchy = 0;	// number of modes the hierarchy was built for
		vector<int>							SubspaceSize;		// subspace/sample size on each level, from level 0 to level n-1
		vector<Eigen::SparseMatrix<double>> U;					// basis matrix on each level
		vector<Eigen::SparseMatrix<double>> Sbar, Mbar;			// reduced system matrices (stiffness + mass matrix) on each level
		vector<Eigen::VectorXd>				geoDist;			// resulting geodesic distance on each level, stored at the last index on each level
		vector<Eigen::VectorXi>				clusters;			// the index of sample point [level k] closest to a certain vertex/sample [level k-1]
		Eigen::VectorXi						sampleLevel;		// to indicate if a vertex is picked as a sample; if yes, at which highest level k

	private:
		double			avgEdgeLength = 0.0;				// average edge length (from the mesh)
		double			surfaceArea = 0.0;					// total surface area (from the mesh)

		// TEst variable
	public:
		double zhaoShiftParam = 0.25;						// relative position of the shift in [0,1), see eigs()
	};

}

#endif // !ML_SOLVER_FUNCTIONS
