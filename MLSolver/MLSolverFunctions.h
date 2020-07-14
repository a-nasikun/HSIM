#ifndef ML_SOLVER_FUNCTIONS
#define ML_SOLVER_FUNCTIONS

#include "MLEigenSolver.h"
namespace MuLES
{
	/**
	*  @brief Multilevel solver for `Laplace--Beltrami eigenproblem`.
	*  @details
	*	In this work, we work specifically with `Laplace--Beltrami Eigenproblem` that is widely adopted for various tasks in geometry processing.
	*	This class contains specific **methods** and **variables** that are unique to the `Laplace--Beltrami eigenproblem`.
	*	The method proceeds in the following steps:
	*   <ol>
	*		<li> Set up the required matrix for `Laplace--Beltrami Eigenproblem`.
	*		<li> Construct the hierarchy.
	*		<li> Use the hierarchy to solve the `Laplace--Beltrami Eigenproblem` using the **subspace iteration method**.
	*   </ol>
	*
	*
	*	### Example of usage
	*	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.cpp
	*	MLSolverFunctions *solver;			// Create a pointer for the object
	*	solver->readMesh(fileLocation);		// Read a mesh
	*	solver->computeEigenpairs(A,B);		// Compute the eigenvalues and eigenvectors
	*	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	*/
	class MLSolverFunctions : public MLEigenSolver
	{
	public:
		void computeArea();
		void virtual constructMassMatrix() override;
		void virtual constructStiffnessMatrix() override;

		void virtual constructBasis(double support, int axis = 0, double offset = 0.0) override;
		void virtual constructBasisCoarseOnly(double support, int axis = 0, double offset = 0.0) override;
		void virtual constructBasisNonSmooth(double support, int axis = 0, double offset = 0.0) override;
		void virtual constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support) override;
		void virtual constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet) override;
		void virtual constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis = 0, double offset = 0.0) override;

		/* Subspace Iteration methods */
		void performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0) override;
		void performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false) override;
		void performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE) override;
		void performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false) override;
		void performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE) override;
		void performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE) override;
		void performMultiLevelSIM_withShiftOnceConverged(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE) override;
		void virtual solve(int numEigs, int maxIter = 50, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0, double levelRatio = 10.0, double basisSupport=7.5) override;

		/* Applications */
		void constructDiffusionTensor(const int numEigs, double t, const Eigen::MatrixXd &EigVects, const Eigen::VectorXd& EigVals, std::vector<Eigen::VectorXd> &DiffusionTensor);

		/* Visualization Tools*/
		void virtual visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k = 0, double scale = 1.0) const override;
		void virtual visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale = 1.0) const override;

		

	private:
		double			avgEdgeLength;						//!< Average edge length (from the mesh)
		double			surfaceArea;						//!< Total surface area (from the mesh)


		void subspaceIterationMethodHelper(int k, Eigen::SparseMatrix<double>& SK, Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd XbarK,  Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int solverStatus, int numEigs, int q, int& iterCount, int maxIter, double tol, Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& factorization, bool isWithShift, bool isConvergedNotUsed = true, int lastConvergedId=0) override;
		int SIMiterationInCPU(const Eigen::MatrixXd& XbarK, const Eigen::SparseMatrix<double>& SK, const Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int& GPUeigsolverStatus);
		int SIMiterationInGPU(const Eigen::MatrixXd& XbarK, const Eigen::SparseMatrix<double>& SK, const Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int iterCount, int& GPUeigsolverStatus);
		void constructDiagonalMassMatrix();
		void constructConsistentMassMatrix();
	};
}
#endif // !ML_SOLVER_FUNCTIONS
