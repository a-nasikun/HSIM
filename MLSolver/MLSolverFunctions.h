#ifndef ML_SOLVER_FUNCTIONS
#define ML_SOLVER_FUNCTIONS

#include "MLEigenSolver.h"

class MLSolverFunctions : public MLEigenSolver
{
public:
	void computeArea();
	void virtual constructMassMatrix();
	void virtual constructStiffnessMatrix();

	void virtual constructBasis(double support, int axis = 0, double offset = 0.0);											// Constructing multi-level basis
	void virtual constructBasisCoarseOnly(double support, int axis = 0, double offset = 0.0);
	void virtual constructBasisNonSmooth(double support, int axis = 0, double offset = 0.0);
	void virtual constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support);
	void virtual constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter);
	void virtual constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis = 0, double offset = 0.0);
	
	/* Subspace Iteration methods */
	void performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0);
	void performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false);
	void performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	void performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false);
	void performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	void performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);

	/* Applications */
	void constructDiffusionTensor(const int numEigs, double t, const Eigen::MatrixXd &EigVects, const Eigen::VectorXd& EigVals, std::vector<Eigen::VectorXd> &DiffusionTensor);

	/* Visualization Tools*/
	void virtual visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k=0, double scale = 1.0) const;
	void virtual visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale = 1.0) const;
private:
	double			avgEdgeLength;						// average edge length (from the mesh)
	double			surfaceArea;						// total surface area (from the mesh)
};

#endif // !ML_SOLVER_FUNCTIONS
