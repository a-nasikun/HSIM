#ifndef ML_SOLVER_ELASTICS
#define ML_SOLVER_ELASTICS

#include "MLEigenSolver.h"

class MLSolverElastics : public MLEigenSolver
{
public:
	void createTetrahedralMesh();
	void computeVolume();	
	void virtual constructMassMatrix();	
	void setLameParametersConstant(double lame1, double lame2);
	void setLameParametersSlices(int axis, int numSlices, double m1l1, double m1l2, double m2l1, double m2l2, Eigen::VectorXd& vColor);
	void setLameParametersBox(int axis, double clampRatio, double m1l1, double m1l2, double m2l1, double m2l2, Eigen::VectorXd& vColor);
	void virtual constructStiffnessMatrix();


	void virtual constructBasis(double support, int axis = 0, double offset = 0.0);
	void virtual constructBasis_LBS(double support, int axis = 0, double offset = 0.0);
	void virtual constructBasisCoarseOnly(double support, int axis = 0, double offset = 0.0);
	void virtual constructBasisCoarseOnly_LBS(double support, int axis = 0, double offset = 0.0);
	void virtual constructBasisNonSmooth(double support, int axis = 0, double offset = 0.0);
	void virtual constructBasisNonSmooth_LBS(double support, int axis = 0, double offset = 0.0);
	void virtual constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support);
	void virtual constructRadialBasisFunctions_LBS(int k, int axis, bool isClamped, double offset, double support);
	void virtual constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter);
	void virtual constructDijkstraDisk_LBS(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter);
	void virtual constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis = 0, double offset = 0.0);
	void formPartitionOfUnity();
	void formPartitionOfUnityMultiLevel();
	void injectRigidModesInBasis();
	void injectRigidModes();

	/* Subspace Iteration methods */
	void performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE, int axis = 0, double offset = 0.0);
	void performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false);
	void performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	void performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = false);
	void performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	void performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter = 10, double tol = 1e-4, bool isWithShift = true, const ShiftType& shiftType = ShiftType::SHIFT_BATHE);
	

	
	/* Visualization Tools*/
	void virtual visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, double scale = 1.0) const;
	void virtual visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale = 1.0) const;

	/* Some getters */
	Eigen::MatrixXi getTF() const;
	Eigen::MatrixXd getVB() const;
private:
	Eigen::MatrixXi						TF;				// id of the faces of (elastics), for rendering purpose only;
	Eigen::MatrixXd						VB;				// barycenters of each tetrahedron
	Eigen::VectorXd						volume;			// volume of each tetrahedral mesh
	Eigen::MatrixXd						Lame;			// Lame parameters of each tet-mesh

};

#endif // !ML_SOLVER_ELASTICS
