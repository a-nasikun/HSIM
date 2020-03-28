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
	void virtual constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support);
	void virtual constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter);

private:
	double			avgEdgeLength;						// average edge length (from the mesh)
	double			surfaceArea;						// total surface area (from the mesh)

};

#endif // !ML_SOLVER_FUNCTIONS
