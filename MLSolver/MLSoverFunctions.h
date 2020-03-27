#ifndef ML_SOLVER_FUNCTIONS
#define ML_SOLVER_FUNCTIONS

#include "MLEigenSolver.h"

class MLSolverFunctions : public MLEigenSolver
{
public:
	void computeArea();
	void virtual constructMassMatrix();
	void virtual constructStiffnessMatrix();

private:
	double			avgEdgeLength;						// average edge length (from the mesh)
	double			surfaceArea;						// total surface area (from the mesh)

};

#endif // !ML_SOLVER_FUNCTIONS
