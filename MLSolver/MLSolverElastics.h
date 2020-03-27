#ifndef ML_SOLVER_ELASTICS
#define ML_SOLVER_ELASTICS

#include "MLEigenSolver.h"

class MLSolverElastics : public MLEigenSolver
{
public:


private:
	Eigen::MatrixXi						TF;				// id of the faces of (elastics), for rendering purpose only;
	Eigen::MatrixXd						VB;				// barycenters of each tetrahedron
	Eigen::VectorXd						volume;			// volume of each tetrahedral mesh// tetahedral and faces index

}

#endif // !ML_SOLVER_ELASTICS
