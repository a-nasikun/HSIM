#include "MLSoverFunctions.h"

#include <igl/doublearea.h>
#include <igl/invert_diag.h>
#include <igl/avg_edge_length.h>


void MLSolverFunctions::computeArea()
{
	/* Constructing Average length of the edges */
	avgEdgeLength = igl::avg_edge_length(V, F);

	Eigen::VectorXd dArea;
	igl::doublearea(V, F, dArea);	
	surfaceArea = dArea.sum() / 2.0;

	//surfaceArea = 0.0;
	//for (int i = 0; i < F.rows(); i++){
	//	surfaceArea += dArea(i) / 2.0;
	//}
}

void MLSolverFunctions::constructMassMatrix()
{
	/* Construct Mass matrix M*/
	//igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
	igl::invert_diag(M, Minv);
}
