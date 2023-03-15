#include "Visualization.h"

using namespace std;

/* Visualize the samples */
void visualizeSample(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample)
{
	//viewer.data().clear();
	//viewer.data().set_mesh(V, F);
	int lastID = viewer.data().points.rows();
	viewer.data().points.conservativeResize(lastID + Sample.size(), Eigen::NoChange_t());

	for (int i = 0; i < Sample.size(); i++)
	{
		//viewer.data().add_points(V.row(Sample(i)), Eigen::RowVector3d(1.0, 0.1, 0.0));
		viewer.data().points.row(lastID + i) << V.row(Sample(i)), Eigen::RowVector3d(1.0, 0.1, 0.0);
	}
}

void visualizeSampleMulti(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const vector<int> &sampleSize)
{
	Eigen::MatrixXd MCol;
	const int totLevel = sampleSize.size();
	cout << "Visualizing the samples \n";
	cout << "Total level: " << totLevel << endl;

	if (totLevel < 12) 
	{
	//
	//	/* Predefined color*/
		MCol.resize(11, 3);
		MCol.row(0)  = Eigen::RowVector3d(228.0,  26.0,  28.0) / 255.0;
	 	MCol.row(1)  = Eigen::RowVector3d( 55.0, 126.0, 184.0) / 255.0;
	 	//MCol.row(2)  = Eigen::RowVector3d( 77.0, 175.0,  74.0) / 255.0;
		MCol.row(2) = Eigen::RowVector3d(77.0, 225.0, 74.0) / 255.0;
	 	MCol.row(3)  = Eigen::RowVector3d(152.0,  78.0, 163.0) / 255.0;
	 	MCol.row(4)  = Eigen::RowVector3d(255.0, 127.0,   0.0) / 255.0;
	 	MCol.row(5)  = Eigen::RowVector3d(255.0, 255.0,  51.0) / 255.0;
	 	MCol.row(6)  = Eigen::RowVector3d(166.0,  86.0,  40.0) / 255.0;
	 	MCol.row(7)  = Eigen::RowVector3d(247.0, 129.0, 191.0) / 255.0;
	 	MCol.row(8)  = Eigen::RowVector3d(153.0, 153.0, 153.0) / 255.0;
	 	MCol.row(9)  = Eigen::RowVector3d( 10.0,  10.0,  10.0) / 255.0;
	 	MCol.row(10) = Eigen::RowVector3d(178.0, 223.0, 138.0) / 255.0;
		
		std::cout << "Color: \n" << 255.0*MCol << std::endl;
	}
	else 
	{
		Eigen::VectorXd vCol(totLevel);
		//Eigen::MatrixXd MCol;
		for (int i = 0; i < totLevel; i++) vCol(i) = i;
		igl::jet(vCol, true, MCol);
	}
	printf("Total sample: %d \n", Sample.size());
	for (int i = 0; i < totLevel; i++) printf("__level %d = %d \n", i, sampleSize[i]);	


	//int lastID = viewer.data().points.rows();
	//printf("Row: %d \n", lastID);
	//viewer.data().points.conservativeResize(lastID + Sample.size(), 6);
	//printf("Data points size: %dx%d \n", viewer.data().points.rows(), viewer.data().points.cols());

	/* Lowest level */
	for (int i = 0; i < sampleSize[totLevel-1]; i++)
	{
		viewer.data().add_points(V.row(Sample(i)), MCol.row(0));
		if (i == 0)
			std::cout << "Color: " << MCol.row(0)*255.0 << std::endl;
	}

	for (int k = (totLevel-1); k > 0; k--) {
		for (int i = sampleSize[k]; i < sampleSize[k-1]; i++)
		{
			viewer.data().add_points(V.row(Sample(i)), MCol.row(k));
		}
	}	
}

/* Visualize basis */
void visualizeBasis(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::SparseMatrix<double> &U, const int &basisID)
{
	//viewer.data().clear();
	//viewer.data().set_mesh(V, F);

	// Map the basis value at particular index to the jet coloring scheme
	Eigen::MatrixXd		vColor;
	Eigen::VectorXd		Z;
	Z					= U.col(basisID);
	igl::jet(Z, true, vColor);
	viewer.data().set_colors(vColor);
}

/* Visualize Approximated Eigenvector */
void visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::SparseMatrix<double> &U, const Eigen::MatrixXd &LDEigVec, const int &eigenID)
{
	//viewer.data().clear();
	//viewer.data().set_mesh(V, F);

	// Some validation, ensuring no out-of-bound indices
	if(eigenID<0 || eigenID>LDEigVec.cols()-1) { 
		printf("Error! The index should be between 0 and %d.\n", LDEigVec.cols()-1);
		return;
	}

	// Map the eigenvectors as the color of the mesh (on each of the vertices)
	Eigen::MatrixXd		vColor;
	Eigen::VectorXd		Z;

	Z = U * LDEigVec.col(eigenID);
	if (eigenID == 0) {
		igl::jet(Z, false, vColor);
	}else {
		igl::jet(Z, true, vColor);
	}
	viewer.data().set_colors(vColor);
}