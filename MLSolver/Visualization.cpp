#include "Visualization.h"
#include "Utility.h"
#include <igl/jet.h>

using namespace std;

/* Visualize the samples */
void visualizeSample(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, double nEps)
{
	//viewer.data().clear();
	//viewer.data().set_mesh(V, F);
	int lastID = viewer.data().points.rows();
	viewer.data().points.conservativeResize(lastID + Sample.size(), Eigen::NoChange_t());

	for (int i = 0; i < Sample.size(); i++)
	{
		//viewer.data().add_points(V.row(Sample(i)), Eigen::RowVector3d(1.0, 0.1, 0.0));
		viewer.data().points.row(lastID + i) << V.row(Sample(i)) + nEps*viewer.data().V_normals.row(Sample(i)), Eigen::RowVector3d(1.0, 0.1, 0.0);
	}
}

void visualizeSampleMulti(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const vector<int> &sampleSize, double nEps, double scaleRatio, int sample1, int sample2)
{
	viewer.data_list.clear();
	viewer.data_list.shrink_to_fit();
	viewer.append_mesh();
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));

	Eigen::MatrixXd MCol;
	const int totLevel = sampleSize.size();
	cout << "Visualizing the samples \n";
	cout << "Total level: " << totLevel << endl;

	Eigen::Vector3d maxVal;
	maxVal << viewer.data().V.col(0).maxCoeff() - viewer.data().V.col(0).minCoeff(), viewer.data().V.col(1).maxCoeff() - viewer.data().V.col(1).minCoeff(), viewer.data().V.col(2).maxCoeff() - viewer.data().V.col(2).minCoeff();
	double pointSphereScale = scaleRatio * maxVal.maxCoeff(); 
	//0.0125 for sphere
	string sphereModelFile = "../Models/small_sphere/small_sphere.obj";
	viewer.load_mesh_from_file(sphereModelFile);

	// Specifying the color for each level
	if (totLevel < 10) 
	{
	//
	//	/* Predefined color*/
		MCol.resize(9, 3);
		MCol.row(0)  = Eigen::RowVector3d(228.0,  26.0,  28.0) / 255.0;		
	 	//MCol.row(1)  = Eigen::RowVector3d( 55.0, 126.0, 184.0) / 255.0;			// default 	
		//MCol.row(1)  = Eigen::RowVector3d( 55.0,  95.0, 184.0) / 255.0;			// most parts on sampling
		MCol.row(1)  = Eigen::RowVector3d(108.0, 212.0, 255.0) / 255.0;			// visualization of geodesic disk
		MCol.row(2)  = Eigen::RowVector3d( 77.0, 175.0,  74.0) / 255.0;				
	 	MCol.row(3)  = Eigen::RowVector3d(152.0,  78.0, 163.0) / 255.0;				
	 	MCol.row(4)  = Eigen::RowVector3d(255.0, 127.0,   0.0) / 255.0;
	 	MCol.row(5)  = Eigen::RowVector3d(255.0, 255.0,  51.0) / 255.0;
	 	MCol.row(6)  = Eigen::RowVector3d(166.0,  86.0,  40.0) / 255.0;
	 	MCol.row(7)  = Eigen::RowVector3d(247.0, 129.0, 191.0) / 255.0;
	 	MCol.row(8)  = Eigen::RowVector3d(153.0, 153.0, 153.0) / 255.0;		
		//std::cout << "Color: \n" << 255.0*MCol << std::endl;
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
	//
	Eigen::RowVector3d sphereCol1(221.0/255.0, 4.0/255.0, 38.0/255.0);
	std::cout << "Loading Mesh \n";
	std::cout << "Num of mesh: " << viewer.data_list.size() << std::endl;
	
	//for (int i = 0; i < 30; i++)
	for (int i = 0; i < sampleSize[totLevel-1]; i++)
	{
		double spScale = 1.0;
		if (i == sample1 || i == sample2) spScale = 1.5;
		viewer.append_mesh();		
		viewer.data_list[i + 1].set_mesh(viewer.data_list[1].V, viewer.data_list[1].F);
		HSIM::scaleMesh(viewer.data_list[i + 1].V, viewer.data_list[i + 1].F, spScale*pointSphereScale);
		Eigen::MatrixXd transVect(viewer.data_list[i+1].V.rows(),3);
		for(int j=0; j<viewer.data_list[i + 1].V.rows(); ++j)			
			transVect.row(j) = viewer.data_list[0].V.row(Sample(i)) + nEps*viewer.data_list[0].V_normals.row(Sample(i));

		viewer.data_list[i+1].V = viewer.data_list[i+1].V + transVect;
		viewer.data_list[i+1].set_colors(MCol.row(0));
		viewer.data_list[i+1].show_lines = false;
	}
	std::cout << "Num of mesh: " << viewer.data_list.size() << std::endl;

	Eigen::RowVector3d sphereCol2(29.0/255.0, 51.0/255.0, 84.0/255.0);
	for (int k = (totLevel-1); k > 0; k--) {
	///for (int k = (totLevel - 1); k > 1; k--) {
		std::cout << "Level" << k << std::endl;		
		for (int i = sampleSize[k]; i < sampleSize[k-1]; i++)
		{
			double spScale = 1.0;
			if (i == sample1 || i == sample2) spScale = 1.5;

			viewer.append_mesh();
			viewer.data_list[i + 1].set_mesh(viewer.data_list[1].V, viewer.data_list[1].F);

			HSIM::scaleMesh(viewer.data_list[i+1].V, viewer.data_list[i+1].F, spScale * pointSphereScale);
			Eigen::MatrixXd transVect(viewer.data_list[i+1].V.rows(), 3);
			for (int j = 0; j<viewer.data_list[i+1].V.rows(); ++j)
				transVect.row(j) = viewer.data_list[0].V.row(Sample(i)) + nEps*viewer.data_list[0].V_normals.row(Sample(i));
			viewer.data_list[i+1].V = viewer.data_list[i+1].V + transVect;
			//viewer.data_list[i+1].set_colors(sphereCol2);
			viewer.data_list[i + 1].set_colors(MCol.row(totLevel-k));
			viewer.data_list[i+1].show_lines = false;
			//viewer.data().add_points(V.row(Sample(i)) + nEps*viewer.data().V_normals.row(Sample(i)), MCol.row(totLevel - k));
		}
		std::cout << "Num of mesh: " << viewer.data_list.size() << std::endl;
	}	
	viewer.selected_data_index = 0;
	viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
}

//void visualizeSampleMulti(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXi& Sample, const vector<int>& sampleSize, double nEps, double scaleRatio, int sample1, int sample2)
//{
//	Eigen::MatrixXd MCol;
//	const int totLevel = sampleSize.size();
//	cout << "Visualizing the samples \n";
//	cout << "Total level: " << totLevel << endl;
//
//	Eigen::Vector3d maxVal;
//	maxVal << viewer.data().V.col(0).maxCoeff() - viewer.data().V.col(0).minCoeff(), viewer.data().V.col(1).maxCoeff() - viewer.data().V.col(1).minCoeff(), viewer.data().V.col(2).maxCoeff() - viewer.data().V.col(2).minCoeff();
//	double pointSphereScale = scaleRatio * maxVal.maxCoeff();
//	//0.0125 for sphere
//	string sphereModelFile = "../Models/small_sphere/small_sphere.obj";
//
//	if (totLevel < 12)
//	{
//		//
//		//	/* Predefined color*/
//		MCol.resize(11, 3);
//		MCol.row(0) = Eigen::RowVector3d(221.0 / 255.0, 4.0 / 255.0, 38.0 / 255.0);
//		MCol.row(1) = Eigen::RowVector3d(29.0 / 255.0, 51.0 / 255.0, 84.0 / 255.0);
//		//MCol.row(2)  = Eigen::RowVector3d( 77.0, 175.0,  74.0) / 255.0;
//		MCol.row(2) = Eigen::RowVector3d(55.0, 126.0, 184.0) / 255.0;
//		MCol.row(3) = Eigen::RowVector3d(152.0, 78.0, 163.0) / 255.0;
//		MCol.row(4) = Eigen::RowVector3d(255.0, 127.0, 0.0) / 255.0;
//		MCol.row(5) = Eigen::RowVector3d(255.0, 255.0, 51.0) / 255.0;
//		MCol.row(6) = Eigen::RowVector3d(166.0, 86.0, 40.0) / 255.0;
//		MCol.row(7) = Eigen::RowVector3d(247.0, 129.0, 191.0) / 255.0;
//		MCol.row(8) = Eigen::RowVector3d(153.0, 153.0, 153.0) / 255.0;
//		MCol.row(9) = Eigen::RowVector3d(10.0, 10.0, 10.0) / 255.0;
//		MCol.row(10) = Eigen::RowVector3d(178.0, 223.0, 138.0) / 255.0;
//
//		std::cout << "Color: \n" << 255.0 * MCol << std::endl;
//	}
//	else
//	{
//		Eigen::VectorXd vCol(totLevel);
//		//Eigen::MatrixXd MCol;
//		for (int i = 0; i < totLevel; i++) vCol(i) = i;
//		igl::jet(vCol, true, MCol);
//	}
//	printf("Total sample: %d \n", Sample.size());
//	for (int i = 0; i < totLevel; i++) printf("__level %d = %d \n", i, sampleSize[i]);
//
//
//	//int lastID = viewer.data().points.rows();
//	//printf("Row: %d \n", lastID);
//	//viewer.data().points.conservativeResize(lastID + Sample.size(), 6);
//	//printf("Data points size: %dx%d \n", viewer.data().points.rows(), viewer.data().points.cols());
//
//	/* Lowest level */
//	//
//	Eigen::RowVector3d sphereCol1(221.0 / 255.0, 4.0 / 255.0, 38.0 / 255.0);
//	std::cout << "Loading Mesh \n";
//	std::cout << "Num of mesh: " << viewer.data_list.size() << std::endl;
//
//	//for (int i = 0; i < 30; i++)
//	for (int i = 0; i < sampleSize[totLevel - 1]; i++)
//	{
//		double spScale = 1.0;
//		if (i == sample1 || i == sample2) spScale = 1.5;
//		viewer.load_mesh_from_file(sphereModelFile);
//		HSIM::scaleMesh(viewer.data_list[i + 1].V, viewer.data_list[i + 1].F, spScale * pointSphereScale);
//		Eigen::MatrixXd transVect(viewer.data_list[i + 1].V.rows(), 3);
//		for (int j = 0; j < viewer.data_list[i + 1].V.rows(); ++j)
//			transVect.row(j) = viewer.data_list[0].V.row(Sample(i)) + nEps * viewer.data_list[0].V_normals.row(Sample(i));
//
//		viewer.data_list[i + 1].V = viewer.data_list[i + 1].V + transVect;
//		viewer.data_list[i + 1].set_colors(sphereCol1);
//		viewer.data_list[i + 1].show_lines = false;
//		//viewer.data().add_points(V.row(Sample(i)) + nEps*viewer.data().V_normals.row(Sample(i)), MCol.row(0));
//		//if (i == 0)
//		//	std::cout << "Color: " << MCol.row(0)*255.0 << std::endl;
//	}
//	std::cout << "Num of mesh: " << viewer.data_list.size() << std::endl;
//
//	Eigen::RowVector3d sphereCol2(29.0 / 255.0, 51.0 / 255.0, 84.0 / 255.0);
//	for (int k = (totLevel - 1); k > 0; k--) {
//		std::cout << "Level" << k << std::endl;
//		for (int i = sampleSize[k]; i < sampleSize[k - 1]; i++)
//		{
//			double spScale = 1.0;
//			if (i == sample1 || i == sample2) spScale = 1.5;
//
//			viewer.load_mesh_from_file(sphereModelFile);
//			HSIM::scaleMesh(viewer.data_list[i + 1].V, viewer.data_list[i + 1].F, spScale * pointSphereScale);
//			Eigen::MatrixXd transVect(viewer.data_list[i + 1].V.rows(), 3);
//			for (int j = 0; j < viewer.data_list[i + 1].V.rows(); ++j)
//				transVect.row(j) = viewer.data_list[0].V.row(Sample(i)) + nEps * viewer.data_list[0].V_normals.row(Sample(i));
//			viewer.data_list[i + 1].V = viewer.data_list[i + 1].V + transVect;
//			viewer.data_list[i + 1].set_colors(sphereCol2);
//			viewer.data_list[i + 1].show_lines = false;
//			//viewer.data().add_points(V.row(Sample(i)) + nEps*viewer.data().V_normals.row(Sample(i)), MCol.row(totLevel - k));
//		}
//		std::cout << "Num of mesh: " << viewer.data_list.size() << std::endl;
//	}
//	viewer.selected_data_index = 0;
//	viewer.core.align_camera_center(viewer.data().V, viewer.data().F);
//
//}

void scaleSampleMulti(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const vector<int> &sampleSize, double nEps, double scaleRatio)
{
	std::cout << "Scaling the points\n";
	Eigen::Vector3d maxVal;
	maxVal << viewer.data().V.col(0).maxCoeff() - viewer.data().V.col(0).minCoeff(), viewer.data().V.col(1).maxCoeff() - viewer.data().V.col(1).minCoeff(), viewer.data().V.col(2).maxCoeff() - viewer.data().V.col(2).minCoeff();

	double pointSphereScale = scaleRatio * maxVal.maxCoeff();
	for (int i = 0; i < viewer.data_list.size()-1; ++i) {
		if (i == 0) {
			cout << "Prior scaling: " << viewer.data_list[i + 1].V.maxCoeff() << std::endl;
			cout << viewer.data_list[i + 1].V.row(0) << std::endl;
		}
		Eigen::MatrixXd transVect(viewer.data_list[i + 1].V.rows(), 3);
		for (int j = 0; j<viewer.data_list[i + 1].V.rows(); ++j)
			transVect.row(j) = viewer.data_list[0].V.row(Sample(i)) + nEps*viewer.data_list[0].V_normals.row(Sample(i));

		viewer.data_list[i + 1].V = viewer.data_list[i + 1].V - transVect;
		if (i == 0) {
			cout << "Prior scaling: " << viewer.data_list[i + 1].V.maxCoeff() << std::endl;
			cout << viewer.data_list[i + 1].V.row(0) << std::endl;
		}
		HSIM::scaleMesh(viewer.data_list[i + 1].V, viewer.data_list[i + 1].F, pointSphereScale);
		if (i == 0) {
			cout << "After scaling: " << viewer.data_list[i + 1].V.maxCoeff() << std::endl;
			cout << viewer.data_list[i + 1].V.row(0) << std::endl;
		}
		viewer.data_list[i + 1].V = viewer.data_list[i + 1].V + transVect;
		if (i == 0) {
			cout << "After scaling: " << viewer.data_list[i + 1].V.maxCoeff() << std::endl;
			printf("ID %d of %d\n", i, viewer.data_list.size());
			cout << viewer.data_list[i + 1].V.row(0) << std::endl;
		}
	}
}

void visualizeSample_1(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps)
{
	Eigen::MatrixXd MCol(1,3);
	MCol.row(0) = Eigen::RowVector3d(228.0, 26.0, 28.0) / 255.0;
	viewer.data().add_points(V.row(Sample(0)) + nEps*viewer.data().V_normals.row(Sample(0)), MCol.row(0));
}
void visualizeSample_2(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps)
{
	Eigen::MatrixXd MCol(1, 3);
	MCol.row(0) = Eigen::RowVector3d(228.0, 26.0, 28.0) / 255.0;
	viewer.data().add_points(V.row(Sample(0)) + nEps*viewer.data().V_normals.row(Sample(0)), MCol.row(0));
	viewer.data().add_points(V.row(Sample(1)) + nEps*viewer.data().V_normals.row(Sample(1)), MCol.row(0));
	//viewer.data().add_points(V.row(Sample(0)), MCol.row(0));
	//viewer.data().add_points(V.row(Sample(1)), MCol.row(0));
}
void visualizeSample_3(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps)
{
	Eigen::MatrixXd MCol(1, 3); 
	MCol.row(0) = Eigen::RowVector3d(228.0, 26.0, 28.0) / 255.0;
	viewer.data().add_points(V.row(Sample(0)) + nEps*viewer.data().V_normals.row(Sample(0)), MCol.row(0));
	viewer.data().add_points(V.row(Sample(1)) + nEps*viewer.data().V_normals.row(Sample(1)), MCol.row(0));
	viewer.data().add_points(V.row(Sample(2)) + nEps*viewer.data().V_normals.row(Sample(2)), MCol.row(0));
}
void visualizeSampleFirstLevel(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps)
{
	Eigen::MatrixXd MCol(1, 3);
	MCol.row(0) = Eigen::RowVector3d(228.0, 26.0, 28.0) / 255.0;
	const int totLevel = sampleSize.size();
	for (int i = 0; i < sampleSize[totLevel - 1]; i++)
	{
		viewer.data().add_points(V.row(Sample(i)) + nEps*viewer.data().V_normals.row(Sample(i)), MCol.row(0));
		if (i == 0)
			std::cout << "Color: " << MCol.row(0)*255.0 << std::endl;
	}
}
void visualizeSampleSecondLevel(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps)
{
	Eigen::MatrixXd MCol(2, 3);
	MCol.row(0) = Eigen::RowVector3d(228.0, 26.0, 28.0) / 255.0;
	MCol.row(1) = Eigen::RowVector3d(77.0, 225.0, 74.0) / 255.0;
	const int totLevel = sampleSize.size();
	for (int i = 0; i < sampleSize[totLevel - 1]; i++)
	{
		viewer.data().add_points(V.row(Sample(i)) + nEps*viewer.data().V_normals.row(Sample(i)), MCol.row(0));
		if (i == 0)
			std::cout << "Color: " << MCol.row(0)*255.0 << std::endl;
	}
	for (int k = (totLevel - 1); k >(totLevel - 2); k--) {
		for (int i = sampleSize[k]; i < sampleSize[k - 1]; i++)
		{
			viewer.data().add_points(V.row(Sample(i)) + nEps*viewer.data().V_normals.row(Sample(i)), MCol.row(1));
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