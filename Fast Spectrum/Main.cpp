#include "FastSpectrum.h"
#include "FastSpectrumGUI.h"
#include <igl/unproject_onto_mesh.h>

/* [MAIN FUNCTION THAT CALLS EVERYTHING ELSE] */
int main(int argc, char *argv[])
{
	Eigen::SparseMatrix<double> Basis;
	Eigen::MatrixXd redEigVects, V;
	Eigen::VectorXd redEigVals;
	Eigen::MatrixXi F;
	string meshFile; 
	
	/* GUI FUNCTIONALITY */	
	igl::opengl::glfw::Viewer			viewer;			// Set up main viewer
	FastSpectrum						fastSpectrum;
	igl::opengl::glfw::imgui::ImGuiMenu menu;			// Attach a menu plugin
	viewer.plugins.push_back(&menu);	
	//IMGUI_CHECKVERSION();								// Setup Dear ImGui binding
	ImGui::CreateContext();
	ImGui::StyleColorsDark();							// Set style

	//meshFile = "../Models/AIM894_Chinese Dragon/894_Chinese Dragon.obj";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Cube/Cube_sharp_50k_2.obj";
	meshFile = "D:/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/dragon/Dragon.obj";

	fastSpectrum.setInitialSystem(2);
	fastSpectrum.computeEigenPairs(meshFile, 1000, Basis, redEigVects, redEigVals);
	fastSpectrum.getV(V);
	fastSpectrum.getF(F);
			
	menu.callback_draw_viewer_window = [&]()
	{
		showMenu(viewer, menu, fastSpectrum);
		applicationMenu(viewer, menu, fastSpectrum);
	};
	/* END OF GUI FUNCTIONALITY */

	/* User interaction via keyboard */
	const auto &key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned char key, int mod)->bool
	{

		/* For selecting faces */
		double xMouse;
		double yMouse;
		int fid;
		Eigen::Vector3f bc;
		int selectedFace;
		//Eigen::MatrixXd C;

		switch (key)
		{
		case '0':
			viewer.data().clear();
			fastSpectrum.getV(V);
			fastSpectrum.getF(F);
			viewer.data().set_mesh(V, F);
			viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
			break;
		case 'n':
		case 'N':
			//selectFace = true; 
			xMouse = viewer.current_mouse_x;
			yMouse = viewer.core.viewport(3) - viewer.current_mouse_y;

			//if (selectFace) {
			if (igl::unproject_onto_mesh(Eigen::Vector2f(xMouse, yMouse), viewer.core.view,
				viewer.core.proj, viewer.core.viewport, V, F, selectedFace, bc))
			{
				// paint hit red
				printf("Selected face is %d [%d, %d, %d]", selectedFace, F(selectedFace,0), F(selectedFace,1), F(selectedFace,2));
				Eigen::MatrixXd C = Eigen::MatrixXd::Constant(F.rows(), 3, 1);
				C.row(selectedFace) << 1, 0, 0;
				viewer.data().set_colors(C);
				//viewer.data().add_points(vectorFields.FC.row(selectedFace), Eigen::RowVector3d(0.0, 0.1, 0.0));

				return true;
			}
			//}
			break;


		default:
			return false;
		}
		//viewer.data.set_vertices(V);
		viewer.data().compute_normals();
		//viewer.core.align_camera_center(V, F);
		return true;
	};

	viewer.callback_init			= [&](igl::opengl::glfw::Viewer &viewer){return false;};
	viewer.data().set_mesh(V, F);
	viewer.callback_key_down		= key_down;

	//Eigen::Vector4f					bgCol(0.5f, 0.4f, 0.9f, 0.2f);
	//Eigen::Vector4f				bgCol(0.95f, 0.95f, 0.95f, 1.0f);
	Eigen::Vector4f				bgCol(1.0f, 1.0f, 1.0f, 1.0f);
	viewer.core.background_color	= bgCol;	
	viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
	viewer.data().point_size		= 10.0f;
	viewer.data().line_width		= 0.5f; 	
	return viewer.launch();
}
