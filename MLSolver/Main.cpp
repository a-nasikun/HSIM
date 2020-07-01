#include "MLEigenSolver.h"
#include "MLSolverFunctions.h"
#include "MLEigSolverGUI.h"
#include "MLTest.h"
#include <igl/unproject_onto_mesh.h>

/* [MAIN FUNCTION THAT CALLS EVERYTHING ELSE] */
int main(int argc, char *argv[])
{
	string							meshFile; 
	MLEigenSolver*					eigenSolver = new MLSolverFunctions();


	/* GUI FUNCTIONALITY */	
	igl::opengl::glfw::Viewer			viewer;			// Set up main viewer
	igl::opengl::glfw::imgui::ImGuiMenu menu;			// Attach a menu plugin
	viewer.plugins.push_back(&menu);	
	//IMGUI_CHECKVERSION();								// Setup Dear ImGui binding
	ImGui::CreateContext();
	ImGui::StyleColorsDark();							// Set style

	//meshFile = "../Models/AIM894_Chinese Dragon/894_Chinese Dragon.obj";
	//meshFile = "D:/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Cube/Cube_sharp_50k_2.obj";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Bar/Bar_rectangle_5k.obj";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/AIM_Pulley_full/pulley_40k.off";
	//meshFile = "D:/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/TurboSquid_Sphere_HighGenus/TurboSquid_Sphere_highgenus.obj";
	meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/AIM_Chinese_dragon_quad_PGP_small_wavelength/894_dragon_tris.obj";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Dragon/dragon.obj";
	//meshFile = "D:/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Sphere/sphere150k.off";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/AIM_Kitten-watertight/366_kitten_final.obj";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Sphere/sphere150k.off";
	//meshFile = "D:/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/blade_smooth/blade_smooth.obj";
	//meshFile = "D:/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/AIM_Fertility/Fertility.obj";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/AIM_Rocker-arm/38_rocker-arm_270k.off";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Large_Ones/771_oil_pump_1Mfaces_holes_filled2.off";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/AIM_Ramesses_clean_watertight/814_Ramesses_1.5Mtriangles_clean.off";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Large_Ones/m3_680k_vert.off";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/Keenan_Nevertiti/Nefertiti.obj";
	//meshFile = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/EigenTrial/models/AIM_Neptune_clean__watertight_4M triangles/803_neptune_4Mtriangles_manifold.off";
		
	eigenSolver->readMesh(meshFile);


	//testInit(viewer, eigenSolver, meshFile);
	//runMLSolverTest(viewer, eigenSolver);
			
	menu.callback_draw_viewer_window = [&]()
	{
		reducedSpaceMenu(viewer, menu, eigenSolver);
		SIMMenu(viewer, menu, eigenSolver);
		//ImGui::ShowDemoWindow();

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
			viewer.data().set_mesh(eigenSolver->getV(), eigenSolver->getF());
			viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
			break;
		case 'n':
		case 'N':
			//selectFace = true; 
			xMouse = viewer.current_mouse_x;
			yMouse = viewer.core.viewport(3) - viewer.current_mouse_y;

			//if (selectFace) {
			if (igl::unproject_onto_mesh(Eigen::Vector2f(xMouse, yMouse), viewer.core.view,
				viewer.core.proj, viewer.core.viewport, eigenSolver->getV(), eigenSolver->getF(), selectedFace, bc))
			{
				// paint hit red
				printf("Selected face is %d [%d, %d, %d]", selectedFace, eigenSolver->getF()(selectedFace,0), eigenSolver->getF()(selectedFace,1), eigenSolver->getF()(selectedFace,2));
				Eigen::MatrixXd C = Eigen::MatrixXd::Constant(eigenSolver->getF().rows(), 3, 1);
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
	viewer.data().set_mesh(eigenSolver->getV(), eigenSolver->getF());
	viewer.callback_key_down		= key_down;

	//Eigen::Vector4f				bgCol(0.5f, 0.4f, 0.9f, 0.2f);
	//Eigen::Vector4f				bgCol(0.95f, 0.95f, 0.95f, 1.0f);
	Eigen::Vector4f					bgCol(1.f, 1.f, 1.f, 1.f);
	viewer.core.background_color	= bgCol;	
	viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
	//viewer.data().set_colors(Eigen::RowVector3d(0.96666666, 0.96666666, 0.96666666));
	viewer.data().point_size		= 10.0f;
	viewer.data().line_width		= 0.5f; 	
	return viewer.launch();
}
