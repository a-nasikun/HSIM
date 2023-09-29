#include "MLEigSolver.h"
#include "MLSolverFunctions.h"
#include "MLEigSolverGUI.h"
#include "MLTest.h"
#include <igl/unproject_onto_mesh.h>


/* [MAIN FUNCTION THAT CALLS EVERYTHING ELSE] */
int main(int argc, char *argv[])
{
	string								meshFile; 
	MLSolverFunctions*					eigenSolver = new MLSolverFunctions();

	/* GUI FUNCTIONALITY */	
	igl::opengl::glfw::Viewer			viewer;			// Set up main viewer
	igl::opengl::glfw::imgui::ImGuiMenu menu;			// Attach a menu plugin
	viewer.plugins.push_back(&menu);	
	ImGui::CreateContext();
	ImGui::StyleColorsDark();							// Set style
	
	meshFile = "../HSIM/Models/AIM366_Kitten/366_Kitten_Flip.obj";
		
	eigenSolver->readMesh(meshFile);

			
	/* Pembuatan menu GUI */
	menu.callback_draw_viewer_window = [&]()
	{
		HSIMMenu(viewer, menu, eigenSolver);
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

		switch (key)
		{
		case '0':
			viewer.data().clear();			
			viewer.data().set_mesh(eigenSolver->getV(), eigenSolver->getF());
			viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
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

	Eigen::Vector4f					bgCol(1.f, 1.f, 1.f, 1.f);
	viewer.core().background_color = bgCol;
	viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
	viewer.data().point_size		= 10.0f;
	viewer.data().line_width		= 0.5f; 	
	return viewer.launch();
}
