// Interactive example of the HSIM library.
// Usage: HSIM_GUI [mesh file]   (default: the kitten model shipped in Models/)
#include "HSIMMenu.h"

#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
	const std::string meshFile = (argc > 1) ? argv[1] : HSIM_MODELS_DIR "/AIM366_Kitten/366_Kitten_Flip.obj";

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	try {
		hsim::readMesh(meshFile, V, F);
		std::cout << "Loaded " << meshFile << ": " << V.rows() << " vertices, " << F.rows() << " faces" << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin;
	viewer.plugins.push_back(&plugin);
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	plugin.widgets.push_back(&menu);

	GuiState state;
	menu.callback_draw_viewer_window = [&]() { drawHSIMMenu(viewer, menu, state); };

	// Key '0' resets the colours.
	viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& v, unsigned char key, int) {
		if (key != '0') return false;
		showMesh(v, v.data().V, v.data().F);
		return true;
	};

	showMesh(viewer, V, F);
	viewer.core().background_color = Eigen::Vector4f(1.f, 1.f, 1.f, 1.f);
	viewer.data().point_size = 10.0f;
	viewer.data().line_width = 0.5f;
	return viewer.launch();
}
