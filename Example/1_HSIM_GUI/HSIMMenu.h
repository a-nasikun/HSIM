// ImGui menu of the GUI example: step-by-step construction of the hierarchy and visualisation.
#pragma once

#include <hsim/hsim.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>

#include <string>

/// Everything the menu needs to remember between frames.
struct GuiState {
	hsim::Options options;
	hsim::Solver solver;
	hsim::Result result;
	bool hasResult = false;

	int numModes = 125;
	int basisLevel = 0;
	int basisIndex = 0;
	int eigenIndex = 1;
	bool samplesShown = false;
	std::string message; ///< last error or status message, shown in the menu

	GuiState();
};

/// Draws the menu window; called once per frame.
void drawHSIMMenu(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu& menu, GuiState& state);

/// Shows a mesh in the viewer with the default appearance.
void showMesh(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
