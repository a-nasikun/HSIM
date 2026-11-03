#include "HSIMMenu.h"

#include "Visualization.h"

#include <igl/file_dialog_open.h>

#include <imgui.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <random>
#include <sstream>

namespace {

const Eigen::RowVector3d kMeshColor(0.93333333, 0.93333333, 0.9333333);

hsim::Options guiDefaultOptions()
{
	hsim::Options options;
	options.tolerance = 1e-2;
	options.maxIterations = 5;
	options.shiftParameter = 0.25;
	options.verbosity = hsim::Verbosity::Summary;
	return options;
}

/// Runs an action and reports exceptions in the menu instead of terminating the program.
template <typename Action>
void guarded(GuiState& state, const char* what, Action&& action)
{
	try {
		state.message.clear();
		action();
	}
	catch (const std::exception& e) {
		state.message = std::string(what) + " failed: " + e.what();
		std::cerr << state.message << std::endl;
	}
}

/// Hands the mesh currently shown by the viewer to the solver.
void setSolverMesh(igl::opengl::glfw::Viewer& viewer, GuiState& state)
{
	state.solver.setOptions(state.options);
	state.solver.setMesh(viewer.data().V, viewer.data().F);
	state.hasResult = false;
}

void showCoarseEigenvector(igl::opengl::glfw::Viewer& viewer, GuiState& state)
{
	const hsim::Hierarchy& h = state.solver.hierarchy();
	if (!h.hasCoarseSolution()) throw std::logic_error("solve the coarse eigenproblem first");
	state.eigenIndex = std::clamp(state.eigenIndex, 0, static_cast<int>(h.coarse.vectors.cols()) - 1);
	showScalarField(viewer, h.basis * h.coarse.vectors.col(state.eigenIndex), state.eigenIndex > 0);
}

void showEigenvector(igl::opengl::glfw::Viewer& viewer, GuiState& state)
{
	if (!state.hasResult) throw std::logic_error("solve the eigenproblem first");
	state.eigenIndex = std::clamp(state.eigenIndex, 0, static_cast<int>(state.result.eigenvectors.cols()) - 1);
	showScalarField(viewer, state.result.eigenvectors.col(state.eigenIndex), state.eigenIndex > 0);
}

void showBasis(igl::opengl::glfw::Viewer& viewer, GuiState& state)
{
	const hsim::Hierarchy& h = state.solver.hierarchy();
	if (!h.hasProlongation()) throw std::logic_error("construct the prolongation first");
	state.basisLevel = std::clamp(state.basisLevel, 0, h.numLevels() - 1);
	const int numBasis = static_cast<int>(h.prolongation[static_cast<size_t>(state.basisLevel)].cols());
	state.basisIndex = std::clamp(state.basisIndex, 0, numBasis - 1);
	if (state.samplesShown) {
		clearPoints(viewer);
		state.samplesShown = false;
	}
	showScalarField(viewer, hsim::basisFunction(h, state.basisLevel, state.basisIndex));
}

void printEigenvalues(const char* title, const Eigen::VectorXd& values, int count)
{
	std::cout << title << ":\n" << values.head(std::min<Eigen::Index>(count, values.size())).transpose() << std::endl;
}

} // namespace

GuiState::GuiState() : options(guiDefaultOptions()), solver(options) {}

void showMesh(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	viewer.data().clear();
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(kMeshColor);
	viewer.data().show_lines = false;
	viewer.data().compute_normals();
	viewer.core().align_camera_center(V, F);
}

void drawHSIMMenu(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu& menu, GuiState& state)
{
	ImGui::SetNextWindowPos(ImVec2(5.f * menu.menu_scaling(), 5.f));
	ImGui::SetNextWindowSize(ImVec2(260.f, 640.f));
	ImGui::Begin("HSIM", nullptr, ImGuiWindowFlags_NoSavedSettings);

	const float w = ImGui::GetContentRegionAvail().x;
	const float p = ImGui::GetStyle().FramePadding.x;

	/* Mesh input / output */
	if (ImGui::Button("Load Mesh", ImVec2((w - p) / 2.f, 0))) {
		guarded(state, "Loading the mesh", [&] {
			const std::string path = igl::file_dialog_open();
			if (path.empty()) return;
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			hsim::readMesh(path, V, F);
			hsim::normalizeMesh(V, 2.0);
			showMesh(viewer, V, F);
			state.solver = hsim::Solver(state.options);
			state.hasResult = false;
			state.samplesShown = false;
			std::cout << "Loaded " << path << ": " << V.rows() << " vertices, " << F.rows() << " faces" << std::endl;
		});
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save Mesh", ImVec2((w - p) / 2.f, 0))) viewer.open_dialog_save_mesh();

	/* Parameters */
	ImGui::InputInt("Num eigs", &state.numModes);
	ImGui::InputInt("Max iter", &state.options.maxIterations);
	ImGui::InputDouble("Tolerance", &state.options.tolerance, 0.0, 0.0, "%.2e");
	ImGui::InputDouble("Shift param", &state.options.shiftParameter, 0.0, 0.0, "%.3f");
	state.numModes = std::max(state.numModes, 1);

	/* One-click usage */
	if (ImGui::Button("Construct Hierarchy", ImVec2(w, 0))) {
		guarded(state, "Constructing the hierarchy", [&] {
			setSolverMesh(viewer, state);
			state.solver.buildHierarchy(state.numModes);
			state.eigenIndex = std::max(state.eigenIndex, 1);
			showCoarseEigenvector(viewer, state);
		});
	}
	if (ImGui::Button("Solve Eigenproblem", ImVec2(w, 0))) {
		guarded(state, "Solving the eigenproblem", [&] {
			if (!state.solver.hasMesh()) setSolverMesh(viewer, state);
			state.solver.setOptions(state.options);
			state.result = state.solver.solve(state.numModes);
			state.hasResult = true;
			printEigenvalues("Eigenvalues", state.result.eigenvalues, state.numModes);
			state.eigenIndex = std::max(state.eigenIndex, 1);
			showEigenvector(viewer, state);
		});
	}

	/* Step-by-step construction */
	if (ImGui::CollapsingHeader("Step by step")) {
		if (ImGui::Button("[1] Assemble operators", ImVec2(w, 0))) guarded(state, "Assembling the operators", [&] { setSolverMesh(viewer, state); });

		ImGui::InputDouble("Level ratio", &state.options.levelRatio, 0.0, 0.0, "%.2f");
		if (ImGui::Button("[2] Sampling", ImVec2(w, 0))) {
			guarded(state, "Sampling", [&] {
				if (!state.solver.hasMesh()) setSolverMesh(viewer, state);
				state.solver.setOptions(state.options);
				state.solver.computeSamples(state.numModes);
			});
		}
		const hsim::Hierarchy& h = state.solver.hierarchy();
		for (int k = 0; k < h.numLevels() && h.hasSamples(); ++k) {
			const std::string label = "Dist " + std::to_string(k);
			if (ImGui::Button(label.c_str(), ImVec2(w / static_cast<float>(h.numLevels()) - p, 0)))
				showScalarField(viewer, h.samples.distance[static_cast<size_t>(k)]);
			if (k + 1 < h.numLevels()) ImGui::SameLine(0, p);
		}

		ImGui::InputDouble("Support size", &state.options.supportSize, 0.0, 0.0, "%.2f");
		if (ImGui::Button("[3] Prolongation", ImVec2(w, 0))) {
			guarded(state, "Constructing the prolongation", [&] {
				state.solver.setOptions(state.options);
				state.solver.computeProlongation();
			});
		}
		for (int k = 0; k < h.numLevels() && h.hasProlongation(); ++k) {
			const std::string label = "Lvl " + std::to_string(k);
			if (ImGui::Button(label.c_str(), ImVec2(w / static_cast<float>(h.numLevels()) - p, 0))) {
				state.basisLevel = k;
				guarded(state, "Showing the basis", [&] { showBasis(viewer, state); });
			}
			if (k + 1 < h.numLevels()) ImGui::SameLine(0, p);
		}

		if (ImGui::Button("[4] Coarse eigenproblem", ImVec2(w, 0))) {
			guarded(state, "Solving the coarse eigenproblem", [&] {
				state.solver.solveCoarseProblem();
				printEigenvalues("Coarse eigenvalues", state.solver.hierarchy().coarse.values, 20);
				showCoarseEigenvector(viewer, state);
			});
		}
	}

	ImGui::Separator();

	/* Visualisation */
	if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Show/Hide Samples", ImVec2(w, 0))) {
			guarded(state, "Showing the samples", [&] {
				const hsim::Hierarchy& h = state.solver.hierarchy();
				if (!h.hasSamples()) throw std::logic_error("compute the samples first");
				state.samplesShown = !state.samplesShown;
				if (state.samplesShown) showSamples(viewer, state.solver.vertices(), h.samples.vertices, h.levelSizes);
				else clearPoints(viewer);
				viewer.data().point_size = 10.0f;
			});
		}

		const float bw = (w - 110.f - 4.f * p) / 3.f;
		ImGui::Text("Basis (lvl %d)", state.basisLevel);
		ImGui::SameLine(110.f);
		if (ImGui::Button("-##basis", ImVec2(bw, 0))) { --state.basisIndex; guarded(state, "Showing the basis", [&] { showBasis(viewer, state); }); }
		ImGui::SameLine(0, p);
		if (ImGui::Button("+##basis", ImVec2(bw, 0))) { ++state.basisIndex; guarded(state, "Showing the basis", [&] { showBasis(viewer, state); }); }
		ImGui::SameLine(0, p);
		if (ImGui::Button("Rand##basis", ImVec2(bw, 0))) {
			static std::mt19937 rng(0);
			state.basisIndex = static_cast<int>(rng() % 100000);
			guarded(state, "Showing the basis", [&] { showBasis(viewer, state); });
		}

		ImGui::Text("Coarse [%d]", state.eigenIndex);
		ImGui::SameLine(110.f);
		if (ImGui::Button("-##coarse", ImVec2(bw, 0))) { state.eigenIndex = std::max(0, state.eigenIndex - 1); guarded(state, "Showing the eigenvector", [&] { showCoarseEigenvector(viewer, state); }); }
		ImGui::SameLine(0, p);
		if (ImGui::Button("+##coarse", ImVec2(bw, 0))) { ++state.eigenIndex; guarded(state, "Showing the eigenvector", [&] { showCoarseEigenvector(viewer, state); }); }
		ImGui::SameLine(0, p);
		if (ImGui::Button("Show##coarse", ImVec2(bw, 0))) guarded(state, "Showing the eigenvector", [&] { showCoarseEigenvector(viewer, state); });

		ImGui::Text("HSIM [%d]", state.eigenIndex);
		ImGui::SameLine(110.f);
		if (ImGui::Button("-##hsim", ImVec2(bw, 0))) { state.eigenIndex = std::max(0, state.eigenIndex - 1); guarded(state, "Showing the eigenvector", [&] { showEigenvector(viewer, state); }); }
		ImGui::SameLine(0, p);
		if (ImGui::Button("+##hsim", ImVec2(bw, 0))) { ++state.eigenIndex; guarded(state, "Showing the eigenvector", [&] { showEigenvector(viewer, state); }); }
		ImGui::SameLine(0, p);
		if (ImGui::Button("Show##hsim", ImVec2(bw, 0))) guarded(state, "Showing the eigenvector", [&] { showEigenvector(viewer, state); });

		if (state.hasResult) {
			const hsim::Diagnostics& d = state.result.diagnostics;
			ImGui::Text("lambda[%d] = %.6g", state.eigenIndex, state.result.eigenvalues(std::min<Eigen::Index>(state.eigenIndex, state.result.eigenvalues.size() - 1)));
			ImGui::Text("hierarchy %.3f s, iteration %.3f s", d.hierarchySeconds, d.iterationSeconds);
		}
	}

	if (!state.message.empty()) {
		ImGui::Separator();
		ImGui::TextWrapped("%s", state.message.c_str());
	}
	ImGui::End();
}
