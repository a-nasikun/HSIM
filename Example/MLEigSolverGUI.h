#ifndef FAST_SPECTRUM_GUI
#define FAST_SPECTRUM_GUI

#include <iostream>

#include <imgui/imgui.h>
#include <imgui/imgui.cpp>
#include <imgui/imgui_demo.cpp>
#include <imgui/imgui_draw.cpp>
#include <imgui_impl_glfw_gl3.h>

#include <igl/igl_inline.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/ViewerPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/isolines.h>
#include <igl/jet.h>

//#define EIGEN_USE_MKL_ALL 
#include <Eigen/Eigen>
//#include <Eigen/PardisoSupport>
//#include <Eigen/CholmodSupport>
//#include <Eigen/SPQRSupport>
//#include <Eigen/UmfPackSupport>
//#include <Eigen/KLUSupport>

//#include "MLEigSolver.h"

#include "MLSolverFunctions.h"

//#include <mkl.h>
//#include <armadillo>

using namespace std;

enum BasisType { Coarse_1Level_Basis = 0, MultiGrid_Geom_Basis = 1, MultiGrid_NonSmooth_Basis =2 };
static int			numOfModes		= 50;
int					basisToShow		= 0;
int					eigToShow		= 0;
static int			eigToCompute    = 500;
bool				boolShowSamples = false;
bool				useNonSmoothBasis = false; 
//static int			numOfSample		= 250;
static int			sampleType		= Sample_Farthest_Point; 
static int		 	basisType		= MultiGrid_Geom_Basis;
static int			dataToShow		= 0;
float				tauScale		= 10.0f;
string				model			= "Sphere40k";
static int			SIMtype			= 2;
static int			problemType		= 0;					// type of problem (0=Laplace--Beltrami)
static int			matType			= 0;					// material type (for hessian/linear elasticity)
static int			defAxis = 0;							// axis in selecting the partial eigenproblem
static double		clampRegion;							// percentage of the clampling
static double		eigScale		= 1.005;

void reducedSpaceMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, MLSolverFunctions*& eigenSolver)
{
	// Variables For Viewer visualization
	float	menuWindowLeft		= 0.f,
			menuWindowWidth		= 250.f,
			menuWindowHeight	= 600.f;


	Eigen::MatrixXd		VColor;
	Eigen::VectorXd		Z;

	//ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5), ImGuiSetCond_FirstUseEver);
	ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5));
	//ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight), ImGuiSetCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight));
	ImGui::Begin("Reduced Space", nullptr, ImGuiWindowFlags_NoSavedSettings);

	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

	static bool isShowAxis = false;

	// Main Algorithm
	if (ImGui::CollapsingHeader("Setting Reduce Space", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Load##Mesh1", ImVec2((w) / 2.f, 0))) {
			// Load mesh + display the correct mesh
			Eigen::MatrixXd tempV;// = viewer.data_list[0].V;
			Eigen::MatrixXi tempF;// = viewer.data_list[0].F;

			// Load mesh + display the correct mesh
			int numOfModel = viewer.data_list.size();
			printf("mesh that we have  now: %d \n", viewer.data_list.size());

			if (numOfModel > 0) {
				tempV = viewer.data_list[0].V;
				tempF = viewer.data_list[0].F;
				viewer.data_list.clear();
			}

			viewer.open_dialog_load_mesh();

			if (viewer.data_list.size() < 1) {
				cout << "data less than 1 \n";
				viewer.append_mesh();
				viewer.data().clear();
				if (tempV.rows() > 0) {
					viewer.data().set_mesh(tempV, tempF);
					viewer.data().compute_normals();
				}
			}

			if (viewer.data().V.rows() > 0) {

				scaleMesh(viewer.data().V, viewer.data().F, 2.0);
				viewer.core().align_camera_center(viewer.data().V, viewer.data().F);

				viewer.data().show_lines = false;
				printf("A new mesh with %d vertices (%d faces).\n", viewer.data().V.rows(), viewer.data().F.rows());
				//viewer.data().set_mesh(V, F);
				viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));

				boolShowSamples = false;
				printf("mesh that we have  now: %d \n", viewer.data_list.size());
			}
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Save##Mesh1", ImVec2((w - p) / 2.f, 0))) {
			viewer.open_dialog_save_mesh();
		}
		static int axis = 0;
	}
	if (ImGui::Button("Construct Hierarchy", ImVec2((w), 0)))
	{
		eigenSolver->setup(viewer.data().V, viewer.data().F, numOfModes);
		eigenSolver->buildHierarchy(viewer.data().V, viewer.data().F, numOfModes);
	}

		/* STEP BY STEP CONSTRUCTION */
		if (ImGui::CollapsingHeader("Program Step by Step##1", ImGuiTreeNodeFlags_DefaultOpen)) {
			
			int		numSlices;
			static	Eigen::VectorXd colorVect;
			static bool	isClampedBasis = false;
			static int numLevels;			

			/* Setup stiffness Matrix */
			if (ImGui::Button("[1] Construct Stiffness Matrix##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1) return;

				eigenSolver->constructMassMatrix();
				eigenSolver->initiateAdM();
				eigenSolver->constructStiffnessMatrix();
			}

			/* For Hierarchy and Sampling */
			static double levelRatio = 10.0;
			static char lvlRatTxt[10] = "10.0"; ImGui::InputText("Levels ratio", lvlRatTxt, 10);
			levelRatio = atof(lvlRatTxt);

			if (ImGui::Button("[2] Sampling##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1) return;
				levelRatio = atof(lvlRatTxt);

				cout << "> Creating samples... " << endl;
				chrono::high_resolution_clock::time_point	t0, t1;
				chrono::duration<double>					duration;
				t0 = chrono::high_resolution_clock::now();

				Eigen::VectorXd sampSize;
				//setSamplingLevels(eigenSolver, levelRatio, numOfModes, sampSize);		
				std::cout << "Sought modes: " << numOfModes << std::endl;

				/* For 1 level (coarsest level) sampling only*/
				eigenSolver->setSamplingLevels(sampSize, numOfModes, levelRatio);
				eigenSolver->setInitialSystem(sampSize.rows());
				eigenSolver->constructSampleHierarchy(sampSize);

				t1 = chrono::high_resolution_clock::now();
				duration = t1 - t0;
				cout << "in " << duration.count() << " seconds." << endl;
			}


			ImGui::Text("Show sample dist");
			for (int i = 0; i < numLevels; ++i)
			{
				string disp = "Dist ";
				disp += to_string(i);
				char const* ca = disp.c_str();
				if (ImGui::Button(ca, ImVec2(w / (double)numLevels - p, 0))) {
					eigenSolver->visualizeGeodesicDistanceMultLevel(viewer, i);
				} ImGui::SameLine(0, p);
			}
			ImGui::Spacing(); ImGui::Spacing();

			/* For Basis */
			ImGui::Checkbox("Clamp Basis##1", &isClampedBasis);
			static double numOfSupport;
			static char numSupTxt[15] = "7.5"; ImGui::InputText("Num of Supports", numSupTxt, 6);
						
			if (isClampedBasis) {
				ImGui::RadioButton("X", &defAxis, 0); ImGui::SameLine();
				ImGui::RadioButton("Y", &defAxis, 1); ImGui::SameLine();
				ImGui::RadioButton("Z", &defAxis, 2); ImGui::SameLine();
				static char clampRegionTxt[10] = "0.01"; ImGui::InputText("Selection", clampRegionTxt, 10);
				clampRegion = atof(clampRegionTxt);
			}

			if (ImGui::Button("[3] Construct Basis##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getStiffnessMatrix().rows() < 1) {
					cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
					return;
				}

				cout << "Constructing basis..." << endl;
				chrono::high_resolution_clock::time_point	t0, t1;
				chrono::duration<double>					duration;
				t0 = chrono::high_resolution_clock::now();

				eigenSolver->setClampBasis(isClampedBasis);
				numOfSupport = atof(numSupTxt);

				eigenSolver->constructBasis(numOfSupport, defAxis, clampRegion);
				
				useNonSmoothBasis = false;

				t1 = chrono::high_resolution_clock::now();
				duration = t1 - t0;
				cout << "in " << duration.count() << " seconds." << endl;
			}

			// Showing basis in different levels
			for (int i = 0; i < eigenSolver->getLevel(); ++i)
			{
				string disp = "Lvl. "; disp += to_string(i);
				char const* ca = disp.c_str();
				if (ImGui::Button(ca, ImVec2(w / (double)eigenSolver->getLevel() - p, 0))) {					
					eigenSolver->visualizeBasis(viewer, basisToShow, i);
				}
				if (i<(eigenSolver->getLevel()-1)) ImGui::SameLine(0, p);
			}
				
			ImGui::Spacing(); ImGui::Spacing();
			

			/* Solving Low-Dim Eigenproblem */
			if (ImGui::Button("[4] Solve Reduced Eigenproblem##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1) {
					cout << "Error! Make sure you execute previous steps before solving the Low-Dim Eigenproblem." << endl;
					return;
				}

				cout << "Computing the restricted eigen problem" << endl;
				chrono::high_resolution_clock::time_point	t0, t1;
				chrono::duration<double>					duration;
				t0 = chrono::high_resolution_clock::now();

				eigenSolver->solveRestrictedProblem();

				t1 = chrono::high_resolution_clock::now();
				duration = t1 - t0;
				cout << "in " << duration.count() << " seconds." << endl;

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_REDUCED, eigScale);
				cout << eigenSolver->getEigenValuesReduced().head(std::min(20, (int)eigenSolver->getEigenValuesReduced().rows())).transpose();				

				cout << eigenSolver->getEigenValuesReduced().head(std::min(20, (int)eigenSolver->getEigenValuesReduced().rows()));					
			}

			static char numRedEigTxt[32] = "20";	
			if (ImGui::Button("Print reduced eigenvals##1", ImVec2(w/2.0-p, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesReduced().rows()<1) {
					cout << "Error! Make sure you execute previous steps before solving the Low-Dim Eigenproblem." << endl;
					return;
				}
				static int numRedEigs = atoi(numRedEigTxt);
				cout << "Red eigenvalues" << eigenSolver->getEigenValuesReduced().head(std::min(numRedEigs, (int)eigenSolver->getEigenValuesReduced().rows())) << endl;
			}
			ImGui::SameLine();
			ImGui::InputText("#eigs", numRedEigTxt, 32);
		}


		ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();

		/* VISUALIZATION WINDOW */
		if (ImGui::CollapsingHeader("Visualization##MLEigSolver1", ImGuiTreeNodeFlags_DefaultOpen)) {
			/* For Sampling */
			if (ImGui::Button("Show Samples##1", ImVec2((w), 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1) {
					cout << "Error! Make sure you compute new samples before viewing them." << endl;
					return;
				}

				boolShowSamples = !boolShowSamples;
				if (boolShowSamples) {
					visualizeSampleMulti(viewer, eigenSolver->getV(), eigenSolver->getF(), eigenSolver->getSample(), eigenSolver->getSubspaceSize());
				}
				else {
					viewer.data().points.resize(0, 6);
				}
				viewer.data().point_size = 10.0f;
			}

			/* For Basis */
			if (problemType == 1) {
				static char scale[25] = "1.05"; ImGui::InputText("Spatial Scaling", scale, 10);
				eigScale = atof(scale);
			}
			ImGui::Text("Show Basis ");
			ImGui::SameLine(0, p);
			if (ImGui::Button("-1##Basis1", ImVec2((w - 85.0f - 5.0*p) / 4.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}

				if (basisToShow > 0) basisToShow--;

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
				}

				eigenSolver->visualizeBasis(viewer, basisToShow, eigScale);
			}
			ImGui::SameLine(0, p);
			ImGui::Text("[%d]", basisToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##Basis1", ImVec2((w - 85.0f - 5.0*p) / 4.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}

				if (basisToShow < (eigenSolver->getBasisMatrix().cols() - 1))
					basisToShow++;

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
				}

				eigenSolver->visualizeBasis(viewer, basisToShow, eigScale);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Rand1", ImVec2((w - 85.0f - 5.0*p) * 2 / 4.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}

				srand(time(NULL));
				basisToShow = rand() % eigenSolver->getBasisMatrix().cols();

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
				}

				eigenSolver->visualizeBasis(viewer, basisToShow, eigScale);
			}
			/* For Approx. Eigenvectors */
			ImGui::Text("Reduced");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##EigVec1", ImVec2((w-120.0-3* p) / 2.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesReduced().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow > 0)
					eigToShow--;

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_REDUCED, eigScale);
			}
			ImGui::SameLine(0, p);
			ImGui::Text("[%d]", eigToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##EigVec1", ImVec2((w-120.0-3 * p) / 2.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesReduced().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow < (eigenSolver->getEigenValuesReduced().rows() - 1))
					eigToShow++;

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_REDUCED, eigScale);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Red##EigVec1", ImVec2((w-85.0-3 * p) / 3.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesReduced().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}
				if (eigToShow >= eigenSolver->getEigenValuesReduced().rows()) {
					cout << "Error! Index out of bound\n";
					return;
				}

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_REDUCED, eigScale);
			}

			/* For Reference Eigenvectors */
			ImGui::Text("Reference");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##RefEigVec1", ImVec2((w-120.0-3* p) / 2.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getEigenValuesRef().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow > 0)
					eigToShow--;

				std::cout << "["<< eigToShow << "]"<< "Eigenvalue=" << eigenSolver->getEigenValueRef(eigToShow) << std::endl;
				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_REFERENCE, eigScale);
			}
			ImGui::SameLine(0, p);
			ImGui::Text("[%d]", eigToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##RefEigVec1", ImVec2((w-120.0-3* p) / 2.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getEigenValuesRef().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}


				if (eigToShow < (eigenSolver->getEigenValuesRef().rows() - 1))
					eigToShow++;

				std::cout << "[" << eigToShow << "]" << "Eigenvalue=" << eigenSolver->getEigenValueRef(eigToShow) << std::endl;
				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_REFERENCE, eigScale);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Ref##EigVec1", ImVec2((w-85.0-3*p) / 3.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getEigenValuesRef().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}
				if (eigToShow >= eigenSolver->getEigenValuesRef().rows()) {
					cout << "Error! Index out of bound\n";
					return;
				}

				std::cout << "[" << eigToShow << "]" << "Eigenvalue=" << eigenSolver->getEigenValueRef(eigToShow) << std::endl;
				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_REFERENCE, eigScale);
			}

			/* For SIM Eigenvectors */
			ImGui::Text("SIM Eigs.");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##SIMEigVec1", ImVec2((w-120.0-3* p) / 2.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesSIM().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow > 0)
					eigToShow--;

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_SIM, eigScale);
			}
			ImGui::SameLine(0, p);
			ImGui::Text("[%d]", eigToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##SIMEigVec1", ImVec2((w-120.0-3* p) / 2.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesSIM().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow < (eigenSolver->getEigenValuesSIM().rows() - 1))
					eigToShow++;

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_SIM, eigScale);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("SIM##EigVec1", ImVec2((w-85.0-3* p) / 3.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesSIM().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}
				if (eigToShow >= eigenSolver->getEigenValuesSIM().rows()) {
					cout << "Error! Index out of bound\n";
					return;
				}

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLSolverFunctions::EigenVectorType::EIGEN_SIM, eigScale);
			}

			/* Overlays */
			//ImGui::Checkbox("Wireframe", &(viewer.data().show_lines));				
			//ImGui::SameLine();
			//ImGui::Checkbox("Fill Faces", &(viewer.data().show_faces));
		}		

		/* Axis*/
		if (ImGui::Button("Show axis", ImVec2((w), 0))) {
			isShowAxis = !isShowAxis;
			if (isShowAxis) {
				double maxAxis = eigenSolver->getV().maxCoeff();
				std::cout << "Max Coef: " << maxAxis << std::endl;
				//viewer.data().line_width = 0.5f;
				viewer.data().add_edges(Eigen::RowVector3d(0, 0, 0), Eigen::RowVector3d(maxAxis, 0, 0), Eigen::RowVector3d(1.0, 0, 0));
				viewer.data().add_edges(Eigen::RowVector3d(0, 0, 0), Eigen::RowVector3d(0, maxAxis, 0), Eigen::RowVector3d(0, 1.0, 0));
				viewer.data().add_edges(Eigen::RowVector3d(0, 0, 0), Eigen::RowVector3d(0, 0, maxAxis), Eigen::RowVector3d(0, 0, 1.0));
				//viewer.data().line_width = 4.5f;

				std::stringstream l1;
				l1 << maxAxis;
				viewer.data().add_label(Eigen::RowVector3d(maxAxis, 0, 0), l1.str());
				//viewer.data().line_width = 0.5f;
			}
			else {
				Eigen::MatrixXd Vtemp = viewer.data().V;
				Eigen::MatrixXi Ftemp = viewer.data().F;
				viewer.data().clear();
				viewer.data().set_mesh(Vtemp, Ftemp);
			}
		}

		if (ImGui::Button("Reset Mesh", ImVec2(w, 0))) {
			viewer.data().set_vertices(eigenSolver->getV());
		}

		if (ImGui::Button("Lighter", ImVec2((w-p) / 2.f, 0))) {
			viewer.data().line_width *= 0.5f;
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Bolder", ImVec2((w-p) / 2.f, 0))) {
			viewer.data().line_width *= 2.0f;
		}

		if (ImGui::Button("Smaller", ImVec2((w - p) / 2.f, 0))) {
			viewer.data().point_size *= 1.0/1.5f;
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Larger", ImVec2((w - p) / 2.f, 0))) {
			viewer.data().point_size *= 1.5f;
		}

		ImGui::Separator(); 

		/* Diffusion distance */
		static std::vector<Eigen::VectorXd> diffTensor1;
		static std::vector<Eigen::VectorXd> diffTensor2;
		static Eigen::VectorXd diffVector(eigenSolver->getV().rows());
		static int sourceID = 0;

		if (ImGui::CollapsingHeader("Diffusion Distance##1", ImGuiTreeNodeFlags_CollapsingHeader))
		{			
			static char weighttxt[32] = "0.5";						ImGui::InputText("Weight", weighttxt, 32);
			if (ImGui::Button("Create Diffusion Tensor", ImVec2(w, 0))) {			

				if (auto solver = dynamic_cast<MLSolverFunctions*>(eigenSolver)) {
					double weight = atof(weighttxt);
					//solver->constructDiffusionTensor(min(numOfModes, 50), 1.0, solver->getEigenVectorsSIM(), solver->getEigenValuesSIM(), diffTensor1);
					solver->constructDiffusionTensor(std::min(numOfModes, 50), weight, solver->getEigenVectorsSIM(), solver->getEigenValuesSIM(), diffTensor2);					
					std::cout << "Diffusion tensor successfully created \n";
				}
			}

			static char sourceIDtxt[32] = "0";						ImGui::InputText("0", sourceIDtxt, 32);
			Eigen::MatrixXd VCol;
			if (ImGui::Button("Compute Diffusion Distance", ImVec2(w, 0))) {
				
				sourceID = atoi(sourceIDtxt);				
				if(sourceID>=eigenSolver->getV().rows()) 
					sourceID = eigenSolver->getV().rows()-1;
				std::cout << "Source point: " << sourceID << std::endl;
				diffVector.resize(eigenSolver->getV().rows());

				for (int i = 0; i < eigenSolver->getV().rows(); ++i) {
					Eigen::VectorXd diff = diffTensor2[sourceID] - diffTensor2[i];
					diffVector(i) = diff.norm();
				}

				
				igl::jet(diffVector, true, VCol);
				viewer.data().set_colors(VCol);
				viewer.data().add_points(eigenSolver->getV().row(sourceID), Eigen::RowVector3d(0.7, 0.1, 0.1));
			}

			if (ImGui::Button("Show Isolines", ImVec2(w, 0))) {
				Eigen::MatrixXd isoV, isoE;									
				//igl::isolines(eigenSolver->getV(), eigenSolver->getF(), diffVector, 10, isoV, isoE);
				
				//for (int i = 0; i < isoE.rows(); i++) {
				//	viewer.data().add_edges(isoV.row(isoE.row(i)(0)), isoV.row(isoE.row(i)(1)), Eigen::RowVector3d(0.6, 0.1, 0.1));
				//}

			}
		}

		ImGui::End();
}

void SIMMenu(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu& menu, MLSolverFunctions*& eigenSolver) {
	// Variables For Viewer visualization
	float	menuWindowLeft = 475.f,
		menuWindowWidth = 200.f,
		menuWindowHeight = 675.f;

	ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5));
	ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight));
	ImGui::Begin("Subspace Iteration Method##Menu", nullptr, ImGuiWindowFlags_NoSavedSettings);

	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

	static int SIMType = 0;			// 0: M-L, 1: SGP, 2:random, 3: Bathe
	static int shiftType = 3;		// 0: Bathe, 1: Itoh/Wilson, 2: Gong, 3: Zhao
	static bool isClampedEigen = false;

	/* some parameters */
	//static int		numOfModes	= 5;
	static int		maxIter = 5;
	static double	tol = 1e-2;
	/* Type of the SIM to be computed*/
	std::map<int, MLSolverFunctions::SIMType>			typeMap;
	typeMap.insert(std::pair<int, MLSolverFunctions::SIMType>(0, MLSolverFunctions::SIMType::SIM_MultiLevel));
	typeMap.insert(std::pair<int, MLSolverFunctions::SIMType>(1, MLSolverFunctions::SIMType::SIM_1level_WarmStart));
	typeMap.insert(std::pair<int, MLSolverFunctions::SIMType>(2, MLSolverFunctions::SIMType::SIM_1level_Random));
	typeMap.insert(std::pair<int, MLSolverFunctions::SIMType>(3, MLSolverFunctions::SIMType::SIM_1level_Bathe));

	/* Type of shifting approach */
	std::map<int, MLSolverFunctions::ShiftType>			shiftMap;
	shiftMap.insert(std::pair<int, MLSolverFunctions::ShiftType>(0, MLSolverFunctions::ShiftType::SHIFT_BATHE));
	shiftMap.insert(std::pair<int, MLSolverFunctions::ShiftType>(1, MLSolverFunctions::ShiftType::SHIFT_ITOH_WILSON));
	shiftMap.insert(std::pair<int, MLSolverFunctions::ShiftType>(2, MLSolverFunctions::ShiftType::SHIFT_GONG));
	shiftMap.insert(std::pair<int, MLSolverFunctions::ShiftType>(3, MLSolverFunctions::ShiftType::SHIFT_ZHAO));



	static bool isShiftedSIM = true;
	static double zhaoShiftParam = 0.25;
	shiftType = 3;
	if (ImGui::CollapsingHeader("SIM Option##1", ImGuiTreeNodeFlags_CollapsingHeader)) {
		static char numModesTxt[10] = "125";	ImGui::InputText("Num Eigs", numModesTxt, 10);
		static char maxIterTxt[10] = "5";		ImGui::InputText("Max Iter", maxIterTxt, 10);
		static char tolTxt[15] = "0.01";	ImGui::InputText("Tolerance##1", tolTxt, 15);
		static char zhaoTxt[10] = "0.25";	ImGui::InputText("Shift param", zhaoTxt, 10);
		numOfModes = atoi(numModesTxt);
		maxIter = atoi(maxIterTxt);
		tol = atof(tolTxt);
		zhaoShiftParam = atof(zhaoTxt);
	}


	Eigen::VectorXd EigValTest;
	Eigen::MatrixXd EigVectTest;
	if (ImGui::Button("Check SIM")) {
		tie(EigValTest, EigVectTest) = eigenSolver->eigs(viewer.data().V, viewer.data().F, numOfModes, tol, maxIter, zhaoShiftParam);

		std::cout << "Eigenvalues " << EigValTest.transpose() << std::endl << std::endl;
		std::cout << "Eigenvectors " << EigVectTest.topLeftCorner(10, 10) << std::endl << std::endl;
	}

	if (ImGui::Button("Solve Eigenproblem.##1", ImVec2(w, 0))) {
		if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows() < 1) return;
		if ((SIMType == 0 || SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;

		eigenSolver->zhaoShiftParam = zhaoShiftParam;
		std::cout << "Shift param = " << zhaoShiftParam << std::endl;
		eigenSolver->performSubspaceIterationMethod(typeMap[SIMType], numOfModes, maxIter, tol, isShiftedSIM, shiftMap[shiftType], defAxis, clampRegion);

		printf("Successfully computing the eigenvalues \n");
		cout << "Eigenvalues: \n" << eigenSolver->getEigenValuesSIM().transpose() << endl;
		cout << "EigenVectors: \n" << eigenSolver->getEigenVectorsSIM().topLeftCorner(6, 9) << endl;
	}

	ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();


	if (ImGui::Button("Scripted (1) OURS", ImVec2(w, 0))) {
		if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows() < 1) return;
		if ((SIMType == 0 || SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;

		/*  Testing on :
		*	(1) random, (2) Bathe
		*	(3) ML without shift
		*   (4) ML with shift-once
		*/

		eigenSolver->zhaoShiftParam = 0.25;
		int maxIters = 250;
		//vector<int> wantedModes{ 50, 100, 250 };
		vector<int> wantedModes{ 50 };
		//vector<double> wantedTols{1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,1e-10};
		vector<double> wantedTols{ 1e-6, 1e-7, 1e-8, 1e-9,1e-10 };

		for (int j = 0; j < wantedModes.size(); ++j) {
			for (int k = 0; k < wantedTols.size(); ++k) {
				// ML-with shift
				eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
				// ML no shift
				//eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
				// random
				//eigenSolver->performSubspaceIterationMethod(typeMap[2], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
				/// // bathe init vector
				///eigenSolver->performSubspaceIterationMethod(typeMap[3], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
			}
		}
	}

}

#endif // !FAST_SPECTRUM_GUI