#ifndef FAST_SPECTRUM_GUI
#define FAST_SPECTRUM_GUI

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

#include <Eigen/PardisoSupport>

#include "MLEigenSolver.h"
#include "MLSolverFunctions.h"

using namespace std;
using namespace MuLES;

enum BasisType { Coarse_1Level_Basis = 0, MultiGrid_Geom_Basis = 1, MultiGrid_NonSmooth_Basis =2 };
static int			numOfModes		= 25;
int					basisToShow		= 0;
int					eigToShow		= 0;
static int			eigToCompute    = 500;
bool				boolShowSamples = false;
bool				useNonSmoothBasis = false; 
//static int			numOfSample		= 250;
static int			sampleType		= MuLES::Sample_Farthest_Point; 
static int		 	basisType		= MultiGrid_Geom_Basis;
static int			dataToShow		= 0;
float				tauScale		= 10.0f;
string				model			= "Sphere40k";
static int			SIMtype			= 2;
static int			problemType		= 0;					// type of problem (0=Laplace--Beltrami, 1=Linear Elasticity)
static int			matType			= 0;					// material type (for hessian/linear elasticity)
static int			defAxis = 0;							// axis in selecting the partial eigenproblem
static double		clampRegion;							// percentage of the clampling
static double		eigScale		= 1.005;

void setSamplingLevels(MLEigenSolver*& eigenSolver, double levelRatio, int numEigen, Eigen::VectorXi& sampSize);
void setSamplingLevels(MLEigenSolver*& eigenSolver, int numEigen, Eigen::VectorXi& sampSize);

void reducedSpaceMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, MLEigenSolver*& eigenSolver) 
{
	// Variables For Viewer visualization
	float	menuWindowLeft		= 0.f,
			menuWindowWidth		= 200.f,
			menuWindowHeight	= 475.f;

	/* Material properties (for linear elasticity) */
	
	Eigen::MatrixXd		VColor;
	Eigen::VectorXd		Z;

	ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5), ImGuiSetCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight), ImGuiSetCond_FirstUseEver);
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

			// if no mesh is selected, re-used the old one
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
				//eigenSolver.setV(viewer.data().V);
				//eigenSolver.setF(viewer.data().F);

				scaleMesh(viewer.data().V, viewer.data().F, 2.0);
				viewer.core.align_camera_center(viewer.data().V, viewer.data().F);

				viewer.data().show_lines = false;
				printf("A new mesh with %d vertices (%d faces).\n", viewer.data().V.rows(), viewer.data().F.rows());
				//viewer.data().set_mesh(V, F);
				viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));

				boolShowSamples = false;
				printf("mesh that we have  now: %d \n", viewer.data_list.size());

				eigenSolver->setV(viewer.data().V);
				eigenSolver->setF(viewer.data().F);
			}
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Save##Mesh1", ImVec2((w - p) / 2.f, 0))) {
			viewer.open_dialog_save_mesh();
		}
			

		//ImGui::RadioButton("Lap--Beltr.", &problemType, 0); ImGui::SameLine();
		//
		//if (ImGui::Button("Set-up Problem##1", ImVec2((w - p) / 2.f, 0))) {
		//	if (viewer.data().V.rows() < 1) return;
		//
		//	delete eigenSolver;
		//	eigenSolver = new MLSolverFunctions();
		//
		//	eigenSolver->setV(viewer.data().V);
		//	eigenSolver->setF(viewer.data().F);
		//
		//	printf("Mesh in eigensolver object: %dx%d \n", eigenSolver->getV().rows(), eigenSolver->getV().cols());
		//}

		static int axis = 0;		
	}

		/* STEP BY STEP CONSTRUCTION */
		if (ImGui::CollapsingHeader("Program Step by Step##1", ImGuiTreeNodeFlags_DefaultOpen)) {
			
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

			if (ImGui::Button("[2] Construct Hierarchy##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1) return;
				levelRatio = atof(lvlRatTxt);

				cout << "> Creating samples... " << endl;
				chrono::high_resolution_clock::time_point	t0, t1;
				chrono::duration<double>					duration;
				t0 = chrono::high_resolution_clock::now();

				Eigen::VectorXi sampSize;
				//setSamplingLevels(eigenSolver, levelRatio, numOfModes, sampSize);		
				std::cout << "Sought modes: " << numOfModes << std::endl;

				/* For 1 level (coarsest level) sampling only*/
				///eigenSolver->setSamplingLevels(sampSize, numOfModes, levelRatio);
				eigenSolver->setSamplingLevelsAuto(sampSize, numOfModes, levelRatio);
				eigenSolver->setInitialSystem(sampSize.rows());
				eigenSolver->constructSampleHierarchy(sampSize);
				numLevels = sampSize.size();

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
			if (isClampedBasis) {
				ImGui::RadioButton("X", &defAxis, 0); ImGui::SameLine();
				ImGui::RadioButton("Y", &defAxis, 1); ImGui::SameLine();
				ImGui::RadioButton("Z", &defAxis, 2); ImGui::SameLine();
				static char clampRegionTxt[10] = "0.01"; ImGui::InputText("Selection", clampRegionTxt, 10);
				clampRegion = atof(clampRegionTxt);
			}
			static double numOfSupport;
			static char numSupTxt[15] = "7.5"; ImGui::InputText("Num of Supports", numSupTxt, 6);

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
				//ml_eigSolver.testConstantFuncsProjection();

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


			if (ImGui::Button("[3] Coarse Basis only##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getStiffnessMatrix().rows() < 1) {
					cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
					return;
				}
				numOfSupport = atof(numSupTxt);
				eigenSolver->setClampBasis(isClampedBasis);
				

				chrono::high_resolution_clock::time_point	t0, t1;
				chrono::duration<double>					duration;
				t0 = chrono::high_resolution_clock::now();

				cout << "> Creating samples... " << endl;
				t0 = chrono::high_resolution_clock::now();

				eigenSolver->setInitialSystem(1);
				Eigen::VectorXi			sampSize;
				sampSize.resize(1);
				sampSize(0) = 2 * numOfModes;
				eigenSolver->constructSampleHierarchy(sampSize);
				t1 = chrono::high_resolution_clock::now();
				duration = t1 - t0;
				cout << "in " << duration.count() << " seconds." << endl;

				eigenSolver->constructBasisCoarseOnly(numOfSupport, defAxis, clampRegion);				

				useNonSmoothBasis = false;
			}

			// Non-smooth basis, based on SIGGRAPH-Asia Paper by Xiang (2019)
			if (ImGui::Button("[3] Non Smooth Basis##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getStiffnessMatrix().rows() < 1 || eigenSolver->getSampleSize() < 1) {
					cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
					return;
				}
				numOfSupport = atof(numSupTxt);
				eigenSolver->setClampBasis(isClampedBasis);
					
				eigenSolver->constructBasisNonSmooth(numOfSupport, defAxis, clampRegion);
				
				//ml_eigSolver.visualizeClusterOfSamplesLevel0(viewer);
				useNonSmoothBasis = true;
			}

			for (int i = 0; i < numLevels; i++)
			{
				string disp = "Cluster "; disp += to_string(i);
				char const* ca = disp.c_str();
				if (ImGui::Button(ca, ImVec2(w / (double)numLevels - p, 0))) {
					if (i == 0)
						eigenSolver->visualizeClusterOfSamplesLevel0(viewer);
					else
						eigenSolver->visualizeClusterOfSamplesLevelK(viewer, i);
				} ImGui::SameLine(0, p);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REDUCED, eigScale);
				cout << eigenSolver->getEigenValuesReduced().head(min(20, (int)eigenSolver->getEigenValuesReduced().rows())).transpose();

				//eigToShow = 3; 
				//
				//cout << "Showing id: " << eigToShow << endl; 
				//if (problemType == 0) {
				//	cout << "Eigen functions. \n";
				//	printf("Pos: %dx%d | vectors: %dx%d | id: %d \n", eigenSolver->getV().rows(), eigenSolver->getV().cols(), eigenSolver->getEigenVectorsReduced().rows(), eigenSolver->getEigenVectorsReduced().cols(), eigToShow);
				//	showEigenFunctions(viewer, eigenSolver->getEigenVectorsReduced(eigToShow), eigToShow);
				//}
				//else if (problemType == 1) {
				//	cout << "Vibration modes \n";
				//	showDeformationModes(viewer, eigenSolver->getV(), eigenSolver->getEigenVectorsReduced(eigToShow), eigScale);
				//}			
			}

			static char numRedEigTxt[32] = "20";	
			if (ImGui::Button("Print reduced eigenvals##1", ImVec2(w/2.0-p, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getBasisMatrix().rows() < 1 || eigenSolver->getEigenValuesReduced().rows()<1) {
					cout << "Error! Make sure you execute previous steps before solving the Low-Dim Eigenproblem." << endl;
					return;
				}
				static int numRedEigs = atoi(numRedEigTxt);
				cout << "Red eigenvalues" << eigenSolver->getEigenValuesReduced().head(min(numRedEigs, (int)eigenSolver->getEigenValuesReduced().rows())) << endl;
			}
			ImGui::SameLine();
			ImGui::InputText("#eigs", numRedEigTxt, 32);
			//
			//if (ImGui::Button("Save EigVals#Approx", ImVec2(w / 2.0, 0))) {
			//	int toStore = min(250, (int)ml_eigSolver.reducedEigVals.rows());
			//	writeDenseMatrixToFile(ml_eigSolver.reducedEigVals.block(0, 0, toStore, 1), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_Red_" + to_string(toStore) + ".txt");
			//}
			//ImGui::SameLine(0, p);
			//if (ImGui::Button("Save EigVect#Approx", ImVec2(w / 2.0, 0))) {
			//	int toStore = 250;
			//	writeDenseMatrixToFile(ml_eigSolver.Basis*ml_eigSolver.reducedEigVects.block(0, 0, ml_eigSolver.V.rows(), toStore), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_Red_" + to_string(toStore) + ".txt");
			//}
		}

		ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();

		/* VISUALIZATION WINDOW */
		if (ImGui::CollapsingHeader("Visualization##MLEigSolver1", ImGuiTreeNodeFlags_DefaultOpen)) {
			/* For Sampling */
			static double nEps = 0.005;
			if (ImGui::Button("Show Samples##1", ImVec2((w), 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1) {
					cout << "Error! Make sure you compute new samples before viewing them." << endl;
					return;
				}

				boolShowSamples = !boolShowSamples;
				if (boolShowSamples) {
					visualizeSampleMulti(viewer, eigenSolver->getV(), eigenSolver->getF(), eigenSolver->getSample(), eigenSolver->getSubspaceSize(), nEps);
				}
				else {
					viewer.data().points.resize(0, 6);
				}
				viewer.data().point_size = 10.0f;
			}

			if (ImGui::Button("S1", ImVec2((w/5.0), 0))) {
				visualizeSample_1(viewer, eigenSolver->getV(), eigenSolver->getF(), eigenSolver->getSample(), eigenSolver->getSubspaceSize(), nEps);
			} ImGui::SameLine(0, p);
			if (ImGui::Button("S2", ImVec2((w / 5.0), 0))) {
				visualizeSample_2(viewer, eigenSolver->getV(), eigenSolver->getF(), eigenSolver->getSample(), eigenSolver->getSubspaceSize()), nEps;
			} ImGui::SameLine(0, p);
			if (ImGui::Button("S3", ImVec2((w / 5.0), 0))) {
				visualizeSample_3(viewer, eigenSolver->getV(), eigenSolver->getF(), eigenSolver->getSample(), eigenSolver->getSubspaceSize(), nEps);
			} ImGui::SameLine(0, p);
			if (ImGui::Button("Lvl1", ImVec2((w / 5.0), 0))) {
				visualizeSampleFirstLevel(viewer, eigenSolver->getV(), eigenSolver->getF(), eigenSolver->getSample(), eigenSolver->getSubspaceSize(), nEps);
			} ImGui::SameLine(0, p);
			if (ImGui::Button("Lvl2", ImVec2((w / 5.0), 0))) {
				visualizeSampleSecondLevel(viewer, eigenSolver->getV(), eigenSolver->getF(), eigenSolver->getSample(), eigenSolver->getSubspaceSize(), nEps);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REDUCED, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REDUCED, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REDUCED, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REFERENCE, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REFERENCE, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REFERENCE, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_SIM, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_SIM, eigScale);
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

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_SIM, eigScale);
			}

			/* Overlays */
			ImGui::Checkbox("Wireframe", &(viewer.data().show_lines));
			ImGui::SameLine();
			ImGui::Checkbox("Fill Faces", &(viewer.data().show_faces));
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
				viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
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
					solver->constructDiffusionTensor(min(numOfModes, 50), weight, solver->getEigenVectorsSIM(), solver->getEigenValuesSIM(), diffTensor2);					
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

void SIMMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, MLEigenSolver*& eigenSolver) {
	// Variables For Viewer visualization
	float	menuWindowLeft		= 475.f,
			menuWindowWidth		= 200.f,
			menuWindowHeight	= 675.f;
	
	ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5), ImGuiSetCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight), ImGuiSetCond_FirstUseEver);
	ImGui::Begin("Subspace Iteration Method##Menu", nullptr, ImGuiWindowFlags_NoSavedSettings);

	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

	static int SIMType = 0;			// 0: M-L, 1: SGP, 2:random, 3: Bathe
	static int shiftType = 3;		// 0: Bathe, 1: Itoh/Wilson, 2: Gong, 3: Zhao
	static bool isClampedEigen = false; 
		
	if (ImGui::CollapsingHeader("Eigenproblem with SIM", ImGuiTreeNodeFlags_DefaultOpen)) {
		/* some parameters */
		//static int		numOfModes	= 5;
		static int		maxIter		= 5;
		static double	tol			= 1e-4;
		/* Type of the SIM to be computed*/
		std::map<int, MLEigenSolver::SIMType>			typeMap;
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(0, MLEigenSolver::SIMType::SIM_MultiLevel));
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(1, MLEigenSolver::SIMType::SIM_1level_WarmStart));
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(2, MLEigenSolver::SIMType::SIM_1level_Random));
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(3, MLEigenSolver::SIMType::SIM_1level_Bathe));
		
		/* Type of shifting approach */
		std::map<int, MLEigenSolver::ShiftType>			shiftMap;
		shiftMap.insert(std::pair<int, MLEigenSolver::ShiftType>(0, MLEigenSolver::ShiftType::SHIFT_BATHE));
		shiftMap.insert(std::pair<int, MLEigenSolver::ShiftType>(1, MLEigenSolver::ShiftType::SHIFT_ITOH_WILSON));
		shiftMap.insert(std::pair<int, MLEigenSolver::ShiftType>(2, MLEigenSolver::ShiftType::SHIFT_GONG));
		shiftMap.insert(std::pair<int, MLEigenSolver::ShiftType>(3, MLEigenSolver::ShiftType::SHIFT_ZHAO));



		if (ImGui::CollapsingHeader("SIM Option##1", ImGuiTreeNodeFlags_DefaultOpen)) {
			static char numModesTxt[10] = "125"; ImGui::InputText("Num Eigs", numModesTxt, 10);			
			static char maxIterTxt[10] = "20"; ImGui::InputText("Max Iter", maxIterTxt, 10);			
			static char tolTxt[15] = "0.0001"; ImGui::InputText("Tolerance##1", tolTxt, 15);
			numOfModes	= atoi(numModesTxt);
			maxIter		= atoi(maxIterTxt);
			tol			= atof(tolTxt);

			ImGui::RadioButton("Random init",	&SIMType, 2);			ImGui::SameLine();
			ImGui::RadioButton("Bathe init",	&SIMType, 3);			//ImGui::SameLine();
			ImGui::RadioButton("SGP init",		&SIMType, 1);			ImGui::SameLine();
			ImGui::RadioButton("Multi-level",	&SIMType, 0);			//ImGui::SameLine();
		}

		

		static bool isShiftedSIM = true;
		ImGui::Checkbox("with shift", &(isShiftedSIM));
		static double zhaoShiftParam = 0.25;
		if (isShiftedSIM) {
			ImGui::RadioButton("Bathe##Shift",		&shiftType, 0);			ImGui::SameLine();
			ImGui::RadioButton("Wilson##ItohShift", &shiftType, 1);			
			ImGui::RadioButton("Gong##Shift",		&shiftType, 2);			ImGui::SameLine();
			ImGui::RadioButton("Zhao##Shift",		&shiftType, 3);
			static char zhaoTxt[10] = "0.25";		ImGui::InputText("Zhao shift", zhaoTxt, 10);
			zhaoShiftParam = atof(zhaoTxt);
		}

		static double qParam;
		static char qParamTxt[10] = "2.0"; ImGui::InputText("q Param", qParamTxt, 10);
		qParam = atof(qParamTxt);

		if (ImGui::Button("Solve Eigenproblem", ImVec2(w, 0))) {
			eigenSolver->qParam = qParam;
			eigenSolver->zhaoShiftParam = zhaoShiftParam;
			eigenSolver->solve(numOfModes, maxIter, tol, isShiftedSIM, shiftMap[shiftType], defAxis, clampRegion, 10.0, 7.5);
		}

		if (ImGui::Button("Perform SIM iters.##1", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows()<1 ) return;
			if ((SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;					
			
			eigenSolver->qParam = qParam;
			eigenSolver->zhaoShiftParam = zhaoShiftParam;
			std::cout << "Shift param = " << zhaoShiftParam << std::endl; 
			eigenSolver->performSubspaceIterationMethod(typeMap[SIMType], numOfModes, maxIter, tol, isShiftedSIM, shiftMap[shiftType], defAxis, clampRegion);

			printf("Successfully computing the eigenvalues \n");
			//cout << "Eigenvalues: \n" << eigenSolver->getEigenValuesSIM() << endl;
			//cout << "EigenVectors: \n" << eigenSolver->getEigenVectorsSIM().topLeftCorner(6, 9) << endl;
		}

		if (ImGui::Button("Scripted (1) OURS", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows()<1) return;
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
				for(int k=0; k<wantedTols.size(); ++k){
					// ML-with shift
					eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
					// ML no shift
					//eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
					// random
					//eigenSolver->performSubspaceIterationMethod(typeMap[2], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
					// // bathe init vector
					//eigenSolver->performSubspaceIterationMethod(typeMap[3], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
				}
			}			
		}

		if (ImGui::Button("Scripted (2)--non-smooth", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows()<1) return;
			//if ((SIMType == 0 || SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;

			/*  Testing on :
			*	(5) NL non smooth (with shift)
			*	(6) ML non smooth (without shift)
			*/

			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;

			Eigen::VectorXi			sampSize;
			//std::vector<double> levelRatio{4,5,7,8,10,15,20};
			std::vector<double>		levelRatio{ 10 };
			//std::vector<double>		levelSupport{ 5.0, 7.5, 10.0, 12.5, 15.0, 20.0 };
			std::vector<double>		levelSupport{ 5 };
			//std::vector<double>	levelZhaoParam{0.1,0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45};
			//std::vector<double>		levelZhaoParam{ 0.25 };			
			std::vector<double>		levelZhaoParam{ 0.15 };
			vector<int>				wantedModes{ 50, 100, 250 };
			//vector<int>				wantedModes{ 500 };
			//vector<double>			wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };
			vector<double>			wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5 };
			//vector<double>			wantedTols{ 1e-8, 1e-9, 1e-10 };
			//vector<double>			wantedTols{ 1e-10 };
			//vector<double>			wantedTols{ 1e-4 };
			int						maxIters = 50;

			for (int j = 0; j < wantedModes.size(); ++j) {
				std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;

				for (int i = 0; i < levelRatio.size(); ++i) {
					std::cout << ">>> LEVEL Ratio of " << levelRatio[i] << std::endl;
					//Sampling
					cout << "> Creating samples... " << endl;
					t0 = chrono::high_resolution_clock::now();
					//setSamplingLevels(eigenSolver, wantedModes[j], sampSize);
					eigenSolver->setSamplingLevels(sampSize, wantedModes[j], levelRatio[i]);
					eigenSolver->setInitialSystem(sampSize.rows());
					eigenSolver->constructSampleHierarchy(sampSize);
					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "in " << duration.count() << " seconds." << endl;

					for (int l = 0; l < levelSupport.size(); ++l) {
						std::cout << ">>> SUPPORT of " << levelSupport[l] << std::endl;

						// Constructing basis... [1] Smooth basis
						cout << "> Constructing basis... " << endl;
						t0 = chrono::high_resolution_clock::now();
						eigenSolver->setClampBasis(false);
						//eigenSolver->constructBasis(levelSupport[l], defAxis, clampRegion);
						//eigenSolver->constructBasisCoarseOnly(levelSupport[l]);		// for SGP setting
						eigenSolver->constructBasisNonSmooth(levelSupport[l], defAxis, clampRegion);
						useNonSmoothBasis = false;
						t1 = chrono::high_resolution_clock::now();
						duration = t1 - t0;
						cout << "in " << duration.count() << " seconds." << endl;

						// Compute restricted problem
						cout << "> Solving restricted system... " << endl;
						t0 = chrono::high_resolution_clock::now();
						eigenSolver->solveRestrictedProblem();
						t1 = chrono::high_resolution_clock::now();
						duration = t1 - t0;
						cout << "in " << duration.count() << " seconds." << endl;

						for (int m = 0; m < levelZhaoParam.size(); ++m) {
							eigenSolver->zhaoShiftParam = levelZhaoParam[m];
							std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;

							std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;
							for (int k = 0; k < wantedTols.size(); ++k) {
								std::cout << ">>> TOLERANCE of " << wantedTols[k] << std::endl;

								// ML-without shift
								eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);

								// ML-with shift 0.25
								//eigenSolver->zhaoShiftParam = 0.25;
								std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;
								eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);

								// ML-with shift 0.35
								//eigenSolver->zhaoShiftParam = 0.35;
								//std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;
								//eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);

								//SGP
								//eigenSolver->performSubspaceIterationMethod(typeMap[1], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);

								//SGP with shift
								//eigenSolver->performSubspaceIterationMethod(typeMap[1], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);

								// Random
								//eigenSolver->performSubspaceIterationMethod(typeMap[2], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);
							}
						}
					}
				}
			}
		}

		if (ImGui::Button("Scripted (3)--SGP", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows()<1) return;
			//if ((SIMType == 0 || SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;


			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;

			/*  Testing on :
			*	(7) SGP as warm start
			*/

			int maxIters = 50;
			//vector<int> wantedModes{ 50, 100, 250 };
			vector<int> wantedModes{ 50, 100 };
			//vector<double> wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5 };
			vector<double> wantedTols{ 1e-4 };
			std::vector<double>		levelSupport{ 7.5 };

			for (int j = 0; j < wantedModes.size(); ++j) {

				cout << "> Creating samples... " << endl;
				t0 = chrono::high_resolution_clock::now();
				//setSamplingLevels(eigenSolver, wantedModes[j], sampSize);
				//eigenSolver->setSamplingLevels(sampSize, wantedModes[j], levelRatio[i]);
				eigenSolver->setInitialSystem(1);
				Eigen::VectorXi			sampSize;
				sampSize.resize(1);
				sampSize(0) = 2 * wantedModes[j];
				eigenSolver->constructSampleHierarchy(sampSize);
				t1 = chrono::high_resolution_clock::now();
				duration = t1 - t0;
				cout << "in " << duration.count() << " seconds." << endl;

				
				for (int l = 0; l < levelSupport.size(); ++l) {
					std::cout << ">>> SUPPORT of " << levelSupport[l] << std::endl;
					// Constructing basis... [1] Smooth basis
					cout << "> Constructing basis... " << endl;
					t0 = chrono::high_resolution_clock::now();
					eigenSolver->setClampBasis(false);
					//eigenSolver->constructBasis(levelSupport[l], defAxis, clampRegion);
					eigenSolver->constructBasisCoarseOnly(levelSupport[l]);		// for SGP setting
					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "in " << duration.count() << " seconds." << endl;
																				// Compute restricted problem
					cout << "> Solving restricted system... " << endl;
					t0 = chrono::high_resolution_clock::now();
					eigenSolver->solveRestrictedProblem();
					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "in " << duration.count() << " seconds." << endl;

					for (int k = 0; k < wantedTols.size(); ++k) {
						// sgp
						eigenSolver->performSubspaceIterationMethod(typeMap[1], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);
						// random
						//eigenSolver->performSubspaceIterationMethod(typeMap[2], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
					
					}
				}
			}
		}

		if (ImGui::Button("Scripted (4) ALL", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows()<1) return;
			if ((SIMType == 0 || SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;

			/*  Testing on :
			*	(1) random
			*	(3) ML without shift
			*   (4) ML with shift-once
			*/

			int maxIters = 50;
			//vector<int> wantedModes{ 50, 100, 250 };
			vector<int> wantedModes{ 50, 100 };
			//vector<int> wantedModes{ 25 };
			vector<double> wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5 };
			//vector<double> wantedTols{ 1e-2};
			eigenSolver->zhaoShiftParam = 0.25;

			for (int j = 0; j < wantedModes.size(); ++j) {
				for (int k = 0; k<wantedTols.size(); ++k) {
					// ML-with shift
					eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
					// ML no shift
					eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
					// random
					eigenSolver->performSubspaceIterationMethod(typeMap[2], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
					// // bathe init vector
					//eigenSolver->performSubspaceIterationMethod(typeMap[3], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
				}
			}

			/*  Testing on :
			*	(5) NL non smooth (with shift)
			*	(6) ML non smooth (without shift)
			*/
			eigenSolver->constructBasisNonSmooth(5.0);
			eigenSolver->solveRestrictedProblem();
			eigenSolver->zhaoShiftParam = 0.15;
			for (int j = 0; j < wantedModes.size(); ++j) {
				for (int k = 0; k<wantedTols.size(); ++k) {
					// ML-with shift (non-smooth)
					eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
					// ML no shift (non-smooth)
					eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[0], defAxis, clampRegion);
				}
			}

			/*  Testing on :
			*	(7) SGP as warm start
			*/
			eigenSolver->constructBasisCoarseOnly(12.5);
			eigenSolver->solveRestrictedProblem();

			for (int j = 0; j < wantedModes.size(); ++j) {
				for (int k = 0; k<wantedTols.size(); ++k) {
					// ML-with shift (non-smooth)
					eigenSolver->performSubspaceIterationMethod(typeMap[1], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);
				}
			}
		}

		if (ImGui::Button("Scripted (5) Shift", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows()<1) return;
			if ((SIMType == 0 || SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;

			/*  Testing on :
			*	varying shift parameters on kitten model
			*/

			int maxIters = 50;
			//vector<int> wantedModes{ 50, 100, 250 };
			vector<int> wantedModes{ 50, 100, 250 };
			vector<double> wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5 };
			vector<double> shiftParams{0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45 };

			for (int j = 0; j < wantedModes.size(); ++j) {
				for (int k = 0; k<wantedTols.size(); ++k) {
					for (int l = 0; l<shiftParams.size(); ++l) {
						eigenSolver->zhaoShiftParam = shiftParams[l];
						eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
					}
				}
			}
		}

		//if (ImGui::Button("Scripted (5) Var. Ratio", ImVec2(w, 0))) {
		if (ImGui::Button("Scripted (5) Var. Support", ImVec2(w, 0))) {
			Eigen::VectorXi sampSize;
			//double numOfSupport = 5.0;

			//std::vector<double> levelRatio{4,5,7,8,10,15,20};
			std::vector<double>		levelRatio{ 10 };
			std::vector<double>		levelSupport{ 5.0, 7.5, 10.0, 12.5, 15.0, 20.0 };
			//std::vector<double>	levelSupport{ 5.0 };
			//std::vector<double>	levelZhaoParam{0.1,0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45};
			std::vector<double>		levelZhaoParam{ 0.25};
			vector<int>				wantedModes{ 50, 100, 250 };
			vector<double>			wantedTols{ 1e-4 };
			int						maxIters = 50;

			for (int j = 0; j < wantedModes.size(); ++j) {
				std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;

				for (int i = 0; i < levelRatio.size(); ++i) {
					std::cout << ">>> LEVEL Ratio of " << levelRatio[i] << std::endl;
					for (int l = 0; l < levelSupport.size(); ++l) {
						std::cout << ">>> SUPPORT of " << levelSupport[l] << std::endl;
						//Sampling
						setSamplingLevels(eigenSolver, levelRatio[i], wantedModes[j], sampSize);
						eigenSolver->setInitialSystem(sampSize.rows());
						eigenSolver->constructSampleHierarchy(sampSize);

						// Constructing basis...
						eigenSolver->setClampBasis(false);
						eigenSolver->constructBasis(levelSupport[l], defAxis, clampRegion);
						useNonSmoothBasis = false;

						// Compute restricted problem
						eigenSolver->solveRestrictedProblem();

						for (int m = 0; m < levelZhaoParam.size(); ++m) {
							std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;
							eigenSolver->zhaoShiftParam = levelZhaoParam[m];
							
							for (int k = 0; k < wantedTols.size(); ++k) {
								std::cout << ">>> TOLERANCE of " << wantedTols[k] << std::endl;
								// ML-with shift
								eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
							}
						}
					}
				}
			}
		}

		if (ImGui::Button("Scripted (6) Auto Levels", ImVec2(w, 0))) {
			
			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;

			Eigen::VectorXi			sampSize;
			//std::vector<double> levelRatio{4,5,7,8,10,15,20};
			std::vector<double>		levelRatio{ 10 };
			//std::vector<double>		levelSupport{ 5.0, 7.5, 10.0, 12.5, 15.0, 20.0 };
			std::vector<double>		levelSupport{ 7.5 };
			//std::vector<double>	levelZhaoParam{0.1,0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45};
			//std::vector<double>		levelZhaoParam{ 0.25 };			
			std::vector<double>		levelZhaoParam{ 0.25 };			
			//vector<int>				wantedModes{ 50, 100, 250};
			vector<int>				wantedModes{25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000 };
			//vector<int>				wantedModes{ 500, 750, 1000 };
			//vector<double>			wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };
			//vector<double>			wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5};
			//vector<double>			wantedTols{ 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };
			//vector<double>			wantedTols{ 1e-10 };
			vector<double>			wantedTols{ 1e-4 };
			int						maxIters = 50;

			for (int j = 0; j < wantedModes.size(); ++j) {
				std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;

				for (int i = 0; i < levelRatio.size(); ++i) {
					std::cout << ">>> LEVEL Ratio of " << levelRatio[i] << std::endl;
					//Sampling
					cout << "> Creating samples... " << endl;
					t0 = chrono::high_resolution_clock::now();
					//setSamplingLevels(eigenSolver, wantedModes[j], sampSize);
					eigenSolver->setSamplingLevels(sampSize, wantedModes[j], levelRatio[i]);
					eigenSolver->setInitialSystem(sampSize.rows());
					eigenSolver->constructSampleHierarchy(sampSize);
					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "in " << duration.count() << " seconds." << endl;

					for (int l = 0; l < levelSupport.size(); ++l) {
						std::cout << ">>> SUPPORT of " << levelSupport[l] << std::endl;

						// Constructing basis... [1] Smooth basis
						cout << "> Constructing basis... " << endl;
						t0 = chrono::high_resolution_clock::now();
						eigenSolver->setClampBasis(false);
						eigenSolver->constructBasis(levelSupport[l], defAxis, clampRegion);
						//eigenSolver->constructBasisCoarseOnly(levelSupport[l]);		// for SGP setting
						//eigenSolver->constructBasisNonSmooth(levelSupport[l], defAxis, clampRegion);
						useNonSmoothBasis = false;
						t1 = chrono::high_resolution_clock::now();
						duration = t1 - t0;
						cout << "in " << duration.count() << " seconds." << endl;

						// Compute restricted problem
						cout << "> Solving restricted system... " << endl;
						t0 = chrono::high_resolution_clock::now();
						eigenSolver->solveRestrictedProblem();
						t1 = chrono::high_resolution_clock::now();
						duration = t1 - t0;
						cout << "in " << duration.count() << " seconds." << endl;

						for (int m = 0; m < levelZhaoParam.size(); ++m) {
							eigenSolver->zhaoShiftParam = levelZhaoParam[m];
							std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;
						
							std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;
							for (int k = 0; k < wantedTols.size(); ++k) {
								std::cout << ">>> TOLERANCE of " << wantedTols[k] << std::endl;
								
								// ML-without shift
								//eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);

								// ML-with shift 0.25
								//eigenSolver->zhaoShiftParam = 0.25;
								//std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;
								eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);

								// ML-with shift 0.35
								//eigenSolver->zhaoShiftParam = 0.35;
								//std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;
								//eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
																
								//SGP
								//eigenSolver->performSubspaceIterationMethod(typeMap[1], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);

								//SGP with shift
								//eigenSolver->performSubspaceIterationMethod(typeMap[1], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
								
								// Random
								//eigenSolver->performSubspaceIterationMethod(typeMap[2], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);
								}
						}
					}
				}
			}
		}

		if (ImGui::Button("Scripted (7) Disc. Converged", ImVec2(w, 0))) {

			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;

			Eigen::VectorXi			sampSize;
			//std::vector<double> levelRatio{4,5,7,8,10,15,20};
			std::vector<double>		levelRatio{ 10 };
			//std::vector<double>		levelSupport{ 5.0, 7.5, 10.0, 12.5, 15.0, 20.0 };
			std::vector<double>		levelSupport{ 7.5 };
			//std::vector<double>	levelZhaoParam{0.1,0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45};
			//std::vector<double>		levelZhaoParam{ 0.25 };			
			std::vector<double>		levelZhaoParam{ 0.25 };
			vector<int>				wantedModes{ 50, 200, 500};
			std::vector<double>		qParam{ 2.0, 1.5 };
			//vector<int>				wantedModes{ 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000 };
			//vector<int>				wantedModes{ 500, 750, 1000 };
			//vector<double>			wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };
			//vector<double>			wantedTols{ 1e-2, 1e-3, 1e-4, 1e-5};
			//vector<double>			wantedTols{ 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };
			//vector<double>			wantedTols{ 1e-10 };
			vector<double>			wantedTols{ 1e-4 };
			int						maxIters = 50;

			for (int j = 0; j < wantedModes.size(); ++j) {
				std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;

				for (int i = 0; i < levelRatio.size(); ++i) {
					std::cout << ">>> LEVEL Ratio of " << levelRatio[i] << std::endl;					

					for (int l = 0; l < levelSupport.size(); ++l) {
						std::cout << ">>> SUPPORT of " << levelSupport[l] << std::endl;		

						for (int n = 0; n < qParam.size(); ++n) {
							eigenSolver->qParam = qParam[n];

							for (int m = 0; m < levelZhaoParam.size(); ++m) {
								eigenSolver->zhaoShiftParam = levelZhaoParam[m];
								std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;

								std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;
								for (int k = 0; k < wantedTols.size(); ++k) {
									std::cout << ">>> TOLERANCE of " << wantedTols[k] << std::endl;
									// Our current solver (shift as FALSE)
									eigenSolver->solve(wantedModes[j], maxIters, wantedTols[k], false, MuLES::MLEigenSolver::ShiftType::SHIFT_BATHE, 0, 0, levelRatio[i], levelSupport[l]);
									// Do not include the converged ones into consideratoin (shift as TRUE)
									eigenSolver->solve(wantedModes[j], maxIters, wantedTols[k], true, MuLES::MLEigenSolver::ShiftType::SHIFT_BATHE, 0, 0, levelRatio[i], levelSupport[l]);
								}	// toleransi
							}		// zhao's shift param
						}			// qParam
					}				// support radius
				}					// level ratio
			}						// wanted modes
		}
	}


	ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();

	/* [REFERENCE] Eigenvectors */
	static int solverType = 0; // 0: Matlab, 1: Spectra, 2: CUDA (GPU)
	if (ImGui::CollapsingHeader("Compute Ref. Eigen", ImGuiTreeNodeFlags_DefaultOpen)) {
		/* Type of the SIM to be computed*/
		std::map<int, MLEigenSolver::EigenSolverType>			solverTypeMap;
		solverTypeMap.insert(std::pair<int, MLEigenSolver::EigenSolverType>(0, MLEigenSolver::EigenSolverType::EIG_SOLVER_MATLAB));
		solverTypeMap.insert(std::pair<int, MLEigenSolver::EigenSolverType>(1, MLEigenSolver::EigenSolverType::EIG_SOLVER_SPECTRA));
		solverTypeMap.insert(std::pair<int, MLEigenSolver::EigenSolverType>(2, MLEigenSolver::EigenSolverType::EIG_SOLVER_CUDA));

		static char numEig[6] = "100"; ImGui::InputText("Num eigenpairs", numEig, 6);

		ImGui::RadioButton("Matlab", &solverType, 0); ImGui::SameLine();
		ImGui::RadioButton("Spectra", &solverType, 1); ImGui::SameLine();
		ImGui::RadioButton("CUDA", &solverType, 2);

		static double tol;
		static char toleTxt[15] = "0.0001"; ImGui::InputText("Tolerance##2", toleTxt, 15);
		static int maxIter;
		static char maxItersTxt[15] = "500"; ImGui::InputText("Max Iters##1", maxItersTxt, 15);
		tol = stof(toleTxt);
		maxIter = stoi(maxItersTxt);

		ImGui::Checkbox("Clamped Eigen", &isClampedEigen);
		if (isClampedEigen) {

			ImGui::RadioButton("X", &defAxis, 0); ImGui::SameLine();
			ImGui::RadioButton("Y", &defAxis, 1); ImGui::SameLine();
			ImGui::RadioButton("Z", &defAxis, 2); ImGui::SameLine();
			static char selScale[10] = "0.01"; ImGui::InputText("Selection", selScale, 10);
			clampRegion = atof(selScale);
		}



		if (ImGui::Button("Compute Reference##1", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows() < 1) return;

			eigToCompute = atoi(numEig);	
			tol = stof(toleTxt);
			maxIter = stoi(maxItersTxt);

			cout << "> Computing reference eigenproblem ";

			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;
			t0 = chrono::high_resolution_clock::now();

			if (!isClampedEigen) {
				eigenSolver->computeRefEigenProblem(eigToCompute, solverTypeMap[solverType], maxIter, tol);
			}
			else {
				eigenSolver->computeRefEigenProblemWithSelector(eigToCompute, solverTypeMap[solverType], defAxis, clampRegion);
			}

			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << " in " << duration.count() << " seconds." << endl;

			cout << "Ref eigenvalues: \n" << eigenSolver->getEigenValuesRef() << endl;
		}




		if (ImGui::Button("Spectra Test", ImVec2(w, 0))) {
			tol = atof(toleTxt);
			//std::vector<double> tolVect{1e-2, 1e-4, 1e-7, 1e-10};
			//std::vector<double>		tolVect{ 5e-2, 1e-2, 1e-3, 1e-4, 1e-5 };
			std::vector<double>		tolVect{1e-1, 5.0e-2, 1e-4 };
			vector<int>				wantedModes{ 250 };
			//vector<int>				wantedModes{ 100, 150, 250 };
			//eigToCompute = atoi(numEig);
			Eigen::MatrixXd EigVects;
			Eigen::VectorXd EigVals;
			Eigen::VectorXd timing(1);

			cout << "> Computing reference eigenproblem ";
			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;

			for (int j = 0; j < wantedModes.size(); ++j) {
				for (int i = 0; i < tolVect.size(); ++i) {

					t0 = chrono::high_resolution_clock::now();

					//computeEigenSpectraSparse(eigenSolver->getStiffnessMatrix(), eigenSolver->getMassMatrix(), eigToCompute, EigVects, EigVals, 0, 1000, tol);
					computeEigenSpectraSparse(eigenSolver->getStiffnessMatrix(), eigenSolver->getMassMatrix(), wantedModes[j], EigVects, EigVals, 1, 5000, tolVect[i]);
					
					Eigen::VectorXd conv2;
					eigenSolver->checkConvergenceCat2(EigVects, EigVals, wantedModes[j], eigenSolver->getMassMatrix(), eigenSolver->getStiffnessMatrix(), conv2);
					std::cout << "Conv (" << tolVect[i] << ") : \n\t" << conv2.transpose() << std::endl;
					//tol /= 10000;

					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "\n in " << duration.count() << " seconds." << endl;
					timing(0) = duration.count();

					string modelName = "Sphere_" + to_string(eigenSolver->getV().rows()) + "_" + to_string(eigenSolver->getF().cols());
					string filename = "../../Data/" + modelName + "_EigVects-Spectra_" + to_string(wantedModes[j]) + "_" + to_string(tolVect[i]) + ".txt";
					writeDenseMatrixToFile(EigVects, filename);
					filename = "../../Data/" + modelName + "_EigVals-Spectra_" + to_string(wantedModes[j]) + "_" + to_string(tolVect[i]) + ".txt";
					writeDenseMatrixToFile(EigVals, filename);
					filename = "../../Data/" + modelName + "_Tolerance-Spectra_" + to_string(wantedModes[j]) + "_" + to_string(tolVect[i]) + ".txt";
					writeDenseMatrixToFile(timing, filename);
					writeDenseMatrixToFile(conv2, filename);


					//cout << "Ref eigenvalues: \n" << EigVals << std::endl;
					cout << "Ref eigenvalues: \n";
					//std::cout << EigVals(0) << std::endl;
					//printVector(EigVals.tail(wantedModes[j]));
					printVector(EigVals.head(wantedModes[j]));
				}
			}
		}

		if (ImGui::Button("Matlab Test", ImVec2(w, 0))) {
			tol = atof(toleTxt);
			//std::vector<double> tolVect{1e-2, 1e-4, 1e-7, 1e-10};
			std::vector<double>		tolVect{ 1e-2, 1e-3, 1e-4, 1e-5, 1e-10 };
			//std::vector<double>		tolVect{1e-5};
			vector<int>				wantedModes{ 50, 100, 250 };
			//eigToCompute = atoi(numEig);
			Eigen::MatrixXd EigVects;
			Eigen::VectorXd EigVals;
			Eigen::VectorXd timing(1);

			cout << "> Computing reference eigenproblem ";
			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;

			for (int j = 0; j < wantedModes.size(); ++j) {
				for (int i = 0; i < tolVect.size(); ++i) {

					t0 = chrono::high_resolution_clock::now();
					
					computeEigenMatlab(eigenSolver->getStiffnessMatrix(), eigenSolver->getMassMatrix(), wantedModes[j], EigVects, EigVals, tolVect[i], "");

					Eigen::VectorXd conv2;
					eigenSolver->checkConvergenceCat2(EigVects, EigVals, wantedModes[j], eigenSolver->getMassMatrix(), eigenSolver->getStiffnessMatrix(), conv2);
					std::cout << "Conv (" << tolVect[i] << ") : \n\t" << conv2.transpose() << std::endl;
					//tol /= 10000;

					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "\n in " << duration.count() << " seconds." << endl;
					timing(0) = duration.count();

					string modelName = "Sphere_" + to_string(eigenSolver->getV().rows()) + "_" + to_string(eigenSolver->getF().cols());
					string filename = "../../Data/" + modelName + "_EigVects-Matlab_" + to_string(wantedModes[j]) + "_" + to_string(tolVect[i]) + ".txt";
					writeDenseMatrixToFile(EigVects, filename);
					filename = "../../Data/" + modelName + "_EigVals-Matlab_" + to_string(wantedModes[j]) + "_" + to_string(tolVect[i]) + ".txt";
					writeDenseMatrixToFile(EigVals, filename);
					filename = "../../Data/" + modelName + "_Tolerance-Matlab_" + to_string(wantedModes[j]) + "_" + to_string(tolVect[i]) + ".txt";
					writeDenseMatrixToFile(timing, filename);
					writeDenseMatrixToFile(conv2, filename);


					//cout << "Ref eigenvalues: \n" << EigVals << std::endl;
					cout << "Ref eigenvalues: \n";
					//std::cout << EigVals(0) << std::endl;
					//printVector(EigVals.tail(wantedModes[j]));
					printVector(EigVals.head(wantedModes[j]));
				}
			}
		}

		if (ImGui::Button("Compute Reference Multi", ImVec2(w, 0))) {
			//vector<int> numEigs{50, 100, 250};
			vector<int> numEigs{ 50};
			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;		


			for (int j = 0; j < 2; ++j) {
				for (int k = 0; k < 3; ++k) {
					if (j == 0) std::cout << "Computing ref in Matlab |" << numEigs[k] << std::endl;
					else if(j==1) std::cout << "Computing ref in Spectra |" << numEigs[k] << std::endl;

					t0 = chrono::high_resolution_clock::now();

					eigenSolver->computeRefEigenProblem(numEigs[k], solverTypeMap[j]);
					Eigen::VectorXd conv2;
					eigenSolver->checkConvergenceCat2(eigenSolver->getEigenVectorsRef(), eigenSolver->getEigenValuesRef(), numEigs[k], eigenSolver->getMassMatrix(), eigenSolver->getStiffnessMatrix(), conv2);
					std::cout << "Conv: " << conv2 << std::endl;

					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "___in " << duration.count() << " seconds." << endl;

					cout << "Ref eigenvalues: \n" << eigenSolver->getEigenValuesRef() << endl;
				}
			}
			
		}

		if (ImGui::Button("Read Ref. Eigs", ImVec2(w, 0))) {
			// read from file 
		}
		if (ImGui::Button("Save EigVals#REF", ImVec2(w / 2, 0))) {
			writeDenseMatrixToFile(eigenSolver->getEigenValuesRef(), "../../Data/" + model + "_EigVals_Ref_" + to_string(eigenSolver->getEigenVectorsRef().rows()) + ".txt");
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Save EigVect#REF", ImVec2(w / 2, 0))) {
			writeDenseMatrixToFile(eigenSolver->getEigenVectorsRef(), "../../Data/" + model + "_EigVects_Ref_" + to_string(eigenSolver->getEigenVectorsRef().rows()) + ".txt");
		}
	}
	ImGui::End();
}

/* Sampling with fixed target,
** The number of levels and ratio between levels is determined accordingly. */
void setSamplingLevels(MLEigenSolver*& eigenSolver, int numEigen, Eigen::VectorXi& sampSize)
{
	std::cout << "Auto sampling for " << numEigen << " number of eigenvalues \n";
	double	spRatio			= 1.25;
	int		sizeSubsp		= spRatio * 2 * numEigen;
	double	desiredRatio	= 10.0;
	double	VSRatio			= (double)eigenSolver->getV().rows() / (double) sizeSubsp;
	double	logRatio		= log(VSRatio) / log(desiredRatio);
	int		numLevels		= std::round(logRatio);
	double	sampleRatio		= pow(VSRatio, 1.0 / (double) numLevels);
	std::cout << "Wanted mode: " << numEigen << " | subspace dim: " << 2 * numEigen << " | provided subspace: " << sizeSubsp << "(" << spRatio << ")\n";
	std::cout << ">>> LEVEL Ratio of " << sampleRatio << std::endl;

	int numSample = eigenSolver->getV().rows();
	sampSize.resize(numLevels);
	for (int i = 0; i < numLevels; ++i) {
		numSample = (int)round(numSample / sampleRatio);
		sampSize(i) = numSample;
	}
}

/* Sampling with fixed Ratio,
** The number of levels and number of coarsest level is determined accordingly. */
void setSamplingLevels(MLEigenSolver*& eigenSolver, double levelRatio, int numEigen, Eigen::VectorXi& sampSize)
{
	double	spRatio = 1.25;
	int		sizeSubsp = spRatio * 2 * numEigen;

	vector<int> sampleLevels;
	sampleLevels.reserve(10);
	int SAMPLE_CUT_OFF = max(250, sizeSubsp);			//500
	int SAMPLE_LOW_BOUND = 2 * SAMPLE_CUT_OFF;
	if (problemType == 1) {
		SAMPLE_LOW_BOUND /= 3;
		SAMPLE_CUT_OFF /= 3;
	}
	int numSample = eigenSolver->getV().rows();
	

	/* automated */
	do {
		numSample = (int)round(numSample / levelRatio);
		if (numSample < SAMPLE_CUT_OFF) break;
		sampleLevels.push_back(numSample);
		std::cout << "Inserted sample: " << numSample << std::endl; 
	} while (numSample > SAMPLE_LOW_BOUND);

	
	sampSize.resize(sampleLevels.size());
	for (int ii = 0; ii < sampleLevels.size(); ++ii)
		sampSize(ii) = sampleLevels[ii];
}

#endif // !FAST_SPECTRUM_GUI