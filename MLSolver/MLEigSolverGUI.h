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

#include <Eigen/PardisoSupport>

#include "MLEigSolver.h"

#include "MLEigenSolver.h"
#include "MLSolverElastics.h"
#include "MLSolverFunctions.h"

using namespace std;

enum BasisType { Coarse_1Level_Basis = 0, MultiGrid_Geom_Basis = 1, MultiGrid_NonSmooth_Basis =2 };

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
static int			problemType		= 0;					// type of problem (0=Laplace--Beltrami, 1=Linear Elasticity)
static int			matType			= 0;					// material type (for hessian/linear elasticity)
static int			defAxis = 0;							// axis in selecting the partial eigenproblem
static double		clampRegion;							// percentage of the clampling
static double		eigScale		= 1.005;

void reducedSpaceMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, MLEigenSolver*& eigenSolver) 
{
	// Variables For Viewer visualization
	float	menuWindowLeft		= 0.f,
			menuWindowWidth		= 200.f,
			menuWindowHeight	= 475.f;

	/* Material properties (for linear elasticity) */
	static double	m1lame1 = 1.49977776131565*1e-9,
					m1lame2 = 0.000333358026520483*1e-9,
					m2lame1 = 19.8175675675676*1e-9,
					m2lame2 = 0.875*1e-9;		//maaterial 1 lame 1, etc.

	Eigen::MatrixXd		VColor;
	Eigen::VectorXd		Z;

	ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5), ImGuiSetCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight), ImGuiSetCond_FirstUseEver);
	ImGui::Begin("Reduced Space", nullptr, ImGuiWindowFlags_NoSavedSettings);

	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

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

				viewer.core.align_camera_center(viewer.data().V, viewer.data().F);

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
			

		ImGui::RadioButton("Lap--Beltr.", &problemType, 0); ImGui::SameLine();
		ImGui::RadioButton("Lin. Elasti.", &problemType, 1);

		if (ImGui::Button("Set-up Problem##1", ImVec2((w - p) / 2.f, 0))) {
			if (viewer.data().V.rows() < 1) return;

			delete eigenSolver;
			if (problemType == 0) {
				eigenSolver = new MLSolverFunctions();
			}
			else if (problemType == 1) {
				eigenSolver = new MLSolverElastics();
			}
			eigenSolver->setV(viewer.data().V);
			eigenSolver->setF(viewer.data().F);

			printf("Mesh in eigensolver object: %dx%d \n", eigenSolver->getV().rows(), eigenSolver->getV().cols());
		}

		static int axis = 0;
		if (problemType == 1) {
			if (ImGui::Button("Create Tet-Mesh", ImVec2((w), 0))) {
				if (auto solver = dynamic_cast<MLSolverElastics*>(eigenSolver)) {
					if (solver->getF().cols() == 4) return;
					solver->createTetrahedralMesh();
					solver->computeVolume();/* Set the parameters of the mesh for deformable objects */
					viewer.data().clear();
					viewer.data().set_mesh(solver->getV(), solver->getTF());
				}

				//viewer.selected_data_index = 0;
				viewer.core.align_camera_center(viewer.data().V, viewer.data().F);
				viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
				viewer.data().invert_normals = true;

				printf("mesh that we have  now: %d \n", viewer.data_list.size());
			}

			/* Visualizing the tetrahedral mesh */
			if (ImGui::Button("Pick Axis", ImVec2((w) / 2.0, 0))) {
				axis = (axis + 1) % 3;
				if (axis == 0) cout << "X axis \n";
				else if (axis == 1) cout << "Y axis \n";
				else cout << "Z axis \n";
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Show Tet Mesh", ImVec2((w) / 2.0, 0))) {
				viewer.data().invert_normals = false;
				viewer.data().show_lines = true;

				srand(time(NULL));
				double t = (double)(rand() % 20 + 1) / 20.0;
				//double t = 1.0;
				cout << "The variable is: " << t << endl;
				int id = axis;
				Eigen::VectorXd v;
				if (auto solver = dynamic_cast<MLSolverElastics*>(eigenSolver)) {
					v = solver->getVB().col(id).array() - solver->getVB().col(id).minCoeff();
					v /= v.col(0).maxCoeff();

					vector<int> s;
					s.reserve(v.size());
					for (unsigned int i = 0; i < v.size(); i++) {
						if (v(i) < t) {
							s.push_back(i);
						}
					}

					Eigen::MatrixXd V_temp(s.size() * 4, 3);
					Eigen::MatrixXi F_temp(s.size() * 4, 3);

					for (unsigned i = 0; i < s.size(); ++i)
					{
						V_temp.row(i * 4 + 0) = solver->getV().row(solver->getF()(s[i], 0));
						V_temp.row(i * 4 + 1) = solver->getV().row(solver->getF()(s[i], 1));
						V_temp.row(i * 4 + 2) = solver->getV().row(solver->getF()(s[i], 2));
						V_temp.row(i * 4 + 3) = solver->getV().row(solver->getF()(s[i], 3));

						F_temp.row(i * 4 + 0) << (i * 4) + 0, (i * 4) + 1, (i * 4) + 3;			// CW
						F_temp.row(i * 4 + 1) << (i * 4) + 0, (i * 4) + 2, (i * 4) + 1;
						F_temp.row(i * 4 + 2) << (i * 4) + 3, (i * 4) + 2, (i * 4) + 0;
						F_temp.row(i * 4 + 3) << (i * 4) + 1, (i * 4) + 2, (i * 4) + 3;
					}
					printf("Size of V_Temp; %dx%d | F=%dx%d \n", V_temp.rows(), V_temp.cols(), F_temp.rows(), F_temp.cols());

					viewer.data().clear();
					viewer.data().set_mesh(V_temp, F_temp);
					viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
					viewer.data().set_face_based(true);
				}

				printf("mesh that we have  now: %d \n", viewer.data_list.size());
			}

			ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
		}
	}

		/* STEP BY STEP CONSTRUCTION */
		if (ImGui::CollapsingHeader("Program Step by Step##1", ImGuiTreeNodeFlags_DefaultOpen)) {
			/* Set material propertries*/
			double	m1lame1 = 1.49977776131565*1e-9,
				m1lame2 = 0.000333358026520483*1e-9,
				m2lame1 = 19.8175675675676*1e-9,
				m2lame2 = 0.875*1e-9;		//maaterial 1 lame 1, etc.
			int		numSlices;
			static	Eigen::VectorXd colorVect;
			static bool	isClampedBasis = false;
			static int numLevels;

			if (problemType == 1) {
				if (ImGui::CollapsingHeader("Elements material##1", ImGuiTreeNodeFlags_CollapsingHeader)) {
					/* Radio button to select material type */
					ImGui::RadioButton("Const", &matType, 0); ImGui::SameLine();
					ImGui::RadioButton("2-Mat", &matType, 1); ImGui::SameLine();
					ImGui::RadioButton("Slices", &matType, 2);
					ImGui::RadioButton("Box", &matType, 3); ImGui::SameLine();
					ImGui::RadioButton("n-Mat", &matType, 4); ImGui::SameLine();
					ImGui::RadioButton("Hetero", &matType, 5);
					if (matType == 2) {
						static char numSlicesTxt[10] = "5";						ImGui::InputText("Num of slices", numSlicesTxt, 10);
						numSlices = atoi(numSlicesTxt);
					}

					/* Setting up the lame parameters */
					static char lame1txt[32] = "1.49977776131565";			ImGui::InputText("L1_1", lame1txt, 32);
					static char lame2txt[32] = "0.000333358026520483";		ImGui::InputText("L1_2", lame2txt, 32);
					static char lame3txt[32] = "19.8175675675676";			ImGui::InputText("L2_1", lame3txt, 32);
					static char lame4txt[32] = "0.875";						ImGui::InputText("L2_2", lame4txt, 32);
					m1lame1 = atof(lame1txt)*1e-9;
					m1lame2 = atof(lame2txt)*1e-9;
					m2lame1 = atof(lame3txt)*1e-9;
					m2lame2 = atof(lame4txt)*1e-9;
				}
			}

			/* Setup stiffness Matrix */
			if (ImGui::Button("[1] Construct Stiffness Matrix##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1) return;

				eigenSolver->constructMassMatrix();
				eigenSolver->initiateAdM();

				/* Setting up materials parameters*/
				if (auto solver = dynamic_cast<MLSolverElastics*>(eigenSolver))
				{
					switch (matType)
					{
					case 0:
						cout << "MatType: " << matType << ", 1 constant materials" << endl;
						solver->setLameParametersConstant(m1lame1, m1lame2);
						break;

						/* 2-Material (left-right) */
					case 1:
						cout << "MatType: " << matType << ", 2 materials in halves" << endl;
						solver->setLameParametersSlices(defAxis, 2, m1lame1, m1lame2, m2lame1, m2lame2, colorVect);
						break;

						/* 2-Material, in slices */
					case 2:
						cout << "MatType: " << matType << ", 2 materials in slices" << endl;
						solver->setLameParametersSlices(defAxis, numSlices, m1lame1, m1lame2, m2lame1, m2lame2, colorVect);
						break;

					case 3:
						cout << "MatType: " << matType << ", Outer box is different from inner. " << endl;
						solver->setLameParametersBox(defAxis, clampRegion, m1lame1, m1lame2, m2lame1, m2lame2, colorVect);
						break;

					default:
						break;
					}
					igl::jet(colorVect, false, VColor);
					viewer.data().set_colors(VColor);
				}

				eigenSolver->constructStiffnessMatrix();
			}


			/* For Hierarchy and Sampling */
			static double levelRatio = 4.0;
			static char lvlRatTxt[10] = "4.0"; ImGui::InputText("Levels ratio", lvlRatTxt, 10);
			levelRatio = atof(lvlRatTxt);

			if (ImGui::Button("[2] Sampling##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1) return;

				cout << "> Creating samples... " << endl;
				chrono::high_resolution_clock::time_point	t0, t1;
				chrono::duration<double>					duration;
				t0 = chrono::high_resolution_clock::now();

				vector<int> sampleLevels; sampleLevels.reserve(10);
				int SAMPLE_LOW_BOUND = 1000;
				int SAMPLE_CUT_OFF = 250;
				int numSample = eigenSolver->getV().rows();

				/* hard coded */
				///numLevels = 4;
				///sampleLevels.resize(numLevels);
				///Eigen::VectorXd sampSize(numLevels);
				///sampSize << 64000, 16000, 4000, 1000;
				///ml_eigSolver.setInitialSystem(numLevels);

				/* automated */
				do {
					numSample = (int)round(numSample / levelRatio);
					if (numSample < SAMPLE_CUT_OFF) break;
					sampleLevels.push_back(numSample);
				} while (numSample > SAMPLE_LOW_BOUND);

				/* Setting the number of levels */
				numLevels = sampleLevels.size();
				eigenSolver->setInitialSystem(numLevels);

				Eigen::VectorXd sampSize(sampleLevels.size());
				for (int ii = 0; ii < sampleLevels.size(); ++ii)
					sampSize(ii) = sampleLevels[ii];

				/* For 1 level (coarsest level) sampling only*/
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
			static char numSupTxt[15] = "40.0"; ImGui::InputText("Num of Supports", numSupTxt, 6);
						
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
				//ml_eigSolver.testConstantFuncsProjection();

				t1 = chrono::high_resolution_clock::now();
				duration = t1 - t0;
				cout << "in " << duration.count() << " seconds." << endl;
			}

			///// Showing basis in different levels
			///for (int i = 0; i < eigenSolver->getLevel(); ++i)
			///{
			///	string disp = "Lvl. "; disp += to_string(i);
			///	char const* ca = disp.c_str();
			///	if (ImGui::Button(ca, ImVec2(w / (double)eigenSolver->getLevel() - p, 0))) {
			///		ml_eigSolver.computeTempBasis(i);
			///		ml_eigSolver.visualizeBasis(viewer, basisToShow);
			///	}
			///	if (i<(ml_eigSolver.level - 1)) ImGui::SameLine(0, p);
			///}


			if (ImGui::Button("[3] Coarse Basis only##1", ImVec2(w, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getStiffnessMatrix().rows() < 1 || eigenSolver->getSampleSize() < 1) {
					cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
					return;
				}
				numOfSupport = atof(numSupTxt);
				eigenSolver->setClampBasis(isClampedBasis);

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

			///for (int i = 0; i < ml_eigSolver.level; i++)
			///{
			///	string disp = "Cluster "; disp += to_string(i);
			///	char const* ca = disp.c_str();
			///	if (ImGui::Button(ca, ImVec2(w / (double)ml_eigSolver.level - p, 0))) {
			///		if (i == 0)
			///			ml_eigSolver.visualizeClusterOfSamplesLevel0(viewer);
			///		else
			///			ml_eigSolver.visualizeClusterOfSamplesLevelK(viewer, i);
			///	} ImGui::SameLine(0, p);
			///}

			///
			///ImGui::Spacing(); ImGui::Spacing();
			///

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

				if (auto solver = dynamic_cast<MLSolverElastics*>(eigenSolver))
				{
					///solver->injectRigidModes();
				}

				cout << eigenSolver->getEigenValuesReduced().head(min(20, (int)eigenSolver->getEigenValuesReduced().rows()));
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
			///
			///if (ImGui::Button("Save EigVals#Approx", ImVec2(w / 2.0, 0))) {
			///	int toStore = min(250, (int)ml_eigSolver.reducedEigVals.rows());
			///	writeDenseMatrixToFile(ml_eigSolver.reducedEigVals.block(0, 0, toStore, 1), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_Red_" + to_string(toStore) + ".txt");
			///}
			///ImGui::SameLine(0, p);
			///if (ImGui::Button("Save EigVect#Approx", ImVec2(w / 2.0, 0))) {
			///	int toStore = 250;
			///	writeDenseMatrixToFile(ml_eigSolver.Basis*ml_eigSolver.reducedEigVects.block(0, 0, ml_eigSolver.V.rows(), toStore), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_Red_" + to_string(toStore) + ".txt");
			///}
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
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getEigenValuesRef().rows() < 1) {
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
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getEigenValuesRef().rows() < 1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}


				if (eigToShow < (eigenSolver->getEigenValuesRef().rows() - 1))
					eigToShow++;

				eigenSolver->visualizeEigenvector(viewer, eigToShow, MLEigenSolver::EigenVectorType::EIGEN_REFERENCE, eigScale);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Ref##EigVec1", ImVec2((w-85.0-3*p) / 3.f, 0))) {
				if (eigenSolver->getV().rows() < 1 || eigenSolver->getSampleSize() < 1 || eigenSolver->getEigenValuesRef().rows() < 1) {
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

	static int SIMType = 2;
	static bool isClampedEigen = false; 

	if (ImGui::CollapsingHeader("Eigenproblem with SIM", ImGuiTreeNodeFlags_DefaultOpen)) {
		/* some parameters */
		static int		numOfModes	= 5;
		static int		maxIter		= 5;
		static double	tol			= 1e-4;
		/* Type of the SIM to be computed*/
		std::map<int, MLEigenSolver::SIMType>			typeMap;
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(0, MLEigenSolver::SIMType::SIM_MultiLevel));
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(1, MLEigenSolver::SIMType::SIM_1level_WarmStart));
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(2, MLEigenSolver::SIMType::SIM_1level_Random));
		typeMap.insert(std::pair<int, MLEigenSolver::SIMType>(3, MLEigenSolver::SIMType::SIM_1level_Bathe));

		if (ImGui::CollapsingHeader("SIM Option##1", ImGuiTreeNodeFlags_CollapsingHeader)) {
			static char numModesTxt[10] = "5"; ImGui::InputText("Num Eigs", numModesTxt, 10);			
			static char maxIterTxt[10] = "5"; ImGui::InputText("Max Iter", maxIterTxt, 10);			
			static char tolTxt[10] = "0.0001"; ImGui::InputText("Tolerance", tolTxt, 10);
			numOfModes	= atoi(numModesTxt);
			maxIter		= atoi(maxIterTxt);
			tol			= atof(tolTxt);

			ImGui::RadioButton("Random init",	&SIMType, 2);			ImGui::SameLine();
			ImGui::RadioButton("Bathe init",	&SIMType, 3);			//ImGui::SameLine();
			ImGui::RadioButton("SGP init",		&SIMType, 1);			ImGui::SameLine();
			ImGui::RadioButton("Multi-level",	&SIMType, 0);			//ImGui::SameLine();
		}

		static bool isShiftedSIM = false;
		ImGui::Checkbox("with shift", &(isShiftedSIM));
		if (ImGui::Button("Perform SIM iters.##1", ImVec2(w, 0))) {
			if (eigenSolver->getV().rows() < 1 || eigenSolver->getMassMatrix().rows() < 1 || eigenSolver->getStiffnessMatrix().rows()<1 ) return;
			if ((SIMType == 0 || SIMType == 1) && eigenSolver->getEigenValuesReduced().rows() < 1) return;
						
			eigenSolver->performSubspaceIterationMethod(typeMap[SIMType], numOfModes, maxIter, tol, isShiftedSIM);

			printf("Successfully computing the eigenvalues \n");
			cout << "Eigenvalues: \n" << eigenSolver->getEigenValuesSIM() << endl;
			cout << "EigenVectors: \n" << eigenSolver->getEigenVectorsSIM().topLeftCorner(6, 9) << endl;
		}

		///if (ImGui::Button("Scripted experiment (1)", ImVec2(w, 0))) {
		///	double thisTol = 1e-2;
		///	vector<int> wantedModes{ 50, 100, 250 };
		///	//vector<int> wantedModes{ 250 };
		///	for (int j = 0; j < wantedModes.size(); j++) {
		///		for (int k = 0; k < 4; k++) {
		///			double tolTol = thisTol / pow(10, k);
		///			ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_MultiLevel, isShiftedSIM, tolTol, wantedModes[j], maxIter);
		///			///ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_Random, tolTol, wantedModes[j], maxIter);
		///			///ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_Bathe,  tolTol, wantedModes[j], maxIter);
		///		}
		///	}
		///}
		///
		///if (ImGui::Button("Scripted experiment (2)", ImVec2(w, 0))) {
		///	double thisTol = 1e-2;
		///	vector<int> wantedModes{ 50, 100, 250 };
		///	for (int j = 0; j < wantedModes.size(); j++) {
		///		for (int k = 0; k < 4; k++) {
		///			double tolTol = thisTol / pow(10, k);
		///			ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_WarmStart, isShiftedSIM, tolTol, wantedModes[j], maxIter);
		///		}
		///	}
		///}
		///
		///if (ImGui::Button("Block Subspace", ImVec2(w, 0))) {
		///	//ml_eigSolver.SubspaceIterMethod(viewer);
		///	ml_eigSolver.SubspaceIterMethodShift(viewer, tol, numOfModes, maxIter);
		///}
		///
		///if (ImGui::Button("Check matrix", ImVec2(w, 0))) {
		///	Eigen::SparseMatrix<double> SS = ml_eigSolver.ST;
		///	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> spSolver;
		///	//spSolver.compute(MM);
		///	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldltSolver;
		///	ldltSolver.compute(SS);
		///	cout << "Diag D: \n" << ldltSolver.vectorD() << endl;
		///}
		///
		///if (ImGui::Button("Save EigVals#RIM", ImVec2(w / 2.0, 0))) {
		///	//writeDenseMatrixToFile(ml_eigSolver.SIMeigVals, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_SIM_" + to_string(ml_eigSolver.SIMeigVals.rows()) + ".txt");
		///	writeDenseMatrixToFile(ml_eigSolver.SIMeigValsStack, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_SIM_" + to_string(ml_eigSolver.SIMeigVals.rows()) + ".txt");
		///}
		///ImGui::SameLine(0, p);
		///if (ImGui::Button("Save EigVect#RIM", ImVec2(w / 2.0, 0))) {
		///	writeDenseMatrixToFile(ml_eigSolver.SIMeigVects, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_SIM_" + to_string(ml_eigSolver.SIMeigVects.cols()) + ".txt");
		///}
		///	

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
			cout << "> Computing reference eigenproblem ";

			chrono::high_resolution_clock::time_point	t0, t1;
			chrono::duration<double>					duration;
			t0 = chrono::high_resolution_clock::now();

			if (!isClampedEigen) {
				eigenSolver->computeRefEigenProblem(eigToCompute, solverTypeMap[solverType]);
			}
			else {
				eigenSolver->computeRefEigenProblemWithSelector(eigToCompute, solverTypeMap[solverType], defAxis, clampRegion);
			}

			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << " in " << duration.count() << " seconds." << endl;

			cout << "Ref eigenvalues: \n" << eigenSolver->getEigenValuesRef() << endl;
		}

		if (ImGui::Button("Read Ref. Eigs", ImVec2(w, 0))) {
			// read from file 
		}
		if (ImGui::Button("Save EigVals#REF", ImVec2(w / 2, 0))) {
			writeDenseMatrixToFile(eigenSolver->getEigenValuesRef(), "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_Ref_" + to_string(eigenSolver->getEigenVectorsRef().rows()) + ".txt");
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Save EigVect#REF", ImVec2(w / 2, 0))) {
			writeDenseMatrixToFile(eigenSolver->getEigenVectorsRef(), "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_Ref_" + to_string(eigenSolver->getEigenVectorsRef().rows()) + ".txt");
		}
	}
	ImGui::End();
}





#endif // !FAST_SPECTRUM_GUI

