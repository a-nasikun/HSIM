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

#include "FastSpectrum.h"

enum BasisType { Coarse_1Level_Basis = 0, MultiGrid_Geom_Basis = 1, MultiGrid_NonSmooth_Basis =2 };

int					basisToShow		= 0;
int					eigToShow		= 0;
int					eigToCompute    = 500;
bool				boolShowSamples = false;
bool				useNonSmoothBasis = false; 
//static int			numOfSample		= 250;
static int			sampleType		= Sample_Farthest_Point; 
static int		 	basisType		= MultiGrid_Geom_Basis;
static int			dataToShow		= 0;
float				tauScale		= 10.0f;
string				model			= "Sphere40k";
static int			SIMtype			= 2;


void fastSpectrumMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, FastSpectrum &fastSpectrum) {
	// MAIN WINDOW 	
	{
		// Variable for Processing
		Eigen::MatrixXd redEigVects, V = viewer.data().V;
		Eigen::VectorXd redEigVals;
		Eigen::MatrixXi F = viewer.data().F;

		// Variables For Viewer visualization
		float	menuWindowLeft		= 0.f,
				menuWindowWidth		= 200.f,
				menuWindowHeight	= 475.f;
		
		Eigen::MatrixXd		vColor;
		Eigen::VectorXd		Z;

		ImGui::SetNextWindowPos(ImVec2(5+menuWindowLeft * menu.menu_scaling(), 5), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Fast Spectrum##2", nullptr, ImGuiWindowFlags_NoSavedSettings);

		float w = ImGui::GetContentRegionAvailWidth();
		float p = ImGui::GetStyle().FramePadding.x;

		// Main Algorithm
		if (ImGui::CollapsingHeader("Executing Algorithm", ImGuiTreeNodeFlags_DefaultOpen)){
			if (ImGui::Button("Load##Mesh", ImVec2((w) / 2.f, 0))){
				// Load mesh + display the correct mesh
				int numOfModel = viewer.data_list.size();
				viewer.open_dialog_load_mesh();
				if (viewer.data_list.size() > numOfModel) {
					viewer.data() = viewer.data_list.at(++dataToShow);
					viewer.data_list.at(dataToShow - 1).clear();
				}

				V.resize(viewer.data().V.rows(), viewer.data().V.cols());
				F.resize(viewer.data().F.rows(), viewer.data().F.cols());
				V = viewer.data().V;
				F = viewer.data().F;	
				fastSpectrum.setV(V);
				fastSpectrum.setF(F);				

				viewer.core.align_camera_center(V, F);

				viewer.data().show_lines = false;
				printf("A new mesh with %d vertices (%d faces).\n", V.rows(), F.rows());
				viewer.data().set_mesh(V, F);
				viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));

				boolShowSamples = false; 
				//for (int i = 0; i < viewer.data_list.size(); i++) {
				//	printf("Data %d: %d (%d) \n", i, viewer.data_list.at(i).V.rows(), dataToShow);
				//}
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0))){
				viewer.open_dialog_save_mesh();
			}			
			
			ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();

			/* STEP BY STEP CONSTRUCTION */
			static int numLevels = 3;
			
			if (ImGui::CollapsingHeader("Program Step by Step", ImGuiTreeNodeFlags_DefaultOpen)) {
				///static char numLvls[10] = "3"; ImGui::InputText("# Levels", numLvls, 10);
				///numLevels = atoi(numLvls);
				/* Setup Laplace-Beltrami Matrix */
				if (ImGui::Button("[1] Construct Laplacian Matrix", ImVec2(w, 0))) {				
					///fastSpectrum.setInitialSystem(numLevels);
					if (fastSpectrum.V.rows() < 1) return;
					fastSpectrum.constructLaplacianMatrix();
				}
				/* For Hierarchy and Sampling */
				static double levelRatio = 8.0;
				static char lvlRatTxt[10] = "8.0"; ImGui::InputText("Levels ratio", lvlRatTxt, 10);
				levelRatio = atof(lvlRatTxt);

				if (ImGui::Button("[2] Sampling", ImVec2(w, 0))) {					
					if (fastSpectrum.M.rows() != fastSpectrum.V.rows() || fastSpectrum.V.rows() < 1) {
						cout << "Error! Make sure you construct Laplacian matrices before creating samples." << endl;
						return;
					}

					vector<int> sampleLevels; sampleLevels.reserve(10);
					int SAMPLE_LOW_BOUND = 1000;
					int numSample = fastSpectrum.V.rows();
					int lvlCounter = 0;
					do {
						numSample = (int)round(numSample / levelRatio);
						sampleLevels.push_back(numSample);
						lvlCounter++;
					} while (numSample > SAMPLE_LOW_BOUND);
					
					/* Setting the number of levels */
					numLevels = lvlCounter;
					fastSpectrum.setInitialSystem(numLevels);

					Eigen::VectorXd sampSize(sampleLevels.size());
					for (int ii = 0; ii < sampleLevels.size(); ii++)
						sampSize(ii) = sampleLevels[ii];
					
					/* For 1 level (coarsest level) sampling only*/
					fastSpectrum.setSample(sampSize(sampleLevels.size()-1), (SamplingType)sampleType);
					/* For the multilevel sampling */
					fastSpectrum.constructSampleHierarchy(sampSize);
										
					boolShowSamples = true; 
				}

				ImGui::Text("Show sample distance");				
				for (int i = 0; i < fastSpectrum.level; i++)
				{
					string disp = "Dist ";
					disp += to_string(i);
					char const* ca = disp.c_str();					
					if (ImGui::Button(ca, ImVec2(w/(double)fastSpectrum.level-p, 0))) {
						fastSpectrum.visualizeGeodesicDistanceMultLevel(viewer, i);
					} ImGui::SameLine(0, p);
				}				
				ImGui::Spacing(); ImGui::Spacing();

				/* For Basis */
				if (ImGui::Button("[3] Construct Basis", ImVec2(w, 0))) {
					Eigen::SparseMatrix<double> M;
					Eigen::VectorXi samples;
					int numSample;
					double avgEdgeLen;
					fastSpectrum.getMassMatrix(M);
					fastSpectrum.getSamples(samples);
					fastSpectrum.getSampleSize(numSample);
					fastSpectrum.getAvgEdgeLength(avgEdgeLen);
					if (numSample < 1 || V.rows() != M.rows() || samples.maxCoeff() >= V.rows() || !(avgEdgeLen > 0.0)) {
						cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
						return;
					}

					fastSpectrum.constructBasis();
					useNonSmoothBasis = false; 
					fastSpectrum.testConstantFuncsProjection();
				}

				// Showing basis in different levels
				for (int i = 0; i < fastSpectrum.level; i++)
				{
					string disp = "Lvl. "; disp += to_string(i);
					char const* ca = disp.c_str();
					if (ImGui::Button(ca, ImVec2(w / (double)fastSpectrum.level - p, 0))) {
						fastSpectrum.computeTempBasis(i);
						fastSpectrum.visualizeBasis(viewer, basisToShow);
					} 
					if(i<(fastSpectrum.level-1)) ImGui::SameLine(0, p);
				}


				if (ImGui::Button("[3] Coarse Basis only", ImVec2(w, 0))) {
					Eigen::SparseMatrix<double> M;
					Eigen::VectorXi samples;
					int numSample;
					double avgEdgeLen;
					fastSpectrum.getMassMatrix(M);
					fastSpectrum.getSamples(samples);
					fastSpectrum.getSampleSize(numSample);
					fastSpectrum.getAvgEdgeLength(avgEdgeLen);
					if (numSample < 1 || V.rows() != M.rows() || samples.maxCoeff() >= V.rows() || !(avgEdgeLen > 0.0)) {
						cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
						return;
					}

					fastSpectrum.constructBasisCoarseOnly();
					useNonSmoothBasis = false;
				}

				// Non-smooth basis, based on SIGGRAPH-Asia Paper by Xiang (2019)
				if (ImGui::Button("[3] Non Smooth Basis", ImVec2(w, 0))) {
					fastSpectrum.constructNonSmoothBasis();
					fastSpectrum.visualizeClusterOfSamplesLevel0(viewer);
					useNonSmoothBasis = true; 
				}

				for (int i = 0; i < fastSpectrum.level; i++)
				{
					string disp = "Cluster "; disp += to_string(i);
					char const* ca = disp.c_str();
					if (ImGui::Button(ca, ImVec2(w / (double)fastSpectrum.level - p, 0))) {
						if(i==0)
							fastSpectrum.visualizeClusterOfSamplesLevel0(viewer);
						else 
							fastSpectrum.visualizeClusterOfSamplesLevelK(viewer, i);
					} ImGui::SameLine(0, p);
				}

				ImGui::Spacing(); ImGui::Spacing();

				/* For Low-Dim Problem */
				if (ImGui::Button("[4] Construct Low-Dim Eigenproblem", ImVec2(w, 0))) {
					Eigen::SparseMatrix<double> Basis;
					int numSample;
					fastSpectrum.getFunctionBasis(Basis);
					fastSpectrum.getSampleSize(numSample);
					if (Basis.rows()!=V.rows() || !(Basis.cols()>0)) {
						cout << "Error! Make sure you execute previous steps before constructing Low-Dim Eigenproblem." << endl;
						return;
					}

					fastSpectrum.constructRestrictedProblem();
				}
				/* Solving Low-Dim Eigenproblem */
				if (ImGui::Button("[5] Solve Low-Dim Eigenproblem", ImVec2(w, 0))) {
					Eigen::SparseMatrix<double> MBar, Basis;
					fastSpectrum.getReducedMassMatrix(MBar);
					fastSpectrum.getFunctionBasis(Basis);
					if (Basis.cols() != MBar.cols() || MBar.cols()<1) {
						cout << "Error! Make sure you execute previous steps before solving the Low-Dim Eigenproblem." << endl;
						return;
					}

					fastSpectrum.solveRestrictedProblem();
					//cout << "(approx) Eigenvector: \n " << fastSpectrum.reducedEigVals << endl << endl; 

					// Show the first non-zero eigenvectors
					Eigen::SparseMatrix<double>	U;
					fastSpectrum.getFunctionBasis(U);
					fastSpectrum.getReducedEigVects(redEigVects);
					printf("Approximated (lifted) eigenvectors %dx%d\n", U.rows(), redEigVects.cols());
					eigToShow					= 1;
					Z							= U * redEigVects.col(eigToShow);
					igl::jet(Z, true, vColor);
					viewer.data().set_colors(vColor);
				}

				if (ImGui::Button("Save EigVals#Approx", ImVec2(w/2.0, 0))) {
					int toStore = min(250, (int) fastSpectrum.reducedEigVals.rows());
					writeDenseMatrixToFile(fastSpectrum.reducedEigVals.block(0, 0, toStore,1), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_Red_" + to_string(toStore) + ".txt");
				}
				ImGui::SameLine(0, p);
				if (ImGui::Button("Save EigVect#Approx", ImVec2(w/2.0, 0))) {
					int toStore = 250;
					writeDenseMatrixToFile(fastSpectrum.Basis*fastSpectrum.reducedEigVects.block(0,0,fastSpectrum.V.rows(), toStore), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_Red_" + to_string(toStore) + ".txt");
				}
			}
		}
				
		ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();

		/* VISUALIZATION WINDOW */
		if (ImGui::CollapsingHeader("Visualization##FastSpectrum", ImGuiTreeNodeFlags_DefaultOpen)){
			/* For Sampling */
			if (ImGui::Button("Show Samples", ImVec2((w), 0))) {
				//Eigen::VectorXi Sample;
				//fastSpectrum.getSamples(Sample);
				boolShowSamples = !boolShowSamples;

				if (fastSpectrum.Sample.maxCoeff() > V.rows()) {
					cout << "Error! Make sure you compute new samples before viewing them." << endl;
					return;
				}

				if (boolShowSamples) {	
					//for (int i = 0; i < Sample.size(); i++) {
					//	viewer.data().add_points(V.row(Sample(i)), Eigen::RowVector3d(0.7, 0.1, 0.1));
					//}
					
					visualizeSampleMulti(viewer, V, F, fastSpectrum.Sample, fastSpectrum.SubspaceSize);
				}
				else {
					//viewer.data().clear();
					//fastSpectrum.getV(V);
					//fastSpectrum.getF(F);
					//viewer.data().set_mesh(V, F);
					viewer.data().points.resize(0, 6);
				}

				viewer.data().point_size = 10.0f;
			}

			/* For Basis */			
			ImGui::Text("Show Basis ");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##Basis", ImVec2((w - 85.0f - 5.0*p) /4.f, 0))) {
				if (fastSpectrum.Basis.rows() != V.rows()) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}

				if (basisToShow > 0)
					basisToShow--;

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
					//viewer.data().clear();
					//viewer.data().set_mesh(V, F);
				}

				fastSpectrum.visualizeBasis(viewer, basisToShow);
			}
			ImGui::SameLine(0, p);	
			ImGui::Text("[%d]", basisToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##Basis", ImVec2((w - 85.0f - 5.0*p) / 4.f, 0))) {
				if (fastSpectrum.Basis.rows() != V.rows()) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}

				if (basisToShow < fastSpectrum.Basis.cols())
					basisToShow++;

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
					//viewer.data().clear();
					//viewer.data().set_mesh(V, F);
				}

				fastSpectrum.visualizeBasis(viewer, basisToShow);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Rand", ImVec2((w - 85.0f - 5.0*p)*2/4.f, 0))) {
				if (fastSpectrum.Basis.rows() != V.rows()) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}
				
				basisToShow = rand() % fastSpectrum.Basis.cols();

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
					//viewer.data().clear();
					//viewer.data().set_mesh(V, F);
				}

				fastSpectrum.visualizeBasis(viewer, basisToShow);
			}
			/* For Approx. Eigenvectors */
			ImGui::Text("Show EigVec");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##EigVec1", ImVec2((w-120.0f-3*p)/2.f, 0))) {
				fastSpectrum.getReducedEigVects(redEigVects);
				Eigen::SparseMatrix<double> U;
				fastSpectrum.getFunctionBasis(U);

				if (U.rows() != V.rows()) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow > 0)
					eigToShow--;

				Z = U * redEigVects.col(eigToShow);
				if(eigToShow>0)
					igl::jet(Z, true, vColor);
				else 
					igl::jet(Z, false, vColor);		// for zero-th eigenvector, it doesn't make sense to scale, since they're supposed to be constant
				viewer.data().set_colors(vColor);
			}
			ImGui::SameLine(0, p);
			ImGui::Text("[%d]", eigToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##EigVec1", ImVec2((w-120.0f-3*p)/2.f, 0))) {
				fastSpectrum.getReducedEigVects(redEigVects);
				Eigen::SparseMatrix<double> U;
				fastSpectrum.getFunctionBasis(U);
				
				if (U.rows() != V.rows()) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow < redEigVects.cols())
					eigToShow++;

				Z = U * redEigVects.col(eigToShow);
				igl::jet(Z, true, vColor);
				viewer.data().set_colors(vColor);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Red##EigVec1", ImVec2((w - 150.0f - 3 * p) / 3.f, 0))) {
				Z = fastSpectrum.Basis * fastSpectrum.reducedEigVects.col(eigToShow);
				igl::jet(Z, true, vColor);
				viewer.data().set_colors(vColor);
			}

			/* For Reference Eigenvectors */
			ImGui::Text("Show Refer.");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##RefEigVec1", ImVec2((w - 150.0f - 3 * p) / 2.f, 0))) {
				if (fastSpectrum.refEigVects.rows() != V.rows()) {
					cout << "Error! Make sure you have computed the reference eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow > 0)
					eigToShow--;

				Z = fastSpectrum.refEigVects.col(eigToShow);
				if (eigToShow>0)
					igl::jet(Z, true, vColor);
				else
					igl::jet(Z, false, vColor);		// for zero-th eigenvector, it doesn't make sense to scale, since they're supposed to be constant
				viewer.data().set_colors(vColor);
			}
			ImGui::SameLine(0, p);
			ImGui::Text("[%d]", eigToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##RefEigVec1", ImVec2((w - 150.0f - 3 * p) / 2.f, 0))) {
				if (fastSpectrum.refEigVects.rows() != V.rows()) {
					cout << "Error! Make sure you have computed the reference eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow < fastSpectrum.refEigVects.cols())
					eigToShow++;

				Z = fastSpectrum.refEigVects.col(eigToShow);
				igl::jet(Z, true, vColor);
				viewer.data().set_colors(vColor);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Ref##EigVec1", ImVec2((w - 150.0f - 3 * p) / 3.f, 0))) {
				Z = fastSpectrum.refEigVects.col(eigToShow);
				igl::jet(Z, true, vColor);
				viewer.data().set_colors(vColor);
			}

			/* Overlays */
			ImGui::Checkbox("Wireframe", &(viewer.data().show_lines));
			ImGui::Checkbox("Fill Faces", &(viewer.data().show_faces));
		}

		ImGui::End();
	}
}

void subspaceMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, FastSpectrum &fastSpectrum)
{
	// Variable for Processing
	Eigen::MatrixXd V = viewer.data().V;
	Eigen::MatrixXi F = viewer.data().F;

	// Variables For Viewer visualization
	float	menuWindowLeft		= 375.f,
			menuWindowWidth		= 200.f,
			menuWindowHeight	= 475.f;

	ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5), ImGuiSetCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight), ImGuiSetCond_FirstUseEver);
	ImGui::Begin("Subspace##Menu", nullptr, ImGuiWindowFlags_NoSavedSettings);

	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

	if (ImGui::CollapsingHeader("Draft on subspace iteration", ImGuiTreeNodeFlags_DefaultOpen)) {
		/* some parameters */
		static int numOfModes = 50;
		static int maxIter = 50;
		static double tol = 1e-4;
		/* Type of the SIM to be computed*/
		std::map<int, SIM_Type>			typeMap;
		typeMap.insert(std::pair<int, SIM_Type>(0, SIM_Type::SIM_MultiLevel));
		typeMap.insert(std::pair<int, SIM_Type>(1, SIM_Type::SIM_1level_WarmStart));
		typeMap.insert(std::pair<int, SIM_Type>(2, SIM_Type::SIM_1level_Random));
		typeMap.insert(std::pair<int, SIM_Type>(3, SIM_Type::SIM_1level_Bathe));


		if (ImGui::CollapsingHeader("SIM Option", ImGuiTreeNodeFlags_CollapsingHeader)) {
			static char numModesTxt[10] = "50"; ImGui::InputText("Num Eigs", numModesTxt, 10);
			numOfModes = atoi(numModesTxt);
			static char maxIterTxt[10] = "50"; ImGui::InputText("Max Iter", maxIterTxt, 10);
			maxIter = atoi(maxIterTxt);
			static char tolTxt[10] = "0.0001"; ImGui::InputText("Tolerance", tolTxt, 10);
			tol = atof(tolTxt);

			ImGui::RadioButton("Random", &SIMtype, 2);			ImGui::SameLine();
			ImGui::RadioButton("Bathe", &SIMtype, 3);			//ImGui::SameLine();
			ImGui::RadioButton("SGP Warm-start", &SIMtype, 1);	ImGui::SameLine();
			ImGui::RadioButton("Multi-level", &SIMtype, 0);		//ImGui::SameLine();
		}

		if (ImGui::Button("Simplest Subspace", ImVec2(w, 0))) {
			fastSpectrum.simpleSubspaceIterMethod(viewer, typeMap[SIMtype], tol, numOfModes, maxIter);
		}

		if (ImGui::Button("Scripted experiment (1)", ImVec2(w, 0))) {
			double thisTol = 1e-2;
			vector<int> wantedModes{ 50, 100, 250 };
			for (int j = 0; j < wantedModes.size(); j++) {
				for (int k = 0; k < 4; k++) {
					double tolTol = thisTol / pow(10, k);
					fastSpectrum.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_Random, tolTol, wantedModes[j], maxIter);
					fastSpectrum.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_Bathe,  tolTol, wantedModes[j], maxIter);
					fastSpectrum.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_MultiLevel,    tolTol, wantedModes[j], maxIter);
				}
			}
		}

		if (ImGui::Button("Scripted experiment (2)", ImVec2(w, 0))) {
			double thisTol = 1e-2;
			vector<int> wantedModes{ 50, 100, 250 };
			for (int j = 0; j < wantedModes.size(); j++) {
				for (int k = 0; k < 4; k++) {
					double tolTol = thisTol / pow(10, k);
					fastSpectrum.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_WarmStart, tolTol, wantedModes[j], maxIter);
				}
			}
		}
		//if (ImGui::Button("Block Subspace", ImVec2(w, 0))) {
		//	fastSpectrum.SubspaceIterMethod(viewer);
		//}

		if (ImGui::Button("Save EigVals#RIM", ImVec2(w/2.0, 0))) {
			//writeDenseMatrixToFile(fastSpectrum.SIMeigVals, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_SIM_" + to_string(fastSpectrum.SIMeigVals.rows()) + ".txt");
			writeDenseMatrixToFile(fastSpectrum.SIMeigValsStack, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_SIM_" + to_string(fastSpectrum.SIMeigVals.rows()) + ".txt");
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Save EigVect#RIM", ImVec2(w/2.0, 0))) {
			writeDenseMatrixToFile(fastSpectrum.SIMeigVects, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_SIM_" + to_string(fastSpectrum.SIMeigVects.cols()) + ".txt");
		}

		/* For Subspace Iteration Method Eigenvectors */

		Eigen::MatrixXd vColor;
		ImGui::Text("Show EigVec");
		ImGui::SameLine(0, p);
		if (ImGui::Button("- 1##EigVecSIM", ImVec2((w - 120.0f - 3 * p) / 2.f, 0))) {			

			if (eigToShow > 0)
				eigToShow--;

			
			if (eigToShow>0)
				igl::jet(fastSpectrum.SIMeigVects.col(eigToShow), true, vColor);
			else
				igl::jet(fastSpectrum.SIMeigVects.col(eigToShow), false, vColor);		// for zero-th eigenvector, it doesn't make sense to scale, since they're supposed to be constant
			viewer.data().set_colors(vColor);
		}
		ImGui::SameLine(0, p);
		ImGui::Text("[%d]", eigToShow);
		ImGui::SameLine(0, p);

		if (ImGui::Button("+ 1##EigVecSIM", ImVec2((w - 120.0f - 3 * p) / 2.f, 0))) {	
			if (eigToShow < fastSpectrum.SIMeigVects.cols())
				eigToShow++;
			
			igl::jet(fastSpectrum.SIMeigVects.col(eigToShow), true, vColor);
			viewer.data().set_colors(vColor);
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("SIM##EigVecSIM", ImVec2((w - 50.0f - 3 * p) / 3.f, 0))) {			
			igl::jet(fastSpectrum.SIMeigVects.col(eigToShow), true, vColor);
			viewer.data().set_colors(vColor);
		}

	}

	ImGui::End();
}

void applicationMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, FastSpectrum &fastSpectrum)
{
	{
		// Variable for Processing
		Eigen::MatrixXd V = viewer.data().V;
		Eigen::MatrixXi F = viewer.data().F;

		// Variables For Viewer visualization
		float	menuWindowLeft = 175.f,
			menuWindowWidth = 200.f,
			menuWindowHeight = 475.f;

		ImGui::SetNextWindowPos(ImVec2(5 + menuWindowLeft * menu.menu_scaling(), 5), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(menuWindowWidth, menuWindowHeight), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Application##Menu", nullptr, ImGuiWindowFlags_NoSavedSettings);

		float w = ImGui::GetContentRegionAvailWidth();
		float p = ImGui::GetStyle().FramePadding.x;

		/* HEAT METHOD */
		//if (ImGui::CollapsingHeader("Heat Method", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::CollapsingHeader("Heat Method", ImGuiTreeNodeFlags_OpenOnArrow)) {
			if (ImGui::Button("Solve Heat Method", ImVec2(w, 0))) {
				fastSpectrum.solveHeatMethod(viewer);
			}

			ImGui::RadioButton("Coarse##1-level_Basis_Only", &basisType, Coarse_1Level_Basis); ImGui::SameLine();
			ImGui::RadioButton("MG_Geom##Multigrid_smooth", &basisType, MultiGrid_Geom_Basis); ImGui::SameLine();
			ImGui::RadioButton("NotSmooth##consnt_basis", &basisType, MultiGrid_NonSmooth_Basis);
						

			
			//static char tauScale[6] = "10.0"; 
			//ImGui::InputText("", tauScale, 10);				
			ImGui::InputFloat("tau_scale", &tauScale);			
			//ImGui::SameLine();
			if (ImGui::Button("Reduced Heat Method", ImVec2(w, 0))) {
				if (useNonSmoothBasis) {
					cout << "Solving heat method using NON-smooth basis approach \n";
					fastSpectrum.solveHeatMethodRedMultiGridNonSmooth(viewer);
				}
				else
				{
					cout << "Solving heat method using SMOOTH basis approach \n";
					fastSpectrum.tauScale =tauScale;
					fastSpectrum.solveHeatMethodRed(viewer);
				}
			}
			
			if (ImGui::Button("Diffuse##Heat Diffusion", ImVec2((w) / 3.f, 0))) {
				fastSpectrum.inputHeatRed.setZero(V.rows());
				fastSpectrum.inputHeatRed(0) = 1.0;
				switch (basisType)
				{
				case Coarse_1Level_Basis:
					fastSpectrum.solveHeatEquationRed_CoarseOnly(pow(fastSpectrum.avgEdgeLength, 2), fastSpectrum.inputHeatRed, fastSpectrum.outputHeatRed);
					break;
				case MultiGrid_Geom_Basis:
					fastSpectrum.solveHeatEquationRed(pow(fastSpectrum.avgEdgeLength, 2), fastSpectrum.inputHeatRed, fastSpectrum.outputHeatRed);
					break;
				case MultiGrid_NonSmooth_Basis:

					break;
				default:
					break;
				}
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Norm.##Normalize Gradient", ImVec2((w) / 3.f, 0))) {
				fastSpectrum.gradRed = fastSpectrum.G * fastSpectrum.outputHeatRed;
				fastSpectrum.normalizeGradientRed(fastSpectrum.gradRed);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Poisson##Solving Poisson system", ImVec2((w) / 3.f, 0))) {
				switch (basisType)
				{
				case Coarse_1Level_Basis:
					fastSpectrum.solvePoissonEquationRed_CoarseOnly(fastSpectrum.gradRed, fastSpectrum.heatDistanceRed);
					break;
				case MultiGrid_Geom_Basis:
					fastSpectrum.solvePoissonEquationRed(fastSpectrum.gradRed, fastSpectrum.heatDistanceRed);
					break;
				case MultiGrid_NonSmooth_Basis:

					break;
				default:
					break;
				}
			}


			/* Tsting gauss-seidel solver for solving Poisson system */
			if (ImGui::Button("Gauss-Seidel test", ImVec2((w), 0))) {
				fastSpectrum.testGaussSeidelSolver(viewer);
			}
		
			/* Visualizing the geodesic distance */
			if (ImGui::Button("Ref.##Heat_Method", ImVec2((w) / 2.f, 0))) {
				fastSpectrum.visualizeHeatGeodesics(viewer, fastSpectrum.heatDistance);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Approx##Heat_Method", ImVec2((w) / 2.f, 0))) 
			{
				fastSpectrum.visualizeHeatGeodesics(viewer, fastSpectrum.heatDistanceRed);
			}
			
		} // End of HEAT METHOD

		/* [REFERENCE] Eigenvectors */

		if (ImGui::CollapsingHeader("Compute Ref. Eigenproblem", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("EigVal to Comp."); ImGui::SameLine();
			static char numEig[6] = "100"; ImGui::InputText("", numEig, 6);
			eigToCompute = atoi(numEig);
			if (ImGui::Button("Compute Ref. eigproblem", ImVec2(w, 0))) {
				fastSpectrum.computeRefEigenProblem(eigToCompute, "hello");
			}

			if (ImGui::Button("Read Ref. Eigs", ImVec2(w, 0))) {
				// read from file 
			}
			if (ImGui::Button("Save EigVals#REF", ImVec2(w/2, 0))) {
				writeDenseMatrixToFile(fastSpectrum.refEigVals, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_Ref_"+to_string(fastSpectrum.refEigVals.rows())+".txt");
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Save EigVect#REF", ImVec2(w / 2, 0))) {
				writeDenseMatrixToFile(fastSpectrum.refEigVects, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_Ref_" + to_string(fastSpectrum.refEigVects.cols()) + ".txt");
			}
		}


		ImGui::End();
	}
}

#endif // !FAST_SPECTRUM_GUI

