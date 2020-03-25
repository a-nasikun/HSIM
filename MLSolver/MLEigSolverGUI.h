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

#include "MLEigSolver.h"

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
int					problemType		= 0;					// type of problem (0=Laplace--Beltrami, 1=Linear Elasticity)
int					matType			= 0;					// material type (for hessian/linear elasticity)
int					defAxis = 0;							// axis in selecting the partial eigenproblem
double				clampRegion;

void fastSpectrumMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, MLEigSolver &ml_eigSolver) {
	// MAIN WINDOW 	
	{
		// Variable for Processing
		Eigen::MatrixXd redEigVects, V = viewer.data().V;
		Eigen::VectorXd redEigVals;
		//Eigen::MatrixXi F = viewer.data().F;

		// Variables For Viewer visualization
		float	menuWindowLeft		= 0.f,
				menuWindowWidth		= 200.f,
				menuWindowHeight	= 475.f;

		/* Material properties (for linear elasticity) */
		static double	m1lame1 = 1.49977776131565*1e-9,
						m1lame2 = 0.000333358026520483*1e-9,
						m2lame1 = 19.8175675675676*1e-9,
						m2lame2 = 0.875*1e-9;		//maaterial 1 lame 1, etc.
		
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
					ml_eigSolver.setV(viewer.data().V);
					ml_eigSolver.setF(viewer.data().F);

					viewer.core.align_camera_center(viewer.data().V, viewer.data().F);

					viewer.data().show_lines = false;
					printf("A new mesh with %d vertices (%d faces).\n", viewer.data().V.rows(), viewer.data().F.rows());
					//viewer.data().set_mesh(V, F);
					viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));

					boolShowSamples = false;
					printf("mesh that we have  now: %d \n", viewer.data_list.size());
				}
				//for (int i = 0; i < viewer.data_list.size(); i++) {
				//	printf("Data %d: %d (%d) \n", i, viewer.data_list.at(i).V.rows(), dataToShow);
				//}
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0))){
				viewer.open_dialog_save_mesh();
			}			
			
			ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();

			ImGui::RadioButton("Lap--Belt.", &problemType, 0); ImGui::SameLine();
			ImGui::RadioButton("Lin. Elast.", &problemType, 1);

			if (problemType == 0)		ml_eigSolver.problemType = Problem_Class::Laplace_Beltrami;
			else if (problemType == 1)	ml_eigSolver.problemType = Problem_Class::Linear_Elasticity;


			static int axis = 0;
			if (problemType == 1) {
				if (ImGui::Button("Tetrahedralize", ImVec2((w), 0))) {
					ml_eigSolver.createTetrahedralMesh();
					ml_eigSolver.computeVolume();

					/* Set the parameters of the mesh for deformable objects */
					viewer.data().clear();
					viewer.data().set_mesh(ml_eigSolver.TV, ml_eigSolver.TF);
					viewer.core.align_camera_center(viewer.data().V, viewer.data().F);		
					
					printf("mesh that we have  now: %d \n", viewer.data_list.size());

					//viewer.selected_data_index = 0;
					viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));
					viewer.data().invert_normals = true;
				}

				/* Visualizing the tetrahedral mesh */
				if (ImGui::Button("Select Axis", ImVec2((w) / 2.0, 0))) {
					axis = (axis + 1) % 3;
					if (axis == 0) cout << "X axis \n";
					else if (axis == 1) cout << "Y axis \n";
					else cout << "Z axis \n";
				}
				ImGui::SameLine(0, p);
				if (ImGui::Button("Show Tets", ImVec2((w) / 2.0, 0))) {
					viewer.data().invert_normals	= false;
					viewer.data().show_lines		= true;

					srand(time(NULL));
					double t = (double)(rand() % 20 + 1) / 20.0;
					//double t = 1.0;
					cout << "The variable is: " << t << endl;
					int id = axis;
					Eigen::VectorXd v = ml_eigSolver.TB.col(id).array() - ml_eigSolver.TB.col(id).minCoeff();
					v /= v.col(0).maxCoeff();

					vector<int> s;
					for (unsigned int i = 0; i < v.size(); i++) {
						if (v(i) < t) {
							s.push_back(i);
						}
					}

					Eigen::MatrixXd V_temp(s.size() * 4, 3);
					Eigen::MatrixXi F_temp(s.size() * 4, 3);

					for (unsigned i = 0; i < s.size(); ++i)
					{
						V_temp.row(i * 4 + 0) = ml_eigSolver.TV.row(ml_eigSolver.TT(s[i], 0));
						V_temp.row(i * 4 + 1) = ml_eigSolver.TV.row(ml_eigSolver.TT(s[i], 1));
						V_temp.row(i * 4 + 2) = ml_eigSolver.TV.row(ml_eigSolver.TT(s[i], 2));
						V_temp.row(i * 4 + 3) = ml_eigSolver.TV.row(ml_eigSolver.TT(s[i], 3));

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

					printf("mesh that we have  now: %d \n", viewer.data_list.size());
				}


				/* Set material propertries*/				
				enum class MaterialType { Homogen = 0, PiecewiseConstant_2 = 1, PiecewiseConstant_Multi = 2, PiecewiseLinear = 3 };
				//int matType;
				if (ImGui::CollapsingHeader("Elements material", ImGuiTreeNodeFlags_CollapsingHeader)) {
					/* Radio button to select material type */
					ImGui::RadioButton("Const", &matType, 0); ImGui::SameLine();
					ImGui::RadioButton("2-Mat", &matType, 1); ImGui::SameLine();
					ImGui::RadioButton("n-Mat", &matType, 2); ImGui::SameLine();
					ImGui::RadioButton("Hetero", &matType, 3);

					/* Setting up the lame parameters */
					//static char lame1txt[15] = "1.499777761"; ImGui::InputText("Lame_1", lame1txt, 15);
					//static char lame2txt[15] = "0.000333358"; ImGui::InputText("Lame_2", lame2txt, 15);					
					static char lame1txt[32] = "1.49977776131565";			ImGui::InputText("L1_1", lame1txt, 32);
					static char lame2txt[32] = "0.000333358026520483";		ImGui::InputText("L1_2", lame2txt, 32);
					static char lame3txt[32] = "19.8175675675676";			ImGui::InputText("L2_1", lame3txt, 32);
					static char lame4txt[32] = "0.875";						ImGui::InputText("L2_2", lame4txt, 32);
					m1lame1 = atof(lame1txt)*1e-9;
					m1lame2 = atof(lame2txt)*1e-9;
					m2lame1 = atof(lame3txt)*1e-9;
					m2lame2 = atof(lame4txt)*1e-9;
				}

				ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); 
			}

			/* STEP BY STEP CONSTRUCTION */
			static int numLevels = 3;
			
			if (ImGui::CollapsingHeader("Program Step by Step", ImGuiTreeNodeFlags_DefaultOpen)) {
				///static char numLvls[10] = "3"; ImGui::InputText("# Levels", numLvls, 10);
				///numLevels = atoi(numLvls);
				/* Setup Laplace-Beltrami Matrix */
				if (ImGui::Button("[1] Construct Laplacian Matrix", ImVec2(w, 0))) {				
					///ml_eigSolver.setInitialSystem(numLevels);
					if (ml_eigSolver.V.rows() < 1) return;

					/* For Laplace--Beltrami Eigenproblem */
					if (problemType == 0)
						ml_eigSolver.constructLaplacianMatrix();

					/* For Hessian of Linear-Elasticity Eigenproblem */
					else if (problemType == 1) {
						if (ml_eigSolver.TV.rows() < 1) return;
						ml_eigSolver.constructVolumetricMassMatrix();
						initiateAdM(ml_eigSolver.TV, ml_eigSolver.TT, ml_eigSolver.AdM);

						Eigen::VectorXd vColor;
						Eigen::MatrixXd VColor;
						switch (matType)
						{

						case 0:
							cout << "Material TYpe: " << matType  << " ,i.e., Constant material" << endl;
							ml_eigSolver.constructHessianMatrixForLinearElasticity(m1lame1, m1lame2);
							break;

						case 1:
							cout << "MatType: " << matType << " ,i.e., heterogeneous (left-right) material" << endl;
							//simulation.constructHessianMatrixForLinearElasticityMM(vColor);
							ml_eigSolver.constructHessianMatrixForLinearElasticityMM(m1lame1, m1lame2, m2lame1, m2lame2, vColor);
							igl::jet(vColor, false, VColor);
							viewer.data().set_colors(VColor);
							break;

						default:
							break;
						}

					}
				}
				/* For Hierarchy and Sampling */
				static double levelRatio = 4.0;
				static char lvlRatTxt[10] = "4.0"; ImGui::InputText("Levels ratio", lvlRatTxt, 10);
				levelRatio = atof(lvlRatTxt);

				if (ImGui::Button("[2] Sampling", ImVec2(w, 0))) {					
					//if (ml_eigSolver.M.rows() != ml_eigSolver.V.rows() || ml_eigSolver.V.rows() < 1) {
					//	cout << "Error! Make sure you construct Laplacian matrices before creating samples." << endl;
					//	return;
					//}
					if (problemType == 0) if (ml_eigSolver.M.rows() < 1) return; 
					else if (problemType == 1) if (ml_eigSolver.MT.rows() < 1) return;

					cout << "Creating samples... " << endl;
					chrono::high_resolution_clock::time_point	t0, t1;
					chrono::duration<double>					duration;
					t0 = chrono::high_resolution_clock::now();
					
					vector<int> sampleLevels; sampleLevels.reserve(10);
					int SAMPLE_LOW_BOUND = 1000;
					int numSample = (problemType == 0 ? ml_eigSolver.V.rows() : ml_eigSolver.TV.rows() );					
					
					/* hard coded */
					///numLevels = 4;
					///sampleLevels.resize(numLevels);
					///Eigen::VectorXd sampSize(numLevels);
					///sampSize << 64000, 16000, 4000, 1000;
					///ml_eigSolver.setInitialSystem(numLevels);

					/* automated */
					int lvlCounter = 0;
					do {
						numSample = (int)round(numSample / levelRatio);
						if (numSample < 250) break;
						sampleLevels.push_back(numSample);
						lvlCounter++;
					} while (numSample > SAMPLE_LOW_BOUND);
					
					/* Setting the number of levels */
					numLevels = lvlCounter;
					ml_eigSolver.setInitialSystem(numLevels);
					
					Eigen::VectorXd sampSize(sampleLevels.size());
					for (int ii = 0; ii < sampleLevels.size(); ii++)
						sampSize(ii) = sampleLevels[ii];
										
					/* For 1 level (coarsest level) sampling only*/
					ml_eigSolver.setSample(sampSize(numLevels-1), (SamplingType)sampleType);
					/* For the multilevel sampling */
					ml_eigSolver.constructSampleHierarchy(sampSize);									
					
					
					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "in " << duration.count() << " seconds." << endl;
				}

				ImGui::Text("Show sample distance");				
				for (int i = 0; i < ml_eigSolver.level; ++i)
				{
					string disp = "Dist ";
					disp += to_string(i);
					char const* ca = disp.c_str();					
					if (ImGui::Button(ca, ImVec2(w/(double)ml_eigSolver.level-p, 0))) {
						ml_eigSolver.visualizeGeodesicDistanceMultLevel(viewer, i);
					} ImGui::SameLine(0, p);
				}				
				ImGui::Spacing(); ImGui::Spacing();

				/* For Basis */
				ImGui::Checkbox("Clamp Basis", &(ml_eigSolver.isClampedBasis));
				static double numOfSupport;
				static char numSup[15] = "40.0"; ImGui::InputText("Num of Supports", numSup, 6);
				static char selScale[10] = "0.01"; ImGui::InputText("Selection", selScale, 10);
				
				
				if (ImGui::Button("[3] Construct Basis", ImVec2(w, 0))) {					
					if (ml_eigSolver.Sample.size()<1) {
						cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
						return;
					}

					cout << "Constructing basis..." << endl;
					chrono::high_resolution_clock::time_point	t0, t1;
					chrono::duration<double>					duration;
					t0 = chrono::high_resolution_clock::now();

					numOfSupport = atof(numSup);
					clampRegion = atof(selScale);

					if (ml_eigSolver.isClampedBasis)
						ml_eigSolver.constructBasisWithClamping(defAxis, clampRegion, numOfSupport);
					else 
						ml_eigSolver.constructBasis();

					useNonSmoothBasis = false; 
					//ml_eigSolver.testConstantFuncsProjection();

					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "in " << duration.count() << " seconds." << endl;
				}

				// Showing basis in different levels
				for (int i = 0; i < ml_eigSolver.level; i++)
				{
					string disp = "Lvl. "; disp += to_string(i);
					char const* ca = disp.c_str();
					if (ImGui::Button(ca, ImVec2(w / (double)ml_eigSolver.level - p, 0))) {
						ml_eigSolver.computeTempBasis(i);
						ml_eigSolver.visualizeBasis(viewer, basisToShow);
					} 
					if(i<(ml_eigSolver.level-1)) ImGui::SameLine(0, p);
				}


				if (ImGui::Button("[3] Coarse Basis only", ImVec2(w, 0))) {					
					if (ml_eigSolver.Sample.size() < 1) {
						cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
						return;
					}
					numOfSupport = atof(numSup);
					clampRegion = atof(selScale);

					ml_eigSolver.constructBasisCoarseOnly(defAxis, clampRegion, numOfSupport);
					useNonSmoothBasis = false;
				}

				// Non-smooth basis, based on SIGGRAPH-Asia Paper by Xiang (2019)
				if (ImGui::Button("[3] Non Smooth Basis", ImVec2(w, 0))) {
					if (ml_eigSolver.Sample.size() < 1) {
						cout << "Error! Make sure you execute previous steps before constructing basis functions." << endl;
						return;
					}

					numOfSupport = atof(numSup);
					clampRegion = atof(selScale);
					ml_eigSolver.constructNonSmoothBasis(defAxis, clampRegion, numOfSupport);
					ml_eigSolver.visualizeClusterOfSamplesLevel0(viewer);
					useNonSmoothBasis = true; 
				}

				for (int i = 0; i < ml_eigSolver.level; i++)
				{
					string disp = "Cluster "; disp += to_string(i);
					char const* ca = disp.c_str();
					if (ImGui::Button(ca, ImVec2(w / (double)ml_eigSolver.level - p, 0))) {
						if(i==0)
							ml_eigSolver.visualizeClusterOfSamplesLevel0(viewer);
						else 
							ml_eigSolver.visualizeClusterOfSamplesLevelK(viewer, i);
					} ImGui::SameLine(0, p);
				}

				ImGui::Spacing(); ImGui::Spacing();
				
				/* Solving Low-Dim Eigenproblem */
				if (ImGui::Button("[4] Solve Low-Dim Eigenproblem", ImVec2(w, 0))) {					
					if (ml_eigSolver.Basis.cols()<1 || ml_eigSolver.Sample.size()<1) {
						cout << "Error! Make sure you execute previous steps before solving the Low-Dim Eigenproblem." << endl;
						return;
					}

					cout << "Computing the restricted eigen problem" << endl;
					chrono::high_resolution_clock::time_point	t0, t1;
					chrono::duration<double>					duration;
					t0 = chrono::high_resolution_clock::now();

					ml_eigSolver.constructRestrictedProblem();
					ml_eigSolver.solveRestrictedProblem();

					t1 = chrono::high_resolution_clock::now();
					duration = t1 - t0;
					cout << "in " << duration.count() << " seconds." << endl;


					//cout << "(approx) Eigenvector: \n " << ml_eigSolver.reducedEigVals << endl << endl; 

					// Show the first non-zero eigenvectors
					eigToShow = 1;
					Z = ml_eigSolver.Basis * ml_eigSolver.reducedEigVects.col(eigToShow);
					igl::jet(Z, true, vColor);
					viewer.data().set_colors(vColor);
				}

				if (ImGui::Button("Save EigVals#Approx", ImVec2(w/2.0, 0))) {
					int toStore = min(250, (int) ml_eigSolver.reducedEigVals.rows());
					writeDenseMatrixToFile(ml_eigSolver.reducedEigVals.block(0, 0, toStore,1), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_Red_" + to_string(toStore) + ".txt");
				}
				ImGui::SameLine(0, p);
				if (ImGui::Button("Save EigVect#Approx", ImVec2(w/2.0, 0))) {
					int toStore = 250;
					writeDenseMatrixToFile(ml_eigSolver.Basis*ml_eigSolver.reducedEigVects.block(0,0,ml_eigSolver.V.rows(), toStore), "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_Red_" + to_string(toStore) + ".txt");
				}
			}
		}
				
		ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();

		/* VISUALIZATION WINDOW */
		if (ImGui::CollapsingHeader("Visualization##MLEigSolver", ImGuiTreeNodeFlags_DefaultOpen)){
			/* For Sampling */
			if (ImGui::Button("Show Samples", ImVec2((w), 0))) {
				//Eigen::VectorXi Sample;
				//ml_eigSolver.getSamples(Sample);
				boolShowSamples = !boolShowSamples;

				if (ml_eigSolver.Sample.size()<1) {
					cout << "Error! Make sure you compute new samples before viewing them." << endl;
					return;
				}

				if (boolShowSamples) {
					/* Vertex-based sampling on surface data*/
					if(problemType==0)
						visualizeSampleMulti(viewer, ml_eigSolver.V, ml_eigSolver.F, ml_eigSolver.Sample, ml_eigSolver.SubspaceSize);
					/* Vertex-based sampling on volumetric data*/
					else if (problemType == 1)
						visualizeSampleMulti(viewer, ml_eigSolver.TV, ml_eigSolver.TT, ml_eigSolver.Sample, ml_eigSolver.SubspaceSize);
				}
				else {
					viewer.data().points.resize(0, 6);
				}

				viewer.data().point_size = 10.0f;
			}

			/* For Basis */			
			ImGui::Text("Show Basis ");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##Basis", ImVec2((w - 85.0f - 5.0*p) /4.f, 0))) {
				if (ml_eigSolver.Basis.rows()<1) {
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

				ml_eigSolver.visualizeBasis(viewer, basisToShow);
			}
			ImGui::SameLine(0, p);	
			ImGui::Text("[%d]", basisToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##Basis", ImVec2((w - 85.0f - 5.0*p) / 4.f, 0))) {
				if (ml_eigSolver.Basis.rows() < 1) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}

				if (basisToShow < ml_eigSolver.Basis.cols())
					basisToShow++;

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
					//viewer.data().clear();
					//viewer.data().set_mesh(V, F);
				}

				ml_eigSolver.visualizeBasis(viewer, basisToShow);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Rand", ImVec2((w - 85.0f - 5.0*p)*2/4.f, 0))) {
				if (ml_eigSolver.Basis.rows() < 1) {
					cout << "Error! Make sure you have computed basis function on the current model." << endl;
					return;
				}
				
				srand(time(NULL));
				basisToShow = rand() % ml_eigSolver.Basis.cols();

				if (boolShowSamples) {
					boolShowSamples = false;
					viewer.data().points.resize(0, 6);
					//viewer.data().clear();
					//viewer.data().set_mesh(V, F);
				}

				ml_eigSolver.visualizeBasis(viewer, basisToShow);
			}
			/* For Approx. Eigenvectors */
			ImGui::Text("Show EigVec");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##EigVec1", ImVec2((w-120.0f-3*p)/2.f, 0))) {		
				if (ml_eigSolver.reducedEigVals.rows()<1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow > 0)
					eigToShow--;

				///if(ml_eigSolver.isClampedBasis)
				///	Z = ml_eigSolver.SelMatrix * ml_eigSolver.Basis * ml_eigSolver.reducedEigVects.col(eigToShow);
				///else 
					Z = ml_eigSolver.Basis * ml_eigSolver.reducedEigVects.col(eigToShow);

				if(eigToShow>0)
					igl::jet(Z, true, vColor);
				else 
					igl::jet(Z, false, vColor);		// for zero-th eigenvector, it doesn't make sense to scale, since they're supposed to be constant

				printf("Color size: %d | V size: %d \n", vColor.rows(), viewer.data().V.rows());

				viewer.data().set_colors(vColor);
			}
			ImGui::SameLine(0, p);
			ImGui::Text("[%d]", eigToShow);
			ImGui::SameLine(0, p);

			if (ImGui::Button("+ 1##EigVec1", ImVec2((w-120.0f-3*p)/2.f, 0))) {				
				if (ml_eigSolver.reducedEigVals.rows()<1) {
					cout << "Error! Make sure you have computed the eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow < ml_eigSolver.reducedEigVals.rows())
					eigToShow++;

				///if (ml_eigSolver.isClampedBasis)
				///	Z = ml_eigSolver.SelMatrix * ml_eigSolver.Basis * ml_eigSolver.reducedEigVects.col(eigToShow);
				///else
					Z = ml_eigSolver.Basis * ml_eigSolver.reducedEigVects.col(eigToShow);

				printf("Color size: %d | V size: %d \n", vColor.rows(), viewer.data().V.rows());

				igl::jet(Z, true, vColor);
				viewer.data().set_colors(vColor);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Red##EigVec1", ImVec2((w - 150.0f - 3 * p) / 3.f, 0))) {
				///if (ml_eigSolver.isClampedBasis)
				///	Z = ml_eigSolver.SelMatrix * ml_eigSolver.Basis * ml_eigSolver.reducedEigVects.col(eigToShow);
				///else
					Z = ml_eigSolver.Basis * ml_eigSolver.reducedEigVects.col(eigToShow);
				igl::jet(Z, true, vColor);
				viewer.data().set_colors(vColor);
			}

			/* For Reference Eigenvectors */
			ImGui::Text("Show Refer.");
			ImGui::SameLine(0, p);
			if (ImGui::Button("- 1##RefEigVec1", ImVec2((w - 150.0f - 3 * p) / 2.f, 0))) {
				if (ml_eigSolver.refEigVects.cols() <1) {
					cout << "Error! Make sure you have computed the reference eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow > 0)
					eigToShow--;

				Z = ml_eigSolver.refEigVects.col(eigToShow);
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
				if (ml_eigSolver.refEigVects.cols() <1) {
					cout << "Error! Make sure you have computed the reference eigenfunctions on the current model." << endl;
					return;
				}

				if (eigToShow < ml_eigSolver.refEigVects.cols())
					eigToShow++;

				Z = ml_eigSolver.refEigVects.col(eigToShow);
				igl::jet(Z, true, vColor);
				viewer.data().set_colors(vColor);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Ref##EigVec1", ImVec2((w - 150.0f - 3 * p) / 3.f, 0))) {
				Z = ml_eigSolver.refEigVects.col(eigToShow);
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

void subspaceMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, MLEigSolver &ml_eigSolver)
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
		static int numOfModes = 5;
		static int maxIter = 5;
		static double tol = 1e-4;
		/* Type of the SIM to be computed*/
		std::map<int, SIM_Type>			typeMap;
		typeMap.insert(std::pair<int, SIM_Type>(0, SIM_Type::SIM_MultiLevel));
		typeMap.insert(std::pair<int, SIM_Type>(1, SIM_Type::SIM_1level_WarmStart));
		typeMap.insert(std::pair<int, SIM_Type>(2, SIM_Type::SIM_1level_Random));
		typeMap.insert(std::pair<int, SIM_Type>(3, SIM_Type::SIM_1level_Bathe));


		if (ImGui::CollapsingHeader("SIM Option", ImGuiTreeNodeFlags_CollapsingHeader)) {
			static char numModesTxt[10] = "5"; ImGui::InputText("Num Eigs", numModesTxt, 10);
			numOfModes = atoi(numModesTxt);
			static char maxIterTxt[10] = "5"; ImGui::InputText("Max Iter", maxIterTxt, 10);
			maxIter = atoi(maxIterTxt);
			static char tolTxt[10] = "0.0001"; ImGui::InputText("Tolerance", tolTxt, 10);
			tol = atof(tolTxt);

			ImGui::RadioButton("Random", &SIMtype, 2);			ImGui::SameLine();
			ImGui::RadioButton("Bathe", &SIMtype, 3);			//ImGui::SameLine();
			ImGui::RadioButton("SGP Warm-start", &SIMtype, 1);	ImGui::SameLine();
			ImGui::RadioButton("Multi-level", &SIMtype, 0);		//ImGui::SameLine();
		}

		if (ImGui::Button("Simplest Subspace", ImVec2(w, 0))) {
			if ((SIMtype == 0 || SIMtype == 1) && ml_eigSolver.reducedEigVals.rows() < 1) return;

			ml_eigSolver.simpleSubspaceIterMethod(viewer, typeMap[SIMtype], tol, numOfModes, maxIter);
			printf("Successfully computing the eigenvalues \n");
			cout << "Eigenvalues: \n" << ml_eigSolver.SIMeigVals << endl; 
			cout << "EigenVectors: \n" << ml_eigSolver.SIMeigVects.topRows(5) << endl; 
		}

		if (ImGui::Button("Scripted experiment (1)", ImVec2(w, 0))) {
			double thisTol = 1e-2;
			vector<int> wantedModes{ 50, 100, 250 };
			//vector<int> wantedModes{ 250 };
			for (int j = 0; j < wantedModes.size(); j++) {
				for (int k = 0; k < 4; k++) {
					double tolTol = thisTol / pow(10, k);
					ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_MultiLevel,    tolTol, wantedModes[j], maxIter);
					///ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_Random, tolTol, wantedModes[j], maxIter);
					///ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_Bathe,  tolTol, wantedModes[j], maxIter);
				}
			}
		}

		if (ImGui::Button("Scripted experiment (2)", ImVec2(w, 0))) {
			double thisTol = 1e-2;
			vector<int> wantedModes{ 50, 100, 250 };
			for (int j = 0; j < wantedModes.size(); j++) {
				for (int k = 0; k < 4; k++) {
					double tolTol = thisTol / pow(10, k);
					ml_eigSolver.simpleSubspaceIterMethod(viewer, SIM_Type::SIM_1level_WarmStart, tolTol, wantedModes[j], maxIter);
				}
			}
		}	
		
		if (ImGui::Button("Block Subspace", ImVec2(w, 0))) {
			//ml_eigSolver.SubspaceIterMethod(viewer);
			ml_eigSolver.SubspaceIterMethodShift(viewer, tol, numOfModes, maxIter);
		}

		if (ImGui::Button("Save EigVals#RIM", ImVec2(w/2.0, 0))) {
			//writeDenseMatrixToFile(ml_eigSolver.SIMeigVals, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_SIM_" + to_string(ml_eigSolver.SIMeigVals.rows()) + ".txt");
			writeDenseMatrixToFile(ml_eigSolver.SIMeigValsStack, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_SIM_" + to_string(ml_eigSolver.SIMeigVals.rows()) + ".txt");
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Save EigVect#RIM", ImVec2(w/2.0, 0))) {
			writeDenseMatrixToFile(ml_eigSolver.SIMeigVects, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_SIM_" + to_string(ml_eigSolver.SIMeigVects.cols()) + ".txt");
		}

		/* For Subspace Iteration Method Eigenvectors */

		Eigen::MatrixXd vColor;
		ImGui::Text("Show EigVec");
		ImGui::SameLine(0, p);
		if (ImGui::Button("- 1##EigVecSIM", ImVec2((w - 120.0f - 3 * p) / 2.f, 0))) {			

			if (eigToShow > 0)
				eigToShow--;

			
			if (eigToShow>0)
				igl::jet(ml_eigSolver.SIMeigVects.col(eigToShow), true, vColor);
			else
				igl::jet(ml_eigSolver.SIMeigVects.col(eigToShow), false, vColor);		// for zero-th eigenvector, it doesn't make sense to scale, since they're supposed to be constant
			viewer.data().set_colors(vColor);
		}
		ImGui::SameLine(0, p);
		ImGui::Text("[%d]", eigToShow);
		ImGui::SameLine(0, p);

		if (ImGui::Button("+ 1##EigVecSIM", ImVec2((w - 120.0f - 3 * p) / 2.f, 0))) {	
			if (eigToShow < ml_eigSolver.SIMeigVects.cols())
				eigToShow++;
			
			igl::jet(ml_eigSolver.SIMeigVects.col(eigToShow), true, vColor);
			viewer.data().set_colors(vColor);
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("SIM##EigVecSIM", ImVec2((w - 50.0f - 3 * p) / 3.f, 0))) {			
			igl::jet(ml_eigSolver.SIMeigVects.col(eigToShow), true, vColor);
			viewer.data().set_colors(vColor);
		}

	}

	ImGui::End();
}

void applicationMenu(igl::opengl::glfw::Viewer &viewer, igl::opengl::glfw::imgui::ImGuiMenu &menu, MLEigSolver &ml_eigSolver)
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
				ml_eigSolver.solveHeatMethod(viewer);
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
					ml_eigSolver.solveHeatMethodRedMultiGridNonSmooth(viewer);
				}
				else
				{
					cout << "Solving heat method using SMOOTH basis approach \n";
					ml_eigSolver.tauScale =tauScale;
					ml_eigSolver.solveHeatMethodRed(viewer);
				}
			}
			
			if (ImGui::Button("Diffuse##Heat Diffusion", ImVec2((w) / 3.f, 0))) {
				ml_eigSolver.inputHeatRed.setZero(V.rows());
				ml_eigSolver.inputHeatRed(0) = 1.0;
				switch (basisType)
				{
				case Coarse_1Level_Basis:
					ml_eigSolver.solveHeatEquationRed_CoarseOnly(pow(ml_eigSolver.avgEdgeLength, 2), ml_eigSolver.inputHeatRed, ml_eigSolver.outputHeatRed);
					break;
				case MultiGrid_Geom_Basis:
					ml_eigSolver.solveHeatEquationRed(pow(ml_eigSolver.avgEdgeLength, 2), ml_eigSolver.inputHeatRed, ml_eigSolver.outputHeatRed);
					break;
				case MultiGrid_NonSmooth_Basis:

					break;
				default:
					break;
				}
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Norm.##Normalize Gradient", ImVec2((w) / 3.f, 0))) {
				ml_eigSolver.gradRed = ml_eigSolver.G * ml_eigSolver.outputHeatRed;
				ml_eigSolver.normalizeGradientRed(ml_eigSolver.gradRed);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Poisson##Solving Poisson system", ImVec2((w) / 3.f, 0))) {
				switch (basisType)
				{
				case Coarse_1Level_Basis:
					ml_eigSolver.solvePoissonEquationRed_CoarseOnly(ml_eigSolver.gradRed, ml_eigSolver.heatDistanceRed);
					break;
				case MultiGrid_Geom_Basis:
					ml_eigSolver.solvePoissonEquationRed(ml_eigSolver.gradRed, ml_eigSolver.heatDistanceRed);
					break;
				case MultiGrid_NonSmooth_Basis:

					break;
				default:
					break;
				}
			}


			/* Tsting gauss-seidel solver for solving Poisson system */
			if (ImGui::Button("Gauss-Seidel test", ImVec2((w), 0))) {
				ml_eigSolver.testGaussSeidelSolver(viewer);
			}
		
			/* Visualizing the geodesic distance */
			if (ImGui::Button("Ref.##Heat_Method", ImVec2((w) / 2.f, 0))) {
				ml_eigSolver.visualizeHeatGeodesics(viewer, ml_eigSolver.heatDistance);
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Approx##Heat_Method", ImVec2((w) / 2.f, 0))) 
			{
				ml_eigSolver.visualizeHeatGeodesics(viewer, ml_eigSolver.heatDistanceRed);
			}
			
		} // End of HEAT METHOD

		/* [REFERENCE] Eigenvectors */
		static int solverType = 0; // 0: Matlab, 1: Spectra
		if (ImGui::CollapsingHeader("Compute Ref. Eigenproblem", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("EigVal to Comp."); ImGui::SameLine();
			static char numEig[6] = "100"; ImGui::InputText("", numEig, 6);			

			ImGui::RadioButton("Matlab", &solverType, 0); ImGui::SameLine();
			ImGui::RadioButton("Spectra", &solverType, 1);			

			if (ImGui::Button("Compute Reference", ImVec2(w, 0))) {
				eigToCompute = atoi(numEig);

				cout << "> Computing reference eigenproblem ";

				chrono::high_resolution_clock::time_point	t0, t1;
				chrono::duration<double>					duration;
				t0 = chrono::high_resolution_clock::now();

				/* Solving in Matlab */
				if (solverType == 0) {
					cout << "in Matlab... ";
					string fileLoc = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/CDragon_" + to_string(eigToCompute);
					
					if (problemType == 0)
						computeEigenMatlab(ml_eigSolver.S, ml_eigSolver.M, eigToCompute, ml_eigSolver.refEigVects, ml_eigSolver.refEigVals, fileLoc);
					else if (problemType == 1)
						computeEigenMatlab(ml_eigSolver.ST, ml_eigSolver.MT, eigToCompute, ml_eigSolver.refEigVects, ml_eigSolver.refEigVals, fileLoc);
					
				}
				/* Solving in Spectra */
				else if (solverType == 1) {
					cout << "in Spectra... ";
					if (problemType == 0)
						computeEigenSpectraSparse(ml_eigSolver.S, ml_eigSolver.M, eigToCompute, ml_eigSolver.refEigVects, ml_eigSolver.refEigVals);
					else if (problemType == 1)
						computeEigenSpectraSparse(ml_eigSolver.ST, ml_eigSolver.MT, eigToCompute, ml_eigSolver.refEigVects, ml_eigSolver.refEigVals);

				}
				t1 = chrono::high_resolution_clock::now();
				duration = t1 - t0;
				cout << " in " << duration.count() << " seconds." << endl;

				cout << "Ref eigenvalues: \n" <<  ml_eigSolver.refEigVals << endl; 

			}

			ImGui::RadioButton("X", &defAxis, 0); ImGui::SameLine();
			ImGui::RadioButton("Y", &defAxis, 1); ImGui::SameLine();
			ImGui::RadioButton("Z", &defAxis, 2); ImGui::SameLine();
			static char selScale[10] = "0.01"; ImGui::InputText("Selection", selScale, 10);
			clampRegion = atof(selScale);


			if (ImGui::Button("Compute Ref with selector", ImVec2(w, 0))) {
				eigToCompute = atoi(numEig);

				ml_eigSolver.computeSelectedRefEigenProblem(defAxis, clampRegion, eigToCompute, "");
			}

			if (ImGui::Button("Read Ref. Eigs", ImVec2(w, 0))) {
				// read from file 
			}
			if (ImGui::Button("Save EigVals#REF", ImVec2(w/2, 0))) {
				writeDenseMatrixToFile(ml_eigSolver.refEigVals, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVals_Ref_"+to_string(ml_eigSolver.refEigVals.rows())+".txt");
			}
			ImGui::SameLine(0, p);
			if (ImGui::Button("Save EigVect#REF", ImVec2(w / 2, 0))) {
				writeDenseMatrixToFile(ml_eigSolver.refEigVects, "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + model + "_EigVects_Ref_" + to_string(ml_eigSolver.refEigVects.cols()) + ".txt");
			}
		}


		ImGui::End();
	}
}

#endif // !FAST_SPECTRUM_GUI

