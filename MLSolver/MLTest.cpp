#include "MLTest.h"

void testInit(igl::opengl::glfw::Viewer &viewer, MLEigenSolver*& eigenSolver, const string& meshFile)
{
	/* Set up the solver */
	delete eigenSolver;
	eigenSolver = new MLSolverFunctions();

	/* Loading + Displaying some mesh */
	eigenSolver->readMesh(meshFile);
	viewer.data().set_mesh(eigenSolver->getV(), eigenSolver->getF());
	scaleMesh(viewer.data().V, viewer.data().F, 2.0);
	viewer.core.align_camera_center(viewer.data().V, viewer.data().F);
	viewer.data().show_lines = false;
	viewer.data().set_colors(Eigen::RowVector3d(0.93333333, 0.93333333, 0.9333333));

	eigenSolver->constructMassMatrix();
	eigenSolver->initiateAdM();
	eigenSolver->constructStiffnessMatrix();

}

void runMLSolverTest(igl::opengl::glfw::Viewer &viewer, MLEigenSolver*& eigenSolver)
{
	/* Type of problem*/
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

	int defAxis = 0;
	double clampRegion = 0.0;
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;

	Eigen::VectorXd			sampSize;
	//std::vector<double> levelRatio{4,5,7,8,10,15,20};
	std::vector<double>		levelRatio{ 10 };
	//std::vector<double>		levelSupport{ 5.0, 7.5, 10.0, 12.5, 15.0, 20.0 };
	std::vector<double>		levelSupport{ 7.5 };
	//std::vector<double>	levelZhaoParam{0.1,0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45};
	std::vector<double>		levelZhaoParam{ 0.25 };
	vector<int>				wantedModes{ 50, 100, 250 };
	vector<double>			wantedTols{ 1e-4 };
	int						maxIters = 50;

	for (int j = 0; j < wantedModes.size(); ++j) {
		std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;

		for (int i = 0; i < levelRatio.size(); ++i) {
			std::cout << ">>> LEVEL Ratio of " << levelRatio[i] << std::endl;
			//Sampling
			cout << "> Creating samples... " << endl;
			t0 = chrono::high_resolution_clock::now();
			//eigenSolver->setSamplingLevels(sampSize, wantedModes[j], levelRatio[i]);
			eigenSolver->setSamplingLevels(sampSize, wantedModes[j], levelRatio[i]);
			eigenSolver->setInitialSystem(sampSize.rows());
			eigenSolver->constructSampleHierarchy(sampSize);
			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << "in " << duration.count() << " seconds." << endl;

			for (int l = 0; l < levelSupport.size(); ++l) {
				std::cout << ">>> SUPPORT of " << levelSupport[l] << std::endl;

				// Constructing basis...
				cout << "> Constructing basis... " << endl;
				t0 = chrono::high_resolution_clock::now();
				eigenSolver->setClampBasis(false);
				eigenSolver->constructBasis(levelSupport[l], defAxis, clampRegion);
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
					std::cout << ">>> SHIFTING on " << levelZhaoParam[m] << " percentage." << std::endl;
					eigenSolver->zhaoShiftParam = levelZhaoParam[m];

					std::cout << ">>> WANTED of MODES: " << wantedModes[j] << std::endl;
					for (int k = 0; k < wantedTols.size(); ++k) {
						std::cout << ">>> TOLERANCE of " << wantedTols[k] << std::endl;
						// ML-with shift
						eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], true, shiftMap[3], defAxis, clampRegion);
						// ML-without shift
						eigenSolver->performSubspaceIterationMethod(typeMap[0], wantedModes[j], maxIters, wantedTols[k], false, shiftMap[3], defAxis, clampRegion);
					}
				}
			}
		}
	}
}
