// Timing of the individual stages of HSIM on one mesh.
// Usage: hsim_benchmark <mesh> [numModes=100] [tolerance=1e-2] [maxIterations=5] [shift=0.25] [repetitions=1]
//                       [verbosity=0 (0 silent, 1 summary, 2 detailed)]
#include <hsim/hsim.h>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

double secondsSince(std::chrono::steady_clock::time_point start)
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main(int argc, char* argv[])
{
	if (argc < 2) {
		std::cerr << "usage: " << argv[0] << " <mesh> [numModes] [tolerance] [maxIterations] [shift] [repetitions]\n";
		return 2;
	}
	try {
		const std::string meshFile = argv[1];
		const int numModes = argc > 2 ? std::atoi(argv[2]) : 100;
		hsim::Options options;
		options.tolerance = argc > 3 ? std::atof(argv[3]) : 1e-2;
		options.maxIterations = argc > 4 ? std::atoi(argv[4]) : 5;
		options.shiftParameter = argc > 5 ? std::atof(argv[5]) : 0.25;
		const int repetitions = argc > 6 ? std::atoi(argv[6]) : 1;
		options.verbosity = static_cast<hsim::Verbosity>(argc > 7 ? std::atoi(argv[7]) : 0);

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		hsim::readMesh(meshFile, V, F);
		std::cout << meshFile << ": " << V.rows() << " vertices, " << numModes << " modes, tol " << options.tolerance
		          << ", maxIter " << options.maxIterations << "\n";

		for (int rep = 0; rep < repetitions; ++rep) {
			hsim::Solver solver(options);
			auto t = std::chrono::steady_clock::now();
			solver.setMesh(V, F);
			const double tOperators = secondsSince(t);

			t = std::chrono::steady_clock::now();
			solver.computeSamples(numModes);
			const double tSamples = secondsSince(t);

			t = std::chrono::steady_clock::now();
			solver.computeProlongation();
			const double tProlongation = secondsSince(t);

			t = std::chrono::steady_clock::now();
			solver.solveCoarseProblem();
			const double tCoarse = secondsSince(t);

			t = std::chrono::steady_clock::now();
			const hsim::Result result = solver.solve(numModes);
			const double tIteration = secondsSince(t);

			std::cout << std::fixed << std::setprecision(3) << "operators " << tOperators << " s | sampling " << tSamples
			          << " s | prolongation " << tProlongation << " s | coarse " << tCoarse << " s | iteration " << tIteration
			          << " s | total " << tOperators + tSamples + tProlongation + tCoarse + tIteration << " s\n";
			std::cout << "  levels:";
			for (int k : result.diagnostics.levelSizes)
				std::cout << ' ' << k;
			std::cout << " | iterations per level:";
			for (const auto& level : result.diagnostics.levels)
				std::cout << ' ' << level.iterations;
			std::cout << std::scientific << std::setprecision(10) << " | lambda_1 = " << result.eigenvalues(1)
			          << ", lambda_last = " << result.eigenvalues(numModes - 1) << "\n";
		}
	}
	catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
	return 0;
}
