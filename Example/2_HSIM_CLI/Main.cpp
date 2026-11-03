// Command-line front end of HSIM: computes eigenpairs of a mesh and writes them together with a complete record of the
// run (parameters, library version, hierarchy, iterations, timings), so that every number can be reproduced.
//
// Usage: hsim_cli <mesh> [options]      (hsim_cli --help lists the options)
#include <hsim/hsim.h>

#include <chrono>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
	std::string mesh;
	int numModes = 100;
	hsim::Options options;
	double normalizeDiameter = 0.0; ///< 0: use the mesh as stored
	std::string output = "hsim";
	bool saveVectors = false;
};

void printUsage(const char* program)
{
	const hsim::Options d;
	std::cout << "Usage: " << program << " <mesh> [options]\n\n"
	          << "Computes the smallest eigenpairs of the Laplace--Beltrami operator with HSIM " << hsim::versionString()
	          << ".\n\n"
	          << "Options:\n"
	          << "  -n, --modes N        number of eigenpairs (default 100)\n"
	          << "  --tol T              relative residual tolerance (default " << d.tolerance << ")\n"
	          << "  --max-iter K         maximal subspace iterations per level (default " << d.maxIterations << ")\n"
	          << "  --shift S            shift parameter in [0, 1) (default " << d.shiftParameter << ")\n"
	          << "  --level-ratio R      ratio of DOFs between levels (default " << d.levelRatio << ")\n"
	          << "  --support A          samples per basis-function support (default " << d.supportSize << ")\n"
	          << "  --threads T          OpenMP threads, 0 = default (default 0)\n"
	          << "  --backend B          dense eigensolver: default, eigen, cusolver\n"
	          << "  --normalize D        scale the mesh to bounding-box diameter D before solving\n"
	          << "  -o, --output PREFIX  output prefix (default 'hsim')\n"
	          << "  --save-vectors       also write the eigenvectors (PREFIX_eigenvectors.txt)\n"
	          << "  -v, --verbose L      progress output: 0 silent, 1 summary, 2 detailed (default 1)\n"
	          << "  -h, --help           show this help\n\n"
	          << "Output: PREFIX_eigenvalues.csv (index, eigenvalue, relative residual) and PREFIX_run.json.\n";
}

Arguments parse(int argc, char* argv[])
{
	Arguments a;
	a.options.verbosity = hsim::Verbosity::Summary;
	std::vector<std::string> args(argv + 1, argv + argc);
	for (size_t i = 0; i < args.size(); ++i) {
		const std::string& key = args[i];
		auto value = [&]() -> const std::string& {
			if (i + 1 >= args.size()) throw std::invalid_argument("missing value for " + key);
			return args[++i];
		};
		if (key == "-h" || key == "--help") {
			printUsage(argv[0]);
			std::exit(0);
		}
		else if (key == "-n" || key == "--modes")
			a.numModes = std::stoi(value());
		else if (key == "--tol")
			a.options.tolerance = std::stod(value());
		else if (key == "--max-iter")
			a.options.maxIterations = std::stoi(value());
		else if (key == "--shift")
			a.options.shiftParameter = std::stod(value());
		else if (key == "--level-ratio")
			a.options.levelRatio = std::stod(value());
		else if (key == "--support")
			a.options.supportSize = std::stod(value());
		else if (key == "--threads")
			a.options.numThreads = std::stoi(value());
		else if (key == "--normalize")
			a.normalizeDiameter = std::stod(value());
		else if (key == "-o" || key == "--output")
			a.output = value();
		else if (key == "--save-vectors")
			a.saveVectors = true;
		else if (key == "-v" || key == "--verbose")
			a.options.verbosity = static_cast<hsim::Verbosity>(std::stoi(value()));
		else if (key == "--backend") {
			const std::string b = value();
			if (b == "default")
				a.options.denseBackend = hsim::DenseEigenBackend::Default;
			else if (b == "eigen")
				a.options.denseBackend = hsim::DenseEigenBackend::Eigen;
			else if (b == "cusolver")
				a.options.denseBackend = hsim::DenseEigenBackend::CuSolver;
			else
				throw std::invalid_argument("unknown backend '" + b + "'");
		}
		else if (!key.empty() && key[0] == '-')
			throw std::invalid_argument("unknown option " + key);
		else if (a.mesh.empty())
			a.mesh = key;
		else
			throw std::invalid_argument("more than one mesh given");
	}
	if (a.mesh.empty()) throw std::invalid_argument("no mesh given (see --help)");
	a.options.validate();
	return a;
}

const char* backendName(hsim::DenseEigenBackend backend)
{
	switch (backend) {
	case hsim::DenseEigenBackend::Eigen:
		return "eigen";
	case hsim::DenseEigenBackend::CuSolver:
		return "cusolver";
	case hsim::DenseEigenBackend::Default:
	default:
		return hsim::builtWithCuda() ? "default (cusolver)" : "default (eigen)";
	}
}

std::string jsonString(const std::string& s)
{
	std::string out = "\"";
	for (const char c : s) {
		if (c == '"' || c == '\\') out += '\\';
		out += c;
	}
	return out + "\"";
}

std::string utcTimestamp()
{
	const std::time_t now = std::time(nullptr);
	char buffer[32];
	std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
	return buffer;
}

void writeRecord(const std::string& path, const Arguments& a, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                 const hsim::Result& result, double totalSeconds)
{
	std::ofstream out(path);
	if (!out) throw std::runtime_error("cannot write " + path);
	const hsim::Options& o = a.options;
	const hsim::Diagnostics& d = result.diagnostics;
	out << std::setprecision(17);
	out << "{\n"
	    << "  \"hsim_version\": " << jsonString(hsim::versionString()) << ",\n"
	    << "  \"date_utc\": " << jsonString(utcTimestamp()) << ",\n"
	    << "  \"mesh\": {\"file\": " << jsonString(a.mesh) << ", \"vertices\": " << V.rows() << ", \"faces\": " << F.rows()
	    << ", \"normalized_diameter\": " << a.normalizeDiameter << "},\n"
	    << "  \"num_modes\": " << a.numModes << ",\n"
	    << "  \"options\": {\"level_ratio\": " << o.levelRatio << ", \"support_size\": " << o.supportSize
	    << ", \"tolerance\": " << o.tolerance << ", \"max_iterations\": " << o.maxIterations
	    << ", \"shift_parameter\": " << o.shiftParameter << ", \"area_factor\": " << o.areaFactor
	    << ", \"dense_backend\": " << jsonString(backendName(o.denseBackend)) << ", \"num_threads\": " << o.numThreads << "},\n"
	    << "  \"level_sizes\": [";
	for (size_t k = 0; k < d.levelSizes.size(); ++k)
		out << (k ? ", " : "") << d.levelSizes[k];
	out << "],\n  \"levels\": [\n";
	for (size_t k = 0; k < d.levels.size(); ++k) {
		const hsim::LevelDiagnostics& l = d.levels[k];
		out << "    {\"level\": " << k << ", \"dimension\": " << l.dimension << ", \"iterations\": " << l.iterations
		    << ", \"shift\": " << l.shift << ", \"seconds\": " << l.seconds << "}" << (k + 1 < d.levels.size() ? ",\n" : "\n");
	}
	out << "  ],\n"
	    << "  \"seconds\": {\"hierarchy\": " << d.hierarchySeconds << ", \"iteration\": " << d.iterationSeconds
	    << ", \"total\": " << totalSeconds << "}\n"
	    << "}\n";
}

} // namespace

int main(int argc, char* argv[])
{
	try {
		const Arguments a = parse(argc, argv);

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		hsim::readMesh(a.mesh, V, F);
		if (a.normalizeDiameter > 0.0) hsim::normalizeMesh(V, a.normalizeDiameter);

		const auto start = std::chrono::steady_clock::now();
		hsim::Solver solver(a.options);
		solver.setMesh(V, F);
		const hsim::Result result = solver.solve(a.numModes);
		const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

		const Eigen::VectorXd residual = hsim::relativeResiduals(result.eigenvectors, result.eigenvalues,
		                                                         solver.operators().stiffness, solver.operators().mass);

		std::ofstream values(a.output + "_eigenvalues.csv");
		if (!values) throw std::runtime_error("cannot write " + a.output + "_eigenvalues.csv");
		values << "index,eigenvalue,relative_residual\n" << std::setprecision(17);
		for (Eigen::Index i = 0; i < result.eigenvalues.size(); ++i)
			values << i << ',' << result.eigenvalues(i) << ',' << residual(i) << '\n';

		if (a.saveVectors) {
			std::ofstream vectors(a.output + "_eigenvectors.txt");
			if (!vectors) throw std::runtime_error("cannot write " + a.output + "_eigenvectors.txt");
			vectors << std::setprecision(17) << result.eigenvectors << '\n';
		}
		writeRecord(a.output + "_run.json", a, V, F, result, totalSeconds);

		std::cout << result.eigenvalues.size() << " eigenpairs in " << totalSeconds << " s; written to " << a.output
		          << "_eigenvalues.csv and " << a.output << "_run.json\n";
	}
	catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
	return 0;
}
