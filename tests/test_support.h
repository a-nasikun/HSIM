// Shared helpers for the HSIM test-suite: deterministic test meshes and baseline-file parsing.
#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef HSIM_SOURCE_DIR
#error "HSIM_SOURCE_DIR must be defined by the build system"
#endif

namespace hsim_test {

/// Absolute path of a file given relative to the repository root.
inline std::string repoPath(const std::string& relative) { return std::string(HSIM_SOURCE_DIR) + "/" + relative; }

/// Unit icosphere obtained by `subdivisions` rounds of 1-to-4 midpoint subdivision.
/// The continuous Laplace--Beltrami eigenvalues of the unit sphere are l(l+1) with multiplicity 2l+1.
inline void icosphere(int subdivisions, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	const double t = (1.0 + std::sqrt(5.0)) / 2.0;
	std::vector<Eigen::RowVector3d> v = { {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
	                                      {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1} };
	std::vector<Eigen::RowVector3i> f = { {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
	                                      {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
	                                      {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1} };
	for (auto& p : v) p.normalize();
	for (int s = 0; s < subdivisions; ++s) {
		std::map<std::pair<int, int>, int> midpoint;
		std::vector<Eigen::RowVector3i> refined;
		auto mid = [&](int a, int b) {
			const auto key = std::minmax(a, b);
			const auto it = midpoint.find(key);
			if (it != midpoint.end()) return it->second;
			v.push_back((v[a] + v[b]).normalized());
			return midpoint[key] = static_cast<int>(v.size()) - 1;
		};
		for (const auto& tri : f) {
			const int a = mid(tri[0], tri[1]), b = mid(tri[1], tri[2]), c = mid(tri[2], tri[0]);
			refined.push_back({ tri[0], a, c });
			refined.push_back({ tri[1], b, a });
			refined.push_back({ tri[2], c, b });
			refined.push_back({ a, b, c });
		}
		f = refined;
	}
	V.resize(static_cast<Eigen::Index>(v.size()), 3);
	F.resize(static_cast<Eigen::Index>(f.size()), 3);
	for (size_t i = 0; i < v.size(); ++i) V.row(static_cast<Eigen::Index>(i)) = v[i];
	for (size_t i = 0; i < f.size(); ++i) F.row(static_cast<Eigen::Index>(i)) = f[i];
}

/// Reference results recorded from the original (2022) implementation, see tests/data/README.md.
struct Baseline {
	std::string mesh;          ///< repository-relative mesh path, or "icoN" for an icosphere
	int numModes = 0;
	double tol = 0.0;
	int maxIter = 0;
	double shift = 0.0;
	int numVertices = 0;
	std::vector<int> levels;
	std::vector<double> eigenvalues;
	std::vector<double> residuals;
};

inline Baseline readBaseline(const std::string& name)
{
	const std::string path = repoPath("tests/data/baseline_" + name + ".txt");
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot open baseline file " + path);

	Baseline b;
	std::string line;
	while (std::getline(in, line)) {
		if (line.rfind("# baseline:", 0) == 0) {
			auto field = [&](const std::string& key, const std::string& next) {
				const size_t p = line.find(key + "=") + key.size() + 1;
				const size_t q = next.empty() ? line.size() : line.find(" " + next + "=");
				return line.substr(p, q - p);
			};
			b.mesh = field("mesh", "nModes");
			b.numModes = std::stoi(field("nModes", "tol"));
			b.tol = std::stod(field("tol", "maxIter"));
			b.maxIter = std::stoi(field("maxIter", "shift"));
			b.shift = std::stod(field("shift", ""));
			continue;
		}
		std::istringstream ss(line);
		std::string key;
		ss >> key;
		if (key == "nV") ss >> b.numVertices;
		else if (key == "levels") { int k; while (ss >> k) b.levels.push_back(k); }
		else if (key == "eigenvalues") { double x; while (ss >> x) b.eigenvalues.push_back(x); }
		else if (key == "residual") { double x; while (ss >> x) b.residuals.push_back(x); }
	}
	return b;
}

} // namespace hsim_test
