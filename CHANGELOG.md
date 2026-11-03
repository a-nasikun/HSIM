# Changelog

## 2.0.0 — 2026 refactoring (branch `2026-refactor`)

The numerical method is unchanged: the regression tests reproduce the eigenvalues of the original implementation
(commit `127634f`) to a relative accuracy of 10⁻⁷ (see `tests/data/README.md`).

### Added
* Public, modular C++17 API in `HSIM/include/hsim/` (`hsim::eigs`, `hsim::Solver`, `hsim::Options`, `hsim::Result`
  with per-level diagnostics), documented in `docs/method.md`.
* CPU backend (Eigen) for the dense eigenproblems; CUDA (cuSOLVER) is optional (`HSIM_WITH_CUDA`) and selectable at
  run time (`Options::denseBackend`).
* Command-line program `hsim_cli` that writes eigenvalues, residuals and a JSON record of the run.
* Benchmark program, test-suite (regression, physics, unit tests), CI (GCC, sanitizers, formatting).
* `CITATION.cff`, `CONTRIBUTING.md`, `.clang-format`, `.clang-tidy`, Doxygen configuration.

### Fixed
* Uninitialised members (e.g. whether the basis is clamped), wrong partition of unity for rows with vanishing sum,
  out-of-bounds access in the convergence history and for fewer than 8 modes, uninitialised shift for tolerances
  below 10⁻⁹, dangling references returned by eigenvector getters, GPU memory leaks and unchecked CUDA errors,
  ineffective fallback for a failing coarse eigensolver, off-by-one random indices, global OpenMP side effects.

### Changed
* `eigs`/`solve` return exactly `numModes` eigenpairs (previously q = max(2·numModes, numModes + 8)).
* Errors are reported with exceptions; the library prints only when `Options::verbosity` asks for it.
* 1.9×–2.4× faster on the CPU (sampling, prolongation, Galerkin operators, parallel solves).
* Build: single top-level CMake project (≥ 3.20), libigl v2.5.0 / Eigen 3.4 fetched at pinned versions; the core library
  no longer depends on OpenGL. Model files renamed without spaces (`Models/AIM894_Chinese_Dragon/…`).

### Removed
* Clamped-basis experiment, Poisson-disk sampling, Gauss–Seidel solvers, empty Spectra stubs, diffusion tensor and
  unused utilities (all available in the history before this release).

### Migration from 1.x

| 1.x (`HSIM::LBEigenSolver`) | 2.0 |
|------------------------------|-----|
| `solver.eigs(V, F, n, tol, maxIters, shift)` | `hsim::Options o; o.tolerance = tol; o.maxIterations = maxIters; o.shiftParameter = shift; auto r = hsim::eigs(V, F, n, o);` |
| `setup(V, F, n)` + `buildHierarchy(n)` | `hsim::Solver s(o); s.setMesh(V, F); s.buildHierarchy(n);` |
| `performSubspaceIterationMethod(...)` | `s.solve(n)` |
| `getEigenValues()`, `getEigenVectors()` | `r.eigenvalues`, `r.eigenvectors` |
| `getStiffnessMatrix()`, `getMassMatrix()` | `s.operators().stiffness`, `s.operators().mass` |
| `getBasisMatrix()`, `getSample()`, `getSubspaceSize()` | `s.hierarchy().basis`, `.samples.vertices`, `.levelSizes` |
| `readMesh(file)` | `hsim::readMesh(file, V, F)` |
| `scaleMesh(V, F, d)` | `hsim::normalizeMesh(V, d)` |
| `-DIF_VERBOSE=ON` | `o.verbosity = hsim::Verbosity::Summary` (or `Detailed`) |

## 1.0 — 2022

Code accompanying the ACM Transactions on Graphics paper (SIGGRAPH 2022).
