# HSIM — The Hierarchical Subspace Iteration Method for Laplace–Beltrami Eigenproblems

[![CI](https://github.com/a-nasikun/HSIM/actions/workflows/ci.yml/badge.svg)](https://github.com/a-nasikun/HSIM/actions/workflows/ci.yml)

C++ library for computing the lowest part of the spectrum of the Laplace–Beltrami operator on triangle meshes, accompanying
the paper [*The Hierarchical Subspace Iteration Method for Laplace–Beltrami Eigenproblems*](https://doi.org/10.1145/3495208)
(ACM Transactions on Graphics 41(2), 2022, presented at SIGGRAPH 2022).

The method accelerates the subspace iteration method (SIM) with a multilevel hierarchy:

1. **Hierarchy construction** — nested farthest-point samples define the degrees of freedom on every level; prolongation
   operators between the levels are built from radial basis functions of the (approximate) geodesic distance.
2. **Hierarchical subspace iteration** — the eigenproblem is solved on the coarsest level; the result is prolongated to the
   next finer level and used as the start of a shifted subspace iteration there, down to the mesh itself.

[`docs/method.md`](docs/method.md) describes exactly what the code computes and maps the notation of the code to the math.

## Quick start

```cpp
#include <hsim/hsim.h>

Eigen::MatrixXd V;  // vertex positions (n x 3)
Eigen::MatrixXi F;  // triangles (m x 3)
hsim::readMesh("mesh.obj", V, F);

hsim::Options options;          // defaults reproduce the published results
options.tolerance = 1e-4;       // relative residual of every eigenpair
options.maxIterations = 10;     // per level
hsim::Result result = hsim::eigs(V, F, 100, options);

result.eigenvalues;             // 100 smallest eigenvalues, ascending
result.eigenvectors;            // n x 100, M-orthonormal
result.diagnostics;             // levels, iterations, shifts, timings, convergence history
```

`hsim::Solver` keeps the operators and the hierarchy of a mesh for repeated solves and exposes the individual stages
(`computeSamples`, `computeProlongation`, `solveCoarseProblem`, `solve`). Invalid input raises exceptions; progress
output is enabled with `options.verbosity`.

## Building

Requirements: CMake ≥ 3.20, a C++17 compiler, optionally OpenMP (parallel construction and solves) and the CUDA toolkit.
Eigen and libigl are downloaded automatically at pinned versions (libigl v2.5.0, Eigen 3.4.0).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

| CMake option | default | meaning |
|--------------|---------|---------|
| `HSIM_WITH_CUDA` | `OFF` | solve the dense eigenproblems with cuSOLVER on the GPU (otherwise Eigen on the CPU) |
| `HSIM_BUILD_EXAMPLES` | `ON` | GUI example `HSIM_GUI` (OpenGL; on Linux the X11 development packages, e.g. `libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev`) |
| `HSIM_BUILD_CLI` | `ON` | command-line program `hsim_cli` |
| `HSIM_BUILD_TESTS` | `ON` | test-suite |
| `HSIM_BUILD_BENCHMARKS` | `OFF` | stage-timing program `hsim_benchmark` |
| `HSIM_BUILD_DOCS` | `OFF` | target `hsim_docs` (Doxygen API documentation) |
| `HSIM_WARNINGS_AS_ERRORS` | `OFF` | treat warnings as errors (used in CI) |

To use HSIM in another CMake project, add it with `add_subdirectory` (or `FetchContent`) and link `HSIM::HSIM`.

## Programs

**Command line** — computes eigenpairs and records everything needed to reproduce the run:

```bash
./build/Example/2_HSIM_CLI/hsim_cli Models/AIM366_Kitten/366_Kitten_Flip.obj --modes 100 --tol 1e-2 --max-iter 5 -o kitten
```

writes `kitten_eigenvalues.csv` (eigenvalue and relative residual per mode) and `kitten_run.json` (library version,
mesh, all options, hierarchy, iterations, shifts and timings per level). `--save-vectors` also writes the eigenvectors;
`--help` lists all options.

**GUI** — `./build/Example/1_HSIM_GUI/HSIM_GUI [mesh]` constructs the hierarchy step by step and visualises samples,
geodesic distances, basis functions of every level and the eigenvectors.

**Benchmark** — `./build/benchmarks/hsim_benchmark <mesh> [modes] [tol] [maxIter] [shift] [repetitions] [verbosity]`
prints the time of every stage.

## Reproducibility and testing

* The algorithm is deterministic (sampling starts at vertex 0, no random numbers); all parameters are in `hsim::Options`
  and are written to the JSON record by `hsim_cli`.
* `tests/` contains regression tests against eigenvalues recorded from the original 2022 implementation
  (reproduced to ~10⁻¹⁴; `tests/data/README.md` documents how they were produced), a physical check on the sphere
  (eigenvalues l(l+1) with multiplicity 2l+1), a comparison with an independent sparse eigensolver (Spectra), unit tests
  of the individual stages, and smoke tests of `hsim_cli`.
* CI builds with GCC (warnings as errors), runs the tests in Release and with AddressSanitizer/UBSan, and checks
  formatting.

## Citation

If you use this code, please cite the paper (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{nasikun2022hierarchical,
  title     = {The Hierarchical Subspace Iteration Method for {Laplace--Beltrami} Eigenproblems},
  author    = {Nasikun, Ahmad and Hildebrandt, Klaus},
  journal   = {ACM Transactions on Graphics},
  volume    = {41},
  number    = {2},
  pages     = {1--14},
  year      = {2022},
  publisher = {ACM},
  doi       = {10.1145/3495208}
}
```

## License and contact

MIT license (see [`LICENSE`](LICENSE)). Questions, bug reports and suggestions are welcome: open an issue or write to
`ahmad.nasikun[at]ugm.ac.id`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the code conventions and
[`CHANGELOG.md`](CHANGELOG.md) for the changes since the 2022 release, including a migration table for the old API.
