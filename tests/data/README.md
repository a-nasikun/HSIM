# Regression baselines

The files `baseline_*.txt` hold reference results of the **original** HSIM implementation
(commit `127634f`, the state of `master` before the 2026 refactoring). They are the safety net
for the refactoring: every later version must reproduce these eigenvalues.

## How they were produced

* Sources: `HSIM/{LBEigenSolver,Sampling,Utility}.cpp` from commit `127634f`, unchanged.
* The cuSOLVER routine `computeEigenGPUDense` was replaced by an Eigen
  `GeneralizedSelfAdjointEigenSolver` (`tests/tools/cpu_eigen.cpp`), because no GPU was available.
  Both solve the same dense generalized symmetric-definite problem on the lower triangle.
* Dependencies: libigl v2.3.0 (Eigen 3.3.7), GCC 13.3, `-O3`, OpenMP, Linux x86-64.
* Driver: `tests/tools/generate_baseline.cpp`. The solver object is value-initialised
  (`new LBEigenSolver()`), exactly like `Example/1_HSIM_GUI/Main.cpp`, so that the
  (then uninitialised) member `isClampedBasis` is `false`.
* Meshes are used as stored in `Models/` (no rescaling); `icoN` is a unit icosphere with `N`
  subdivision rounds (generator in `tests/test_support.h`).

Command line of the driver: `generate_baseline <name> <mesh|icoN> <nModes> <tol> <maxIter> <shift> <out.txt>`.

| file | mesh | modes | tol | max. iter. | shift |
|------|------|-------|-----|------------|-------|
| `baseline_dragon2000_n20.txt` | Chinese dragon, 2001 vertices | 20 | 1e-4 | 20 | 0.25 |
| `baseline_dragon5000_n20.txt` | Chinese dragon, 5001 vertices | 20 | 1e-4 | 20 | 0.25 |
| `baseline_dragon5000_gui.txt` | Chinese dragon, 5001 vertices | 20 | 1e-2 | 5  | 0.25 |
| `baseline_sphere_n25.txt`     | icosphere, 2562 vertices      | 25 | 1e-4 | 20 | 0.25 |

## File format

```
# baseline: <name> mesh=<path> nModes=<n> tol=<tol> maxIter=<it> shift=<s>
nV <number of vertices>
levels <subspace size of level 0> <level 1> ...
eigenvalues <lambda_0> ... <lambda_{n-1}>
residual <r_0> ... <r_{n-1}>      # r_i = ||S x_i - lambda_i M x_i||_M / ||S x_i||_M
```

`r_0` is meaningless (the constant mode has `S x_0 = 0`).
