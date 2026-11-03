# The method as implemented

This page describes exactly what the code computes, with the notation used in the source. For the derivation and the
analysis see the paper:

> A. Nasikun and K. Hildebrandt. *The Hierarchical Subspace Iteration Method for Laplace–Beltrami Eigenproblems.*
> ACM Transactions on Graphics 41(2), 2022. [doi:10.1145/3495208](https://doi.org/10.1145/3495208)

## 1. Discrete eigenproblem

For a triangle mesh with vertices `V` (n × 3) and faces `F`, HSIM computes the `numModes` smallest eigenpairs of

$$ S\,\phi_i = \lambda_i\, M\,\phi_i ,\qquad \phi_i^\top M \phi_j = \delta_{ij}, $$

where `S` is the cotangent stiffness matrix (`-igl::cotmatrix`) and `M` the lumped Voronoi mass matrix
(`igl::massmatrix`, `MASSMATRIX_TYPE_VORONOI`) — see `hsim::assembleOperators`. On a closed surface λ₀ = 0 with a
constant eigenvector.

## 2. Hierarchy (`hsim::buildHierarchy`)

**Level sizes** (`samplingLevelSizes`). With q = max(2·numModes, numModes + 8) (`subspaceSize`) and ratio
r = `Options::levelRatio`: starting from m = n, repeatedly set m ← round(m / r); stop *before* adding a level with
fewer than q samples, and *after* adding a level with at most 2q samples. This gives m₀ > m₁ > … > m_{L−1} ≥ q.

**Samples** (`farthestPointSampling`). Nested farthest-point sampling with respect to the graph distance along mesh
edges (Dijkstra). The first sample is vertex 0; the coarsest level is sampled first, finer levels extend it, so the first
m_k entries of `Samples::vertices` are the samples of level k.

**Prolongation** (`buildProlongation`). The basis function of sample j on level k is

$$ \varphi(d) = 2\,(d/r_k)^3 - 3\,(d/r_k)^2 + 1 \quad (d \le r_k),\qquad 0 \text{ otherwise}, $$

of the graph distance d to the sample, with support radius r_k from π r_k² = `areaFactor` · `supportSize` · A / m_k
(A: surface area). `U_0` (n × m₀) holds these functions at the vertices, `U_k` (m_{k−1} × m_k) at the samples of level
k−1. Each `U_k` is normalised to a partition of unity (rows sum to one; rows with vanishing sum are left unchanged).

> `areaFactor` defaults to 2 because the original implementation summed the *doubled* triangle areas in this formula.
> The default keeps the published results; `areaFactor = 1` uses the plain area (equivalent to halving `supportSize`).

**Galerkin operators.** S₋₁ = S, S_k = U_kᵀ S_{k−1} U_k, and likewise M_k (`Hierarchy::levelStiffness/levelMass`).
P = U₀U₁⋯U_{L−1} (`Hierarchy::basis`) maps coarsest coefficients to vertex values.

**Coarsest problem** (`solveCoarseProblem`). All eigenpairs of the dense pencil (S_{L−1}, M_{L−1}). If the dense solver
fails it is retried with S_{L−1} + εI, ε = 10⁻¹⁵, 10⁻¹⁴, … (at most nine times).

## 3. Hierarchical subspace iteration (`hsim::hierarchicalSubspaceIteration`)

Start with the q lowest coarse eigenpairs (X, Λ). For k = L−1, …, 0: prolongate X ← U_k X and iterate on the
operators of that space (S_{k−1}, M_{k−1}; S, M on level 0):

1. **Shift, once per level.** μ = Λ_s with s = c + ⌊`shiftParameter` · (numModes − c)⌋ and c = 1 (index 0, the
   constant mode, is never used). Factorise S − μM with a sparse LDLᵀ (`Eigen::SimplicialLDLT`).
   On the finest level with `tolerance` < 10⁻⁹ no shift is used (μ = 0, S + 10³ε_mach I is factorised).
2. **Inverse iteration.** X̄ = (S − μM)⁻¹ M X (q independent solves).
3. **Rayleigh–Ritz.** Solve the dense q × q problem (X̄ᵀSX̄) Q = (X̄ᵀMX̄) Q Λ, set X = X̄ Q.
   If it fails, the operator is refactorised as S + εI with ε growing by a factor 10 each time.
4. **Convergence.** Relative residuals r_i = ‖S x_i − λ_i M x_i‖_M / ‖S x_i‖_M (`relativeResiduals`).
   Let c be the largest index such that r_c', …, r_c are all below the tolerance, where c' is the value of c from the
   previous iteration (initially 1). The level is finished when `maxIterations` is reached, or when
   r_{numModes−1} < `tolerance` and c = numModes − 1.

The result consists of the first numModes columns of X on level 0 (M-orthonormal) and their Ritz values.
`Diagnostics` reports per level the dimension, iterations, shift and time, and on level 0 the history of the Ritz
values, residuals, and the criterion √|1 − λ_i² / (q_iᵀq_i + μ(2λ_i − μ))| of the Ritz coefficient vectors q_i.

## 4. Notation in the code

| symbol | code | meaning |
|--------|------|---------|
| n, L | `V.rows()`, `Hierarchy::numLevels()` | vertices, levels |
| m_k | `Hierarchy::levelSizes[k]` | samples on level k |
| q | `subspaceSize(numModes)` | size of the iterated subspace |
| S, M | `Operators::stiffness`, `Operators::mass` | stiffness and mass matrices |
| U_k | `Hierarchy::prolongation[k]` | prolongation from level k to level k−1 (the mesh for k = 0) |
| P | `Hierarchy::basis` | U₀ ⋯ U_{L−1} |
| S_k, M_k | `Hierarchy::levelStiffness[k]`, `levelMass[k]` | operators on the samples of level k |
| r_k | `Hierarchy::supportRadii[k]` | support radius of the basis functions on level k |
| μ | `LevelDiagnostics::shift` | spectral shift |

## 5. Determinism

The sampling starts at vertex 0 and contains no randomness; the prolongation is assembled from triplets without
duplicates, so its value does not depend on the number of OpenMP threads. The remaining sources of run-to-run
differences are the floating-point summation order in multi-threaded dense products (Eigen) and the GPU backend.
