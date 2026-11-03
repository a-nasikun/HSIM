## The Hierarchical Subspace Iteration Method for Laplace–Beltrami Eigenproblems

This is the public release code for **The Hierarchical Subspace Iteration Method for Laplace–Beltrami Eigenproblems**. For more details, please refer to [the paper](https://dl.acm.org/doi/abs/10.1145/3495208). 

In this paper, we propose a new method to compute the lowest spectrum of Laplace--Beltrami operator on surface mesh. The main contribution is designing a multigrid prolongation operator to speed up the Subspace Iteration Method (SIM). The main ingredients of our method are:
1. Hierarchy construction
  - Setting up degrees of freedom (DOFs) on multiple levels and performing sampling accordingly
  - Computing the prolongation operator, using the (approximated) geodesic distance as the weight
2. Performing SIM on the hierarchy
  - Executing SIM from the coarsest level
  - Using the result as an approximation for the next (finer) level

Dependencies: **Eigen** and **libigl** (both downloaded automatically by CMake), **OpenMP**, and optionally **CUDA** (cuSOLVER) for the dense eigenproblems. Without CUDA, a CPU implementation based on Eigen is used.

## Installation and Compilation

We use **CMake** (version 3.20 or newer) and a C++17 compiler.

1. **Clone** the repository: `git clone https://github.com/a-nasikun/HSIM.git`
2. **Configure and build** from the repository root:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j8
   ```
   CMake options:

   | option | default | meaning |
   |--------|---------|---------|
   | `HSIM_WITH_CUDA` | `OFF` | solve the dense eigenproblems with cuSOLVER (needs the CUDA toolkit) |
   | `HSIM_BUILD_EXAMPLES` | `ON` | build the GUI example `HSIM_GUI` (needs OpenGL; on Linux the X11 development packages) |
   | `HSIM_BUILD_TESTS` | `ON` | build the test-suite (`ctest --test-dir build`) |

3. **Run** the example: `./build/Example/1_HSIM_GUI/HSIM_GUI [mesh.obj]`

This program is still under continuous development, so please expect some errors/bugs/mistakes within it. If you have any problem compiling it or having some nice suggestion(s) or question(s), please drop me an **email** at `ahmad.nasikun[at]ugm.ac.id`.

## ACM Transactions on Graphics 2022 (ACM SIGGRAPH 2022)

This work was presented at [SIGGRAPH 2022](https://s2022.siggraph.org/full-program/) at Vancouver Convention Center, Vancouver, Canada, and is published at [ACM Transactions on Graphics](https://dl.acm.org/doi/abs/10.1145/3495208). 

If you use our application in your work, please kindly cite our work.
```
@article{nasikun2022hierarchical,
  title={The hierarchical subspace iteration method for Laplace--Beltrami eigenproblems},
  author={Nasikun, Ahmad and Hildebrandt, Klaus},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={2},
  pages={1--14},
  year={2022},
  publisher={ACM New York, NY}
}
```
