## The Hierarchical Subspace Iteration Method for Laplace–Beltrami Eigenproblems

This is the public release code for **The Hierarchical Subspace Iteration Method for Laplace–Beltrami Eigenproblems**. For more details, please refer to [the paper](https://dl.acm.org/doi/abs/10.1145/3495208). 

In this paper, we propose a new method to compute the lowest spectrum of Laplace--Beltrami operator on surface mesh. The main contribution is designing a multigrid prolongation operator to speed up the Subspace Iteration Method (SIM). The main ingredients of our method are:
1. Hierarchy construction
  - Setting up degrees of freedom (DOFs) on multiple levels and performing sampling accordingly
  - Computing the prolongation operator, using the (approximated) geodesic distance as the weight
2. Performing SIM on the hierarchy
  - Executing SIM from the coarsest level
  - Using the result as an approximation for the next (finer) level

Please note that this program requires **Eigen**, **LibIGL**, and an **GPU-based eigensolver** (e.g. **CUDA's cuSOLVER** function).

## Installation and Compilation

We use **CMake** for easy installation/deployment on your machine. Here are the steps to get our program run:

1. **Clone** our repository **recursively** using `git clone https://github.com/a-nasikun/HSIM.git --recursive`.
2. Ensure that you have **CUDA set-up** on your machine. You can download from [here](https://developer.nvidia.com/cuda-zone "CUDA Developer Zone") if you haven't done so.
<!---3. **Update dependency** submodule(s). Among many external libraries in LibIGL, we'll be using eigen, glfw, glad, and imgui only. During our test, the commit version of eigen in libIGL is an old one and doesn't support `Eigen::SparseMatrix.` We can solve this by simply updating the eigen, using `git submodule update --remote external\eigen`. --->
3. Run the Cmake command (either the command line or the `CMakeGUI`)
   ```C++
   cd Example
   mkdir Build
   cd Build
   cmake -DCMAKE_BUILD_TYPE=Release
   make -j8
   ```
   If some headers/libraries are missing (perhaps because you located the wrong folder), you could specify it directly in your application. For example, in Visual Studio you could go to `Project` > `Properties` > `C/C++` > `General` > `Additional Include Directories`.

Having followed those steps, our program **should compile and run** well. This program is still under continuous development, so please expect some errors/bugs/mistakes within it. If you have any problem compiling it or having some nice suggestion(s) or question(s), please drop me an **email** at `ahmad.nasikun[at]ugm.ac.id`.

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
