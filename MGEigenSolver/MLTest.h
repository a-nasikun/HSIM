#ifndef ML_TEST_HPP
#define ML_TEST_HPP

#include "MLSolverFunctions.h"
void testInit(igl::opengl::glfw::Viewer &viewer, MLEigenSolver*& eigenSolver, const string& meshFile);
void runMLSolverTest(igl::opengl::glfw::Viewer &viewer, MLEigenSolver*& eigenSolver);


#endif // !ML_TEST_HPP