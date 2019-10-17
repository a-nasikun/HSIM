#ifndef EIGEN_SOLVER_H
#define EIGEN_SOLVER_H

#include <iostream>
#include <assert.h> 

#include <Eigen/Eigen>

// For CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

/* Computing Eigenstructure in GPU or via Matlab */
void computeEigenGPU(Eigen::SparseMatrix<double> &S_, Eigen::SparseMatrix<double> &M_, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal);
void computeEigenMatlab(Eigen::SparseMatrix<double> &S, Eigen::SparseMatrix<double> &M, const int& numEigs, Eigen::MatrixXd &EigVec, Eigen::VectorXd &EigVal, const string& filename);

void GaussSeidelSparseSolver(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x, int maxIter, double tol);
void GaussSeidel1Iter(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0,  Eigen::VectorXd& x);

#endif // !EIGEN_SOLVER_H

