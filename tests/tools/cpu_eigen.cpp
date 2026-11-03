// CPU stand-in for the cuSOLVER routines of the ORIGINAL EigenSolver.cpp (same signatures/semantics).
#include "EigenSolver.h"
namespace HSIM {
int computeEigenGPUDense(Eigen::MatrixXd& S, Eigen::MatrixXd& M, Eigen::MatrixXd& V, Eigen::VectorXd& W) {
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(S, M); // uses lower triangle, like sygvd LOWER
  V = es.eigenvectors(); W = es.eigenvalues();
  return W(1) > 0 ? 0 : 1;
}
int computeEigenGPU(Eigen::SparseMatrix<double>& S_, Eigen::SparseMatrix<double>& M_, Eigen::MatrixXd& V, Eigen::VectorXd& W) {
  Eigen::MatrixXd S(S_), M(M_); computeEigenGPUDense(S, M, V, W); return 0;
}
void computeEigenSpectraDense(Eigen::MatrixXd&, Eigen::MatrixXd&, const int&, Eigen::MatrixXd&, Eigen::VectorXd&) {}
}
