#ifndef EIGEN_SOLVER_H
#define EIGEN_SOLVER_H

#include <Eigen/Eigen>

using namespace std;

/* Computing Eigenstructure in GPU or via Matlab */
int computeEigenGPU(const Eigen::SparseMatrix<double> &S_, const Eigen::SparseMatrix<double> &M_, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal);
int computeEigenGPUDense(Eigen::MatrixXd &S, Eigen::MatrixXd &M, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal);
///void computeEigenGPUDenseJacobi(Eigen::MatrixXd &S, Eigen::MatrixXd &M, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal);
void computeEigenGPUDense2(Eigen::MatrixXd &S, Eigen::MatrixXd &M, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal);
///void computeEigenGPUDenseJacobi(Eigen::MatrixXd &S, Eigen::MatrixXd &M, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal);
void computeEigenSpectraDense(Eigen::MatrixXd &S, Eigen::MatrixXd &M, const int& numEigs, Eigen::MatrixXd &EigVec, Eigen::VectorXd &EigVal);
//type: 0=smallest_ALGE, 1=smallest_MAGN
void computeEigenSpectraSparse(Eigen::SparseMatrix<double> &S, Eigen::SparseMatrix<double> &M, const int& numEigs, Eigen::MatrixXd &EigVec, Eigen::VectorXd &EigVal, int type=0, int numIters=1000, double tol=1.0e-10);


void GaussSeidelSparseSolver(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x, int maxIter, double tol);
void GaussSeidelSparseSolver(Eigen::SparseMatrix<double>& LHS, Eigen::SparseMatrix<double>& M, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x, int maxIter, double tol);
void GaussSeidelSparseSolverMatrixForm(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x, int maxIter, double tol);
void GaussSeidel1Iter(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0,  Eigen::VectorXd& x);
void GaussSeidel1IterMatrixForm(const Eigen::SparseLU<Eigen::SparseMatrix<double>>& factor, const Eigen::SparseMatrix<double>& U, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x);


#endif // !EIGEN_SOLVER_H

