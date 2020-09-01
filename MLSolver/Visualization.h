#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <igl/opengl/glfw/Viewer.h>

#include <vector>

/* Visualize the samples */
void visualizeSample(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, double nEps = 0.001);
void visualizeSampleMulti(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps=0.001, double scaleRatio = 0.005, int sample1=0, int sample2=1);
void scaleSampleMulti(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps=0.001, double scaleRatio = 0.005);
void visualizeSample_1(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps=0.001);
void visualizeSample_2(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps = 0.001);
void visualizeSample_3(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps = 0.001);
void visualizeSampleFirstLevel(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps = 0.001);
void visualizeSampleSecondLevel(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &Sample, const std::vector<int> &sampleSize, double nEps = 0.001);

/* Visualize basis */
void visualizeBasis(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::SparseMatrix<double> &U, const int &basisID);
/* Visualize Approximated Eigenvector */
void visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::SparseMatrix<double> &U, const Eigen::MatrixXd &LDEigVec, const int &eigenID);

#endif // !VISUALIZATION_H

