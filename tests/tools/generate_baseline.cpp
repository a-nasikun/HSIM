// Generator for tests/data/baseline_*.txt (see tests/data/README.md).
// Builds against the ORIGINAL sources (commit 127634f) together with cpu_eigen.cpp; not part of the normal build.
#include "LBEigenSolver.h"
#include <igl/readOBJ.h>
#include <igl/massmatrix.h>
#include <fstream>
#include <iomanip>
// Icosphere generator (deterministic) for the analytic test case
static void icosphere(int sub, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  std::vector<Eigen::RowVector3d> v = {{-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0},{0,-1,t},{0,1,t},{0,-1,-t},{0,1,-t},{t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1}};
  std::vector<Eigen::RowVector3i> f = {{0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},{1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},{3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},{4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
  for (auto& p : v) p.normalize();
  for (int s = 0; s < sub; ++s) {
    std::map<std::pair<int,int>,int> mid; std::vector<Eigen::RowVector3i> nf;
    auto m = [&](int a, int b) { auto k = std::minmax(a,b); auto it = mid.find(k); if (it != mid.end()) return it->second;
      v.push_back((v[a]+v[b]).normalized()); return mid[k] = (int)v.size()-1; };
    for (auto& t3 : f) { int a=m(t3[0],t3[1]), b=m(t3[1],t3[2]), c=m(t3[2],t3[0]);
      nf.push_back({t3[0],a,c}); nf.push_back({t3[1],b,a}); nf.push_back({t3[2],c,b}); nf.push_back({a,b,c}); }
    f = nf;
  }
  V.resize(v.size(),3); F.resize(f.size(),3);
  for (size_t i=0;i<v.size();++i) V.row(i)=v[i]; for (size_t i=0;i<f.size();++i) F.row(i)=f[i];
}
int main(int argc, char** argv) {
  // args: name meshfile|icoN numModes tol maxIter shift out
  std::string name=argv[1], mesh=argv[2]; int n=atoi(argv[3]); double tol=atof(argv[4]); int it=atoi(argv[5]); double sh=atof(argv[6]);
  Eigen::MatrixXd V; Eigen::MatrixXi F;
  if (mesh.rfind("ico",0)==0) icosphere(atoi(mesh.c_str()+3), V, F); else igl::readOBJ(mesh, V, F);
  auto* s = new HSIM::LBEigenSolver();   // value-initialised, as in Example/1_HSIM_GUI/Main.cpp
  Eigen::VectorXd L; Eigen::MatrixXd X;
  std::tie(L, X) = s->eigs(V, F, n, tol, it, sh);
  std::ofstream o(argv[7]);
  o << std::setprecision(17);
  o << "# baseline: " << name << " mesh=" << mesh << " nModes=" << n << " tol=" << tol << " maxIter=" << it << " shift=" << sh << "\n";
  o << "nV " << V.rows() << "\nlevels"; for (int k : s->getSubspaceSize()) o << " " << k; o << "\n";
  o << "eigenvalues"; for (int i=0;i<n;++i) o << " " << L(i); o << "\n";
  // M-orthonormality and residual of returned vectors (columns 0..n-1)
  Eigen::SparseMatrix<double> M = s->getMassMatrix(), S = s->getStiffnessMatrix();
  o << "residual";
  for (int i=0;i<n;++i){ Eigen::VectorXd r = S*X.col(i) - L(i)*(M*X.col(i)); Eigen::VectorXd k = S*X.col(i);
    o << " " << std::sqrt(r.dot(M*r)/k.dot(M*k)); } o << "\n";
  return 0;
}
