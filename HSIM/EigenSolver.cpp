#include <cstdio>
#include <cstdlib>
#include <vector>

#include "EigenSolver.h"
#include <chrono>

#include <iostream>
#include <assert.h> 

//#include <Eigen/PardisoSupport>

// For CUDA (optional: without HSIM_WITH_CUDA a CPU implementation based on Eigen is used)
#ifdef HSIM_WITH_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cusolverMg.h"
#endif

#include <stdexcept>
#include <string>
#include <tuple>


// Matlab interface
//#include <igl/matlab/matlabinterface.h> 

// SPECTRA EIGENSOLVER
//#include <Spectra/GenEigsSolver.h>
//#include <Spectra/SymEigsSolver.h>
//#include <Spectra/SymEigsShiftSolver.h>
//#include <Spectra/GenEigsRealShiftSolver.h>
//#include <Spectra/SymGEigsSolver.h>
//#include <Spectra/MatOp/SparseCholesky.h>

namespace HSIM {
	/* Computing the eigenstructure of a sparse pencil (S_, M_) by converting it to dense matrices. */
	int computeEigenGPU(Eigen::SparseMatrix<double>& S_, Eigen::SparseMatrix<double>& M_, Eigen::MatrixXd& LDEigVec, Eigen::VectorXd& LDEigVal)
	{
		Eigen::MatrixXd Snew(S_);
		Eigen::MatrixXd Mnew(M_);
		return computeEigenGPUDense(Snew, Mnew, LDEigVec, LDEigVal);
	}

#ifndef HSIM_WITH_CUDA
	/* CPU implementation: dense generalized symmetric-definite eigenproblem S x = lambda M x (Eigen, lower triangle).
	 * Returns 0 on success, 1 if the solver failed or produced non-finite values. */
	int computeEigenGPUDense(Eigen::MatrixXd& S, Eigen::MatrixXd& M, Eigen::MatrixXd& LDEigVec, Eigen::VectorXd& LDEigVal)
	{
		Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(S, M);
		if (solver.info() != Eigen::Success) return 1;
		LDEigVec = solver.eigenvectors();
		LDEigVal = solver.eigenvalues();
		return LDEigVal.allFinite() ? 0 : 1;
	}
#else
	namespace {
		void checkCuda(cudaError_t err, const char* what)
		{
			if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
		}
		void checkCusolver(cusolverStatus_t status, const char* what)
		{
			if (status != CUSOLVER_STATUS_SUCCESS)
				throw std::runtime_error(std::string(what) + " failed (cuSOLVER status " + std::to_string(static_cast<int>(status)) + ")");
		}

		/* RAII wrapper of a device buffer */
		template <typename T>
		class DeviceBuffer {
		public:
			explicit DeviceBuffer(size_t n) { checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * n), "cudaMalloc"); }
			~DeviceBuffer() { cudaFree(ptr_); }
			DeviceBuffer(const DeviceBuffer&) = delete;
			DeviceBuffer& operator=(const DeviceBuffer&) = delete;
			T* get() const { return ptr_; }
		private:
			T* ptr_ = nullptr;
		};

		/* RAII wrapper of a cuSOLVER dense handle */
		class CusolverDnHandle {
		public:
			CusolverDnHandle() { checkCusolver(cusolverDnCreate(&handle_), "cusolverDnCreate"); }
			~CusolverDnHandle() { cusolverDnDestroy(handle_); }
			CusolverDnHandle(const CusolverDnHandle&) = delete;
			CusolverDnHandle& operator=(const CusolverDnHandle&) = delete;
			cusolverDnHandle_t get() const { return handle_; }
		private:
			cusolverDnHandle_t handle_ = nullptr;
		};
	}

	/* GPU implementation (cuSOLVER sygvd), adapted from the cuSOLVER example codes.
	 * Returns 0 on success, 1 if the solver reported an error or produced non-finite values;
	 * CUDA runtime errors are reported by exceptions. All device resources are released on every path. */
	int computeEigenGPUDense(Eigen::MatrixXd& S, Eigen::MatrixXd& M, Eigen::MatrixXd& LDEigVec, Eigen::VectorXd& LDEigVal)
	{
		const int m = static_cast<int>(S.rows());
		const int lda = m;
		const size_t numEntries = static_cast<size_t>(lda) * static_cast<size_t>(m);

		CusolverDnHandle handle;
		DeviceBuffer<double> d_A(numEntries), d_B(numEntries), d_W(static_cast<size_t>(m));
		DeviceBuffer<int> d_info(1);

		checkCuda(cudaMemcpy(d_A.get(), S.data(), sizeof(double) * numEntries, cudaMemcpyHostToDevice), "cudaMemcpy (S)");
		checkCuda(cudaMemcpy(d_B.get(), M.data(), sizeof(double) * numEntries, cudaMemcpyHostToDevice), "cudaMemcpy (M)");

		const cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;      // A*x = (lambda)*B*x
		const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;  // compute eigenvalues and eigenvectors
		const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

		int lwork = 0;
		checkCusolver(cusolverDnDsygvd_bufferSize(handle.get(), itype, jobz, uplo, m, d_A.get(), lda, d_B.get(), lda, d_W.get(), &lwork),
			"cusolverDnDsygvd_bufferSize");
		DeviceBuffer<double> d_work(static_cast<size_t>(lwork));

		checkCusolver(cusolverDnDsygvd(handle.get(), itype, jobz, uplo, m, d_A.get(), lda, d_B.get(), lda, d_W.get(), d_work.get(), lwork, d_info.get()),
			"cusolverDnDsygvd");

		int info = 0;
		LDEigVec.resize(m, m);
		LDEigVal.resize(m);
		checkCuda(cudaMemcpy(LDEigVec.data(), d_A.get(), sizeof(double) * numEntries, cudaMemcpyDeviceToHost), "cudaMemcpy (eigenvectors)");
		checkCuda(cudaMemcpy(LDEigVal.data(), d_W.get(), sizeof(double) * m, cudaMemcpyDeviceToHost), "cudaMemcpy (eigenvalues)");
		checkCuda(cudaMemcpy(&info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy (info)");

		if (info != 0) return 1;
		return LDEigVal.allFinite() ? 0 : 1;
	}
#endif // HSIM_WITH_CUDA

	void computeEigenSpectraDense(Eigen::MatrixXd& S, Eigen::MatrixXd& M, const int& numEigs, Eigen::MatrixXd& EigVec, Eigen::VectorXd& EigVal)
	{

		///////fake mass matrix
		////Eigen::SparseMatrix<double> M2;
		////M2.resize(S.rows(), S.cols());
		////M2.setIdentity();
		////
		////cout << "Describe operator \n";
		////Spectra::DenseSymMatProd<double> Sop(S);
		////Spectra::DenseCholesky<double> Mop(M);
		////Spectra::SparseCholesky<double> Mop2(M2);
		////
		////cout << "Creating the solver \n";
		//////Spectra::SymGEigsSolver<double, Spectra::SortRule::SmallestAlge,Spectra::GEIGS_CHOLESKY> eigensolver(&Sop, &Mop, 50, 2 * 50);
		//////Spectra::SymGEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigensolver(&Sop, &Mop2, 50, 2 * 50);
		//////Spectra::SymGEigsSolver<Spectra::DenseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky> eigensolver(Sop, Mop2, 50, 2 * 50);
		////Spectra::SymGEigsSolver<Spectra::DenseSymMatProd<double>, Spectra::DenseCholesky<double>, Spectra::GEigsMode::Cholesky> eigensolver(Sop, Mop, numEigs, 2 * numEigs);
		////
		////
		////cout << "Initiating the eigenproblem \n";
		////eigensolver.init();
		////
		////cout << "Computing the eigenproblem \n";
		////int nconv = eigensolver.compute(Spectra::SortRule::SmallestAlge);
		////if (eigensolver.info() == Spectra::CompInfo::Successful) {
		////	cout << "Successful! \n";
		////}
		////else {
		////	cout << "Failsedl! \n";
		////}
		////
		////EigVec = eigensolver.eigenvectors().rowwise().reverse();
		////EigVal = eigensolver.eigenvalues().reverse();
		////
		////cout << "Eigenvalues: " << EigVal.transpose() << endl; 
	}

	/* Sparse spectra eigensolver */
	void computeEigenSpectraSparse(Eigen::SparseMatrix<double>& S, Eigen::SparseMatrix<double>& M, const int& numEigs, Eigen::MatrixXd& EigVec, Eigen::VectorXd& EigVal, int type, int numIters, double tol)
	{
		///////cout << "Computing eigenproblem in Spectra...";
		///////chrono::high_resolution_clock::time_point	t1, t2;
		///////chrono::duration<double>					duration;
		////
		////printf("M: %dx%d | S:%dx%d \n", M.rows(), M.cols(), S.rows(), S.cols());
		////
		////Spectra::SparseSymMatProd<double> SOp(S);
		////Spectra::SparseCholesky<double> MOp(M);
		////
		////printf("Wrapper \n");
		////
		////map<int, Spectra::SortRule> typeMap;
		////typeMap.insert(std::pair<int, Spectra::SortRule>(0, Spectra::SortRule::SmallestAlge));
		////typeMap.insert(std::pair<int, Spectra::SortRule>(1, Spectra::SortRule::SmallestImag));
		////const Spectra::SortRule paramType = typeMap[type];
		////
		//////Spectra::SymGEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> spectraSolver(&SOp, &MOp, numEigs, 2 * numEigs);
		////Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky> spectraSolver(SOp, MOp, numEigs, 2*numEigs);
		////
		////cout << "Trying to initiate \n";
		////spectraSolver.init();
		////std::cout << "Trying to solve | " << numEigs << " eigenpairs |" << numIters << "max iters |" << tol << " tolerance \n";
		//////int info = spectraSolver.compute(numIters, tol, typeMap[type]);
		//////int info = spectraSolver.compute(typeMap[type], numIters, tol);
		////int info = spectraSolver.compute(Spectra::SortRule::SmallestAlge, numIters, tol);
		////cout << "Result of solving: " << (int) spectraSolver.info() << endl;
		////
		////if (spectraSolver.info() == Spectra::CompInfo::Successful) {
		////	EigVal = spectraSolver.eigenvalues().inverse();
		////	EigVec = spectraSolver.eigenvectors().rowwise().reverse();
		////	//EigVal = spectraSolver.eigenvalues();
		////	//EigVec = spectraSolver.eigenvectors();
		////	cout << "Yes! Made it! \n";
		////}
		////else {
		////	int numConverged = spectraSolver.compute();
		////	std::cout << "Converged eigenpairs " << numConverged << std::endl;
		////	std::cout << "EigVals" << spectraSolver.eigenvalues().topRows(numConverged - 1).transpose() << std::endl;
		////	EigVal = spectraSolver.eigenvalues().topRows(numConverged-1).inverse();
		////	EigVec = spectraSolver.eigenvectors().leftCols(numConverged-1).rowwise().reverse();
		////	//std::cout << "Computation failed \n";
		////}
		////
		///////t2 = chrono::high_resolution_clock::now();
		///////duration = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
		///////cout << "in " << duration.count()/1000 << " seconds." << endl;
	}

	/* Iterative Gauss-Seidel Solver */
	void GaussSeidelSparseSolver(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x, int maxIter, double tol)
	{
		//printf("Dimensionality check: \n");
		printf("____LHS: %dx%d | rhs: %d | init guess: %d | ", LHS.rows(), LHS.cols(), rhs.rows(), x0.rows());
		cout << tol << " | ";

		// For Timing
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;
		//cout << "____Iterative Gauss-Seidel solver...\n";
		t1 = chrono::high_resolution_clock::now();

		Eigen::VectorXd xk1 = x0;
		Eigen::VectorXd xk2;
		double diffIter;
		double maxDiffNum, minDiffNum;
		double maxCurNum, minCurNum;
		int iterUsed = maxIter;
		double res;
		Eigen::VectorXd residu;

		residu = rhs - LHS * xk1;
		res = residu.transpose() * LHS * residu;
		cout << "residue: " << res << endl;

		for (int i = 0; i < maxIter; i++) {
			GaussSeidel1Iter(LHS, rhs, xk1, xk2);

			maxDiffNum = (xk2 - xk1).maxCoeff();
			minDiffNum = (xk2 - xk1).minCoeff();
			maxCurNum = xk2.maxCoeff();
			minCurNum = xk2.minCoeff();
			//printf("MaxDiff: %.4f | minDiff: %.4f | maxCur: %.4f | minCur: %.4f \n", maxDiffNum, minDiffNum, maxCurNum, minCurNum);
			diffIter = max(maxDiffNum, abs(minDiffNum)) / max(maxCurNum, abs(minCurNum));
			//diffIter = maxDiffNum / maxCurNum;

			//cout << "x: " << xk2.transpose() << endl;


			residu = rhs - LHS * xk2;
			res = residu.transpose() * LHS * residu;
			//res = residu.norm();
			if (i % 1000 == 0) {
				//printf("______<%d> iter, the convergence: %.16f |", i, diffIter);			
				//printf(" residu: %.16f \n", res);
				cout << "______" << i << " iter | convergence: " << diffIter << " | residue: " << res << endl;
			}

			xk1 = xk2;

			if (res < tol)
				//if (diffIter < tol)
			{
				iterUsed = i + 1;
				break;
			}
		}
		x = xk2;

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		///cout << "in " << duration.count() << " seconds" << endl;
		//printf("____It takes %d iter, with %.16f tolerance \n", iterUsed, res);
		cout << "____" << iterUsed << " iterations | residue:" << res << ".\n";
	}

	void GaussSeidelSparseSolver(Eigen::SparseMatrix<double>& LHS, Eigen::SparseMatrix<double>& M, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x, int maxIter, double tol)
	{
		//printf("Dimensionality check: \n");
		printf("____LHS: %dx%d | rhs: %d | init guess: %d | ", LHS.rows(), LHS.cols(), rhs.rows(), x0.rows());
		cout << tol << " | ";

		// For Timing
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;
		//cout << "____Iterative Gauss-Seidel solver...\n";
		t1 = chrono::high_resolution_clock::now();

		Eigen::VectorXd xk1 = x0;
		Eigen::VectorXd xk2;
		double diffIter;
		double maxDiffNum, minDiffNum;
		double maxCurNum, minCurNum;
		int iterUsed = maxIter;
		double res;
		Eigen::VectorXd residu;

		residu = rhs - LHS * xk1;
		res = residu.transpose() * LHS * residu;
		cout << "residue: " << res << endl;

		for (int i = 0; i < maxIter; i++) {
			GaussSeidel1Iter(LHS, rhs, xk1, xk2);

			maxDiffNum = (xk2 - xk1).maxCoeff();
			minDiffNum = (xk2 - xk1).minCoeff();
			maxCurNum = xk2.maxCoeff();
			minCurNum = xk2.minCoeff();
			//printf("MaxDiff: %.4f | minDiff: %.4f | maxCur: %.4f | minCur: %.4f \n", maxDiffNum, minDiffNum, maxCurNum, minCurNum);
			diffIter = max(maxDiffNum, abs(minDiffNum)) / max(maxCurNum, abs(minCurNum));
			//diffIter = maxDiffNum / maxCurNum;

			//cout << "x: " << xk2.transpose() << endl;

			residu = rhs - LHS * xk2;
			//res = residu.transpose()*LHS*residu;
			res = residu.transpose() * M * residu;
			res = sqrt(res);
			//res = residu.norm();
			if (i % 1000 == 0) {
				//printf("______<%d> iter, the convergence: %.16f |", i, diffIter);			
				//printf(" residu: %.16f \n", res);
				cout << "______" << i << " iter | convergence: " << diffIter << " | residue: " << res << endl;
			}

			xk1 = xk2;

			if (res < tol)
				//if (diffIter < tol)
			{
				iterUsed = i + 1;
				break;
			}
		}
		x = xk2;

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		///cout << "in " << duration.count() << " seconds" << endl;
		//printf("____It takes %d iter, with %.16f tolerance \n", iterUsed, res);
		cout << "____" << iterUsed << " iterations | residue:" << res << ".\n";
	}

	void GaussSeidelSparseSolverMatrixForm(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x, int maxIter, double tol)
	{
		cout << "Setting up the L, D, and U matrices \n";
		/* Defining holder for upper, diagonal, and lower matrices of the LHS*/
		Eigen::SparseMatrix<double> U, D, L;
		U.resize(LHS.rows(), LHS.cols());
		D.resize(LHS.rows(), LHS.cols());
		L.resize(LHS.rows(), LHS.cols());
		vector<Eigen::Triplet<double>> UTriplet, DTriplet, LTriplet;
		UTriplet.reserve(4 * LHS.rows());
		DTriplet.reserve(LHS.rows());
		LTriplet.reserve(4 * LHS.rows());

		//Getting the sum of every non-zero elements in a row
		for (int k = 0; k < LHS.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(LHS, k); it; ++it) {
				if (it.row() > it.col())		LTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
				else if (it.row() < it.col())	UTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
				else if (it.row() == it.col())	DTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
			}
		}
		U.setFromTriplets(UTriplet.begin(), UTriplet.end());
		D.setFromTriplets(DTriplet.begin(), DTriplet.end());
		L.setFromTriplets(LTriplet.begin(), LTriplet.end());
		UTriplet.clear(); UTriplet.shrink_to_fit();
		DTriplet.clear(); DTriplet.shrink_to_fit();
		LTriplet.clear(); LTriplet.shrink_to_fit();

		printf("L: %dx%d (%d nnz) | D:%dx%d (%d nnz) | U:%dx%d (%d nnz) \n", L.rows(), L.cols(), L.nonZeros(), D.rows(), D.cols(), D.nonZeros(), U.rows(), U.cols(), U.nonZeros());

		Eigen::SparseMatrix<double> DL = D + L;
		//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> sparseSolver;
		Eigen::SparseLU<Eigen::SparseMatrix<double>> sparseSolver;
		sparseSolver.analyzePattern(DL);
		sparseSolver.factorize(DL);

		printf("Dimensionality check: \n");
		printf("__LHS: %dx%d | rhs: %d | init guess: %d \n", LHS.rows(), LHS.cols(), rhs.rows(), x0.rows());
		// For Timing
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;
		cout << "Iterative Gauss-Seidel solver...";
		t1 = chrono::high_resolution_clock::now();

		Eigen::VectorXd xk1 = x0;
		Eigen::VectorXd xk2;
		double diffIter;
		double maxDiffNum, minDiffNum;
		double maxCurNum, minCurNum;
		int iterUsed = maxIter;

		cout << "Iterating for solution \n";
		for (int i = 0; i < maxIter; i++) {
			GaussSeidel1IterMatrixForm(sparseSolver, U, rhs, xk1, xk2);
			//GaussSeidel1Iter(LHS, rhs, xk1, xk2);

			maxDiffNum = (xk2 - xk1).maxCoeff();
			minDiffNum = (xk2 - xk1).minCoeff();
			maxCurNum = xk2.maxCoeff();
			minCurNum = xk2.minCoeff();
			//printf("MaxDiff: %.4f | minDiff: %.4f | maxCur: %.4f | minCur: %.4f \n", maxDiffNum, minDiffNum, maxCurNum, minCurNum);
			//diffIter = max(maxDiffNum, abs(minDiffNum)) / max(maxCurNum, abs(minCurNum));
			diffIter = maxDiffNum / maxCurNum;

			//cout << "x: " << xk2.transpose() << endl;

			if (i % 10 == 0)
			{
				printf("%d iter, the convergence: %.16f |", i, diffIter);
				Eigen::VectorXd residu = rhs - LHS * xk2;
				double res = residu.transpose() * LHS * residu;
				printf(" reside: %.16f \n", res);
			}

			xk1 = xk2;

			if (diffIter < tol)
			{
				iterUsed = i + 1;
				break;
			}
		}
		x = xk2;

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;
		printf("It takes %d iter, with %.10f tolerance \n", iterUsed, diffIter);
	}

	void GaussSeidel1Iter(Eigen::SparseMatrix<double>& LHS, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x)
	{
		x = x0;

		/* Getting the sum of every non-zero elements in a row */
		for (int k = 0; k < LHS.outerSize(); ++k) {
			double l = 0.0;
			double u = 0.0;

			for (Eigen::SparseMatrix<double>::InnerIterator it(LHS, k); it; ++it)
			{
				if (it.row() < k) {
					l += it.value() * x(it.row());
				}
				else if (it.row() > k) {
					u += it.value() * x(it.row());
				}
			}
			x(k) = (rhs(k) - l - u) / LHS.coeff(k, k);
		}
	}

	void GaussSeidel1IterMatrixForm(const Eigen::SparseLU<Eigen::SparseMatrix<double>>& factor, const Eigen::SparseMatrix<double>& U, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x)
	{
		Eigen::VectorXd xk = rhs - (U * x0);
		x = factor.solve(xk);


		//Eigen::VectorXd comp1 = factor.solve(xk);
		//Eigen::VectorXd comp2 = factor.solve(rhs);
		//x = comp1 + comp2;	
	}

}