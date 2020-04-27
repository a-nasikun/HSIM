#include "EigenSolver.h"
#include <chrono>

#include <iostream>
#include <assert.h> 

#include <Eigen/PardisoSupport>

// For CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// For MATLAB
#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>
#include <tuple>
#include "engine.h"
#include "mex.h"

// Matlab interface
#include <igl/matlab/matlabinterface.h> 

// SPECTRA EIGENSOLVER
//#include <Spectra/GenEigsSolver.h>
//#include <Spectra/SymEigsSolver.h>
//#include <Spectra/SymEigsShiftSolver.h>
//#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>

/* Computing Eigenstructure in GPU */
/* This code is mostly taken from CUDA's cuSOLVER's example codes */
int computeEigenGPU(const Eigen::SparseMatrix<double> &S_, const Eigen::SparseMatrix<double> &M_, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal)
{
	cusolverDnHandle_t	cusolverH = NULL;
	cusolverStatus_t	cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t			cudaStat1 = cudaSuccess;
	cudaError_t			cudaStat2 = cudaSuccess;
	cudaError_t			cudaStat3 = cudaSuccess;
	cudaError_t			cudaStat4 = cudaSuccess;

	const int m = S_.rows();
	const int lda = m;

	double *A = (double*)std::malloc(m*lda * sizeof(double));
	double *B = (double*)std::malloc(m*lda * sizeof(double));
	Eigen::MatrixXd Snew(S_);
	Eigen::MatrixXd Mnew(M_);
	A = Snew.data();
	B = Mnew.data();

	double	*V = (double*)std::malloc(m*lda * sizeof(double)); // [lda*m];		// eigenvectors 
	double	*W = (double*)std::malloc(m * sizeof(double)); 			// eigenvalues 
	double	*d_A = NULL;
	double	*d_B = NULL;
	double	*d_W = NULL;
	int		*devInfo = NULL;
	double	*d_work = NULL;
	int		lwork = 0;
	int		info_gpu = 0;

	// step 1: create cusolver/cublas handle 
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device 
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc((void**)&d_B, sizeof(double) * lda * m);
	cudaStat3 = cudaMalloc((void**)&d_W, sizeof(double) * m);
	cudaStat4 = cudaMalloc((void**)&devInfo, sizeof(int));
	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0 || cudaStat4 !=0) {
		cout << "Error! Cannot allocate memory in GPU." << endl; 
	}	
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	
	cudaStat1			= cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	cudaStat2			= cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	if (cudaStat1 != 0 || cudaStat2 != 0) {
		cout << "Error! Cannot copy data to GPU." << endl;
	}

	// step 3: query working space of sygvd 
	cusolverEigType_t	itype = CUSOLVER_EIG_TYPE_1;		// A*x = (lambda)*B*x 
	cusolverEigMode_t	jobz = CUSOLVER_EIG_MODE_VECTOR;	// compute eigenvalues and eigenvectors. 
	cublasFillMode_t	uplo = CUBLAS_FILL_MODE_UPPER;
	cusolver_status = cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == cudaStat1);
	if (cudaStat1 != 0) {
		cout << "Error! Cannot request space to solve the eigenproblem." << endl;
	}

	// step 4: compute spectrum of (A,B) 
	cusolver_status = cusolverDnDsygvd(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, d_work, lwork, devInfo);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);
	if (cudaStat1 != 0 || cudaStat2 != 0) {
		cout << "Error! CUDA cuSolver's eigensolver FAILED to solve the eigenproblem." << endl;
	}

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0) {
		cout << "Error! Cannot copy data from GPU to CPU." << endl;
	}
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (info_gpu != 0) {
		cout << "Error! There's some error with sygvd." << endl;
	}
	assert(0 == info_gpu);

	// Copying back to EigenFormat (dense) for Eigenvalues and Eigenvectors
	LDEigVec = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(V, m, lda, Eigen::OuterStride<>(m));
	LDEigVal = Eigen::Map<Eigen::VectorXd>(W, m);

	// free resources 
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_W) cudaFree(d_W);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	if (cusolverH) cusolverDnDestroy(cusolverH);
	cudaDeviceReset();

	return ((int)cusolver_status || info_gpu || cudaStat1 || cudaStat2 || cudaStat3 || cudaStat4);
}

int computeEigenGPUDense(Eigen::MatrixXd &S, Eigen::MatrixXd &M, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal)
{
	cusolverDnHandle_t	cusolverH = NULL;
	cusolverStatus_t	cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t			cudaStat1 = cudaSuccess;
	cudaError_t			cudaStat2 = cudaSuccess;
	cudaError_t			cudaStat3 = cudaSuccess;
	cudaError_t			cudaStat4 = cudaSuccess;

	const int m = S.rows();
	const int lda = m;

	double *A = (double*)std::malloc(m*lda * sizeof(double));
	double *B = (double*)std::malloc(m*lda * sizeof(double));	
	A = S.data();
	B = M.data();

	double	*V = (double*)std::malloc(m*lda * sizeof(double)); // [lda*m];		// eigenvectors 
	double	*W = (double*)std::malloc(m * sizeof(double)); 			// eigenvalues 
	double	*d_A = NULL;
	double	*d_B = NULL;
	double	*d_W = NULL;
	int		*devInfo = NULL;
	double	*d_work = NULL;
	int		lwork = 0;
	int		info_gpu = 0;

	// step 1: create cusolver/cublas handle 
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device 
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc((void**)&d_B, sizeof(double) * lda * m);
	cudaStat3 = cudaMalloc((void**)&d_W, sizeof(double) * m);
	cudaStat4 = cudaMalloc((void**)&devInfo, sizeof(int));
	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0 || cudaStat4 != 0) {
		cout << "Error! Cannot allocate memory in GPU." << endl;
	}
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	if (cudaStat1 != 0 || cudaStat2 != 0) {
		cout << "Error! Cannot copy data to GPU." << endl;
	}

	// step 3: query working space of sygvd 
	cusolverEigType_t	itype = CUSOLVER_EIG_TYPE_1;		// A*x = (lambda)*B*x 
	cusolverEigMode_t	jobz = CUSOLVER_EIG_MODE_VECTOR;	// compute eigenvalues and eigenvectors. 
	cublasFillMode_t	uplo = CUBLAS_FILL_MODE_UPPER;
	cusolver_status = cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == cudaStat1);
	if (cudaStat1 != 0) {
		cout << "Error! Cannot request space to solve the eigenproblem." << endl;
	}

	// step 4: compute spectrum of (A,B) 
	///cout << "Prior to computation \n";
	///printf("__Lwork=%d |", lwork);
	///cout << " jobz: " << jobz << endl; 

	cusolver_status = cusolverDnDsygvd(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, d_work, lwork, devInfo);
	//cout << "Dev inf:" << devInfo << endl;
	//cout << "Dev inf:" << *devInfo << endl;

	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);
	if (cudaStat1 != 0 || cudaStat2 != 0) {
		cout << "Error! CUDA cuSolver's eigensolver FAILED to solve the eigenproblem." << endl;
	}
	else if(cudaStat1 == 0) {
		///cout << "Yeay! I can compute the eigenvalues!!!\n";
	}

	//cout << "DevInfo: " << (devInfo) << endl; 
	//printf("dev info(2) %d:", *devInfo);

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0) {
		cout << "Error! Cannot copy data from GPU to CPU." << endl;
	}
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (info_gpu != 0) {
		cout << info_gpu << "| Error! Cannot copy back to CPU sygvd." << endl;
	}
	assert(0 == info_gpu);

	// Copying back to EigenFormat (dense) for Eigenvalues and Eigenvectors
	LDEigVec = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(V, m, lda, Eigen::OuterStride<>(m));
	LDEigVal = Eigen::Map<Eigen::VectorXd>(W, m);

	// free resources 
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_W) cudaFree(d_W);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	if (cusolverH) cusolverDnDestroy(cusolverH);
	cudaDeviceReset();

	return ((int) cusolver_status || info_gpu || cudaStat1 || cudaStat2 || cudaStat3 || cudaStat4); 
}

///void computeEigenGPUDenseJacobi(Eigen::MatrixXd &S, Eigen::MatrixXd &M, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal)
///{
///	cusolverDnHandle_t	cusolverH = NULL;
///	cudaStream_t		stream = NULL;
///	syevjInfo_t			syevj_params = NULL;
///	cusolverStatus_t	cusolver_status = CUSOLVER_STATUS_SUCCESS;
///	cudaError_t			cudaStat1 = cudaSuccess;
///	cudaError_t			cudaStat2 = cudaSuccess;
///	cudaError_t			cudaStat3 = cudaSuccess;
///	cudaError_t			cudaStat4 = cudaSuccess;
///
///	const int m = S.rows();
///	const int lda = m;
///
///	double *A = (double*)std::malloc(m*lda * sizeof(double));
///	double *B = (double*)std::malloc(m*lda * sizeof(double));
///	A = S.data();
///	B = M.data();
///
///	double	*V = (double*)std::malloc(m*lda * sizeof(double)); // [lda*m];		// eigenvectors 
///	double	*W = (double*)std::malloc(m * sizeof(double)); 			// eigenvalues 
///	double	*d_A = NULL;
///	double	*d_B = NULL;
///	double	*d_W = NULL;
///	int		*devInfo = NULL;
///	double	*d_work = NULL;
///	int		lwork = 0;
///	int		info_gpu = 0;
///
///	// step 1: create cusolver/cublas handle 
///	cusolver_status = cusolverDnCreate(&cusolverH);
///	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
///
///	// step 2: copy A and B to device 
///	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
///	cudaStat2 = cudaMalloc((void**)&d_B, sizeof(double) * lda * m);
///	cudaStat3 = cudaMalloc((void**)&d_W, sizeof(double) * m);
///	cudaStat4 = cudaMalloc((void**)&devInfo, sizeof(int));
///	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0 || cudaStat4 != 0) {
///		cout << "Error! Cannot allocate memory in GPU." << endl;
///	}
///	assert(cudaSuccess == cudaStat1);
///	assert(cudaSuccess == cudaStat2);
///	assert(cudaSuccess == cudaStat3);
///	assert(cudaSuccess == cudaStat4);
///
///	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
///	cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
///	assert(cudaSuccess == cudaStat1);
///	assert(cudaSuccess == cudaStat2);
///	if (cudaStat1 != 0 || cudaStat2 != 0) {
///		cout << "Error! Cannot copy data to GPU." << endl;
///	}
///
///	// configuring the jacobi 
///	const double tol = 1.e-7;
///	const int max_sweeps = 15;
///	double residual = 0;
///	int executed_sweeps = 0;
///
///	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
///	assert(cudaSuccess == cudaStat1);
///
///	cusolver_status = cusolverDnSetStream(cusolverH, stream);
///	assert(CUSOLVER_STATUS_SUCCESS == status);
///
///	/* step 2: configuration of syevj */
///	cusolver_status = cusolverDnCreateSyevjInfo(&syevj_params);
///	assert(CUSOLVER_STATUS_SUCCESS == status);
///
///	/* default value of tolerance is machine zero */
///	cusolver_status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
///	assert(CUSOLVER_STATUS_SUCCESS == status);
///
///	/* default value of max. sweeps is 100 */
///	cusolver_status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
///	assert(CUSOLVER_STATUS_SUCCESS == status);
///
///	// step 3: query working space of sygvj 
///	cusolverEigType_t	itype = CUSOLVER_EIG_TYPE_1;		// A*x = (lambda)*B*x 
///	cusolverEigMode_t	jobz = CUSOLVER_EIG_MODE_VECTOR;	// compute eigenvalues and eigenvectors. 
///	cublasFillMode_t	uplo = CUBLAS_FILL_MODE_UPPER;
/// ///<<<<<<< HEAD
/// ///	cusolver_status = cusolverDnDsygvj_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, &lwork);
/// ///=======
///	cusolver_status = cusolverDnDsygvj_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, &lwork, syevj_params);
/// ///>>>>>>> 1cc065c4333e22e0cb956b8e9d5cff4985043e48
///	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
///	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
///	assert(cudaSuccess == cudaStat1);
///	if (cudaStat1 != 0) {
///		cout << "Error! Cannot request space to solve the eigenproblem." << endl;
///	}
/// ///<<<<<<< HEAD
/// ///=======
///
///	// step 4: compute spectrum of (A,B) 
///	cout << "Prior to computation \n";
///	printf("__Lwork=%d |", lwork);
///	cout << " jobz: " << jobz << endl;
///
///	
///	//cusolver_status = cusolverDnDsygvj(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, d_work, lwork, devInfo);
///	cusolver_status = cusolverDnDsygvj(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, d_work, lwork, devInfo, syevj_params);
///
///	//cout << "Dev inf:" << devInfo << endl;
///	//cout << "Dev inf:" << *devInfo << endl;
/// ///>>>>>>> 1cc065c4333e22e0cb956b8e9d5cff4985043e48
///
///	cudaStat1 = cudaDeviceSynchronize();
///	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
///	assert(cudaSuccess == cudaStat1);
///	if (cudaStat1 != 0 || cudaStat2 != 0) {
///		cout << "Error! CUDA cuSolver's eigensolver FAILED to solve the eigenproblem." << endl;
///	}
///	else if (cudaStat1 == 0) {
///		cout << "Yeay! I can compute the eigenvalues!!!\n";
///	}
///
///	//cout << "DevInfo: " << (devInfo) << endl; 
///	//printf("dev info(2) %d:", *devInfo);
///
///	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
///	cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
///	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
///	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0) {
///		cout << "Error! Cannot copy data from GPU to CPU." << endl;
///	}
///	assert(cudaSuccess == cudaStat1);
///	assert(cudaSuccess == cudaStat2);
///	assert(cudaSuccess == cudaStat3);
///
///	if (info_gpu != 0) {
///		cout << info_gpu << "| Error! Cannot copy back to CPU sygvd." << endl;
///	}
///	assert(0 == info_gpu);
///
///	// Copying back to EigenFormat (dense) for Eigenvalues and Eigenvectors
///	LDEigVec = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(V, m, lda, Eigen::OuterStride<>(m));
///	LDEigVal = Eigen::Map<Eigen::VectorXd>(W, m);
///
///	// free resources 
///	if (d_A) cudaFree(d_A);
///	if (d_B) cudaFree(d_B);
///	if (d_W) cudaFree(d_W);
///	if (devInfo) cudaFree(devInfo);
///	if (d_work) cudaFree(d_work);
///	if (cusolverH) cusolverDnDestroy(cusolverH);
///	cudaDeviceReset();
///}

void computeEigenGPUDense2(Eigen::MatrixXd &S, Eigen::MatrixXd &M, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal)
{
	cusolverDnHandle_t	cusolverH = NULL;
	cusolverStatus_t	cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t			cudaStat1 = cudaSuccess;
	cudaError_t			cudaStat2 = cudaSuccess;
	cudaError_t			cudaStat3 = cudaSuccess;
	cudaError_t			cudaStat4 = cudaSuccess;

	const int m = S.rows();
	const int lda = m;

	double *A = (double*)std::malloc(m*lda * sizeof(double));
	double *B = (double*)std::malloc(m*lda * sizeof(double));
	A = S.data();
	B = M.data();

	double	*V = (double*)std::malloc(m*lda * sizeof(double)); // [lda*m];		// eigenvectors 
	double	*W = (double*)std::malloc(m * sizeof(double)); 			// eigenvalues 
	double	*d_A = NULL;
	double	*d_B = NULL;
	double	*d_W = NULL;
	int		*devInfo = NULL;
	double	*d_work = NULL;
	int		lwork = 0;
	int		info_gpu = 0;

	// step 1: create cusolver/cublas handle 
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device 
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc((void**)&d_B, sizeof(double) * lda * m);
	cudaStat3 = cudaMalloc((void**)&d_W, sizeof(double) * m);
	cudaStat4 = cudaMalloc((void**)&devInfo, sizeof(int));
	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0 || cudaStat4 != 0) {
		cout << "Error! Cannot allocate memory in GPU." << endl;
	}
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	if (cudaStat1 != 0 || cudaStat2 != 0) {
		cout << "Error! Cannot copy data to GPU." << endl;
	}

	// step 3: query working space of sygvd 
	cusolverEigType_t	itype = CUSOLVER_EIG_TYPE_1;		// A*x = (lambda)*B*x 
	cusolverEigMode_t	jobz = CUSOLVER_EIG_MODE_VECTOR;	// compute eigenvalues and eigenvectors. 
	cublasFillMode_t	uplo = CUBLAS_FILL_MODE_UPPER;
	cusolver_status = cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == cudaStat1);
	if (cudaStat1 != 0) {
		cout << "Error! Cannot request space to solve the eigenproblem." << endl;
	}

	// step 4: compute spectrum of (A,B) 
	cout << "Prior to computation \n";
	printf("__Lwork=%d |", lwork);
	cout << " jobz: " << jobz << endl;

	cusolver_status = cusolverDnDsygvd(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, lda, d_W, d_work, lwork, devInfo);
	cout << "Dev inf:" << devInfo << endl;
	cout << "Dev inf:" << *devInfo << endl;

	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);
	if (cudaStat1 != 0 || cudaStat2 != 0) {
		cout << "Error! CUDA cuSolver's eigensolver FAILED to solve the eigenproblem." << endl;
	}
	else if (cudaStat1 == 0) {
		cout << "Yeay! I can compute the eigenvalues!!!\n";
	}

	//cout << "DevInfo: " << (devInfo) << endl; 
	//printf("dev info(2) %d:", *devInfo);

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStat1 != 0 || cudaStat2 != 0 || cudaStat3 != 0) {
		cout << "Error! Cannot copy data from GPU to CPU." << endl;
	}
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (info_gpu != 0) {
		cout << "Error! Cannot copy back to CPU sygvd." << endl;
	}
	assert(0 == info_gpu);

	// Copying back to EigenFormat (dense) for Eigenvalues and Eigenvectors
	LDEigVec = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(V, m, lda, Eigen::OuterStride<>(m));
	LDEigVal = Eigen::Map<Eigen::VectorXd>(W, m);

	// free resources 
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_W) cudaFree(d_W);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	if (cusolverH) cusolverDnDestroy(cusolverH);
	cudaDeviceReset();
}

void computeEigenMatlab(Eigen::SparseMatrix<double> &S, Eigen::SparseMatrix<double> &M, const int& numEigs, Eigen::MatrixXd &EigVec, Eigen::VectorXd &EigVal, const string& filename)
{
	//printf("Size of S = %dx%d\n", S.rows(), S.cols());
	using namespace matlab::engine;
	Engine *ep;
	mxArray *MM = NULL, *MS = NULL, *result = NULL, *eigVecResult, *nEigsBuff;

	/* Storing element to determine how many eigenpairs to compute*/
	if (numEigs > M.rows())
	{
		cout << "ERROR! The number of the eigenpairs cannot exceed the size of input matrix(-ces)." << endl;
		return;
	}
	nEigsBuff = mxCreateDoubleMatrix((mwSize)1, (mwSize)1, mxREAL);
	double *numEigsPtr = mxGetPr(nEigsBuff);
	*numEigsPtr = numEigs;

	chrono::high_resolution_clock::time_point	t1, t2, t3, t4;
	chrono::duration<double>					time_span, ts2;

	t1 = chrono::high_resolution_clock::now();

	const int NNZ_S = S.nonZeros();
	const int NNZ_M = M.nonZeros();
	double *eigVal, *eigVec;

	// Allocate memory for S and M (sparse representation)
	double	*srs = (double*)malloc(NNZ_S * sizeof(double));
	mwIndex *irs = (mwIndex*)malloc(NNZ_S * sizeof(mwIndex));
	mwIndex *jcs = (mwIndex*)malloc((S.cols() + 1) * sizeof(mwIndex));

	double	*srm = (double*)malloc(NNZ_M * sizeof(double));
	mwIndex *irm = (mwIndex*)malloc(NNZ_M * sizeof(mwIndex));
	mwIndex *jcm = (mwIndex*)malloc((M.cols() + 1) * sizeof(mwIndex));

	// Bind MM with M, and MS with S
	MS = mxCreateSparse(S.rows(), S.cols(), NNZ_S, mxREAL);
	srs = mxGetPr(MS);
	irs = mxGetIr(MS);
	jcs = mxGetJc(MS);

	MM = mxCreateSparse(M.rows(), M.cols(), NNZ_M, mxREAL);
	srm = mxGetPr(MM);
	irm = mxGetIr(MM);
	jcm = mxGetJc(MM);

	// Setting initial variable value
	int nnzSCounter = 0;
	int nnzMCounter = 0;



	// Getting matrix S
	jcs[0] = nnzSCounter;
	for (int i = 0; i < S.outerSize(); i++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(S, i); it; ++it) {
			srs[nnzSCounter] = it.value();
			irs[nnzSCounter] = it.row();
			nnzSCounter++;
		}
		jcs[i + 1] = nnzSCounter;
	}

	// Getting matrix M
	jcm[0] = nnzMCounter;
	for (int i = 0; i < M.outerSize(); i++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it) {
			srm[nnzMCounter] = it.value();
			irm[nnzMCounter] = it.row();
			nnzMCounter++;
		}
		jcm[i + 1] = nnzMCounter;
	}

	// Start Matlab Engine
	int *matlabStatus;
	ep = engOpen(NULL);
	if (!(ep = engOpen(""))) {
		//ep = engOpenSingleUse(NULL, NULL, matlabStatus);
		//if (!(ep = engOpenSingleUse(NULL, NULL, matlabStatus))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		cout << "CANNOT START MATLAB " << endl;
	}
	else {
		//cout << "MATLAB STARTS. OH YEAH!!!" << endl;
	}

	//cout << "Status => " << matlabStatus << endl; 

	// Compute Eigenvalue in Matlab
	int NUM_EIGEN = numEigs;

	engPutVariable(ep, "MS", MS);
	engPutVariable(ep, "MM", MM);
	engPutVariable(ep, "Num", nEigsBuff);

	t3 = chrono::high_resolution_clock::now();
	engEvalString(ep, "[data, EigVal]=eigs(MS,MM, Num(1,1),'smallestabs');");
	//engEvalString(ep, "[EigVec, EigVal]=eigs(MS,MM);");
	engEvalString(ep, "EigVal=diag(EigVal);");
	if (numEigs > 1)
	{
		engEvalString(ep, "hold on; plot(1:Num(1,1), EigVal(1:Num(1,1)),'LineWidth',1.5);"); // has to do it this way for "correct" plot		
		string approxFile = "save('" + filename + "_eigFields','data','EigVal');";
		//string approxFile = "save('" + filename + "_eigFields','EigVec');";
		//string approxFile = "save('" + filename + "_eigvalues','EigVal');";
		//cout << "Saving the eigen problem\n";
		//engEvalString(ep, approxFile.c_str());
	}
	t4 = chrono::high_resolution_clock::now();


	result = engGetVariable(ep, "EigVal");
	eigVal = (double*)malloc(NUM_EIGEN * sizeof(double));
	memcpy((void *)eigVal, (void *)mxGetPr(result), NUM_EIGEN * sizeof(double));

	eigVecResult = engGetVariable(ep, "data");
	eigVec = (double*)malloc(M.rows() * NUM_EIGEN * sizeof(double));
	memcpy((void *)eigVec, (void *)mxGetPr(eigVecResult), M.rows() * NUM_EIGEN * sizeof(double));

	EigVal = Eigen::Map<Eigen::VectorXd>(eigVal, NUM_EIGEN);
	EigVec = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(eigVec, M.rows(), NUM_EIGEN, Eigen::OuterStride<>(M.rows()));

	t2 = chrono::high_resolution_clock::now();
	time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	ts2 = chrono::duration_cast<chrono::duration<double>>(t4 - t3);

	engEvalString(ep, "clear;");

	mxDestroyArray(MM);
	mxDestroyArray(MS);
	mxDestroyArray(result);
	mxDestroyArray(eigVecResult);
	mxDestroyArray(nEigsBuff);

	engClose(ep);
}

void computeEigenMatlabDense(Eigen::MatrixXd &S, Eigen::MatrixXd &M, const int& numEigs, Eigen::MatrixXd &EigVec, Eigen::VectorXd &EigVal, const string& filename)
{
	/* Initiating the matlab engine */
	cout << "Initiating the matlab engine\n";
	using namespace matlab::engine;
	Engine *ep;
	igl::matlab::mlinit(&ep);

	/* Setting the matrix to matlab */
	cout << "Setting the matrix to matlab\n";
	igl::matlab::mlsetmatrix(&ep, "S", S);
	igl::matlab::mlsetmatrix(&ep, "M", M);
	igl::matlab::mlsetscalar(&ep, "numEigen", numEigs);

	/* Performing the eigenvalue computation */
	cout << "Performing the eigenvalue computation \n";
	igl::matlab::mleval(&ep, "[EVec, EVal]=eig(S,M,numEigen,'smallestreal');");
	igl::matlab::mleval(&ep, "EVal=diag(EVal);");
	igl::matlab::mleval(&ep, "plot(EVal);");

	cout << "Getting back the matrices\n";
	/* Getting back the matrices */
	igl::matlab::mlgetmatrix(&ep, "EVec", EigVec);
	Eigen::MatrixXd EValMat;
	igl::matlab::mlgetmatrix(&ep, "EVal", EValMat);
	EigVal = EValMat.col(0);
}

void computeEigenSpectraDense(Eigen::MatrixXd &S, Eigen::MatrixXd &M, const int& numEigs, Eigen::MatrixXd &EigVec, Eigen::VectorXd &EigVal)
{

	/* fake mass matrix */
	Eigen::SparseMatrix<double> M2;
	M2.resize(S.rows(), S.cols());
	M2.setIdentity();

	cout << "Describe operator \n";
	Spectra::DenseSymMatProd<double> Sop(S);
	Spectra::DenseCholesky<double> Mop(M);
	Spectra::SparseCholesky<double> Mop2(M2);

	cout << "Creating the solver \n";
	Spectra::SymGEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<double>, Spectra::DenseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigensolver(&Sop, &Mop, 50, 2 * 50);
	//Spectra::SymGEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigensolver(&Sop, &Mop2, 50, 2 * 50);

	cout << "Initiating the eigenproblem \n";
	eigensolver.init();

	cout << "Computing the eigenproblem \n";
	int nconv = eigensolver.compute();
	if (eigensolver.info() == Spectra::SUCCESSFUL) {
		cout << "Successful! \n";
	}
	else {
		cout << "Failsedl! \n";
	}


	EigVec = eigensolver.eigenvectors();
	EigVal = eigensolver.eigenvalues();
	cout << "Eigenvalues: " << EigVal.transpose() << endl; 
}

/* Sparse spectra eigensolver */
void computeEigenSpectraSparse(Eigen::SparseMatrix<double> &S, Eigen::SparseMatrix<double> &M, const int& numEigs, Eigen::MatrixXd &EigVec, Eigen::VectorXd &EigVal, int type, int numIters, double tol)
{
	///cout << "Computing eigenproblem in Spectra...";
	///chrono::high_resolution_clock::time_point	t1, t2;
	///chrono::duration<double>					duration;

	printf("M: %dx%d | S:%dx%d \n", M.rows(), M.cols(), S.rows(), S.cols());

	Spectra::SparseSymMatProd<double> SOp(S);
	Spectra::SparseCholesky<double> MOp(M);

	printf("Wrapper \n");

	map<int, int> typeMap;
	typeMap.insert(std::pair<int, int>(0, Spectra::SMALLEST_ALGE));
	typeMap.insert(std::pair<int, int>(1, Spectra::SMALLEST_MAGN));
	const int paramType = typeMap[type];

	Spectra::SymGEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> spectraSolver(&SOp, &MOp, numEigs, 2 * numEigs);
	

	cout << "Trying to initiate \n";
	spectraSolver.init();
	std::cout << "Trying to solve | " << numEigs << " eigenpairs |" << numIters << "max iters |" << tol << " tolerance \n";
	int info = spectraSolver.compute(numIters, tol, typeMap[type]);
	cout << "Result of solving: " << info << endl; 

	if (spectraSolver.info() == Spectra::SUCCESSFUL) {
		//EigVal = spectraSolver.eigenvalues().reverse();
		//EigVec = spectraSolver.eigenvectors().rowwise().reverse();
		EigVal = spectraSolver.eigenvalues();
		EigVec = spectraSolver.eigenvectors();
		cout << "Yes! Made it! \n";
	}
	else {
		cout << "Computation failed \n";
	}

	///t2 = chrono::high_resolution_clock::now();
	///duration = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	///cout << "in " << duration.count()/1000 << " seconds." << endl;
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

	residu = rhs - LHS*xk1;
	res = residu.transpose()*LHS*residu;
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

		
		residu = rhs - LHS*xk2;
		res = residu.transpose()*LHS*residu;
		//res = residu.norm();
		if (i % 1000 == 0){
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

	residu = rhs - LHS*xk1;
	res = residu.transpose()*LHS*residu;
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

		residu = rhs - LHS*xk2;
		//res = residu.transpose()*LHS*residu;
		res = residu.transpose()*M*residu;
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
	UTriplet.reserve(4*LHS.rows());
	DTriplet.reserve(LHS.rows());
	LTriplet.reserve(4*LHS.rows());

	/* Getting the sum of every non-zero elements in a row */
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
			Eigen::VectorXd residu = rhs - LHS*xk2;
			double res = residu.transpose()*LHS*residu;
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
			if (it.row() < k){
				l += it.value()*x(it.row());
			} else if (it.row() > k){
				u += it.value()*x(it.row());
			}			
		}
		x(k) = (rhs(k)-l-u) / LHS.coeff(k, k);
	}
}

void GaussSeidel1IterMatrixForm(const Eigen::SparseLU<Eigen::SparseMatrix<double>>& factor, const Eigen::SparseMatrix<double>& U, Eigen::VectorXd& rhs, Eigen::VectorXd& x0, Eigen::VectorXd& x)
{
	Eigen::VectorXd xk = rhs - (U*x0) ;
	x = factor.solve(xk);


	//Eigen::VectorXd comp1 = factor.solve(xk);
	//Eigen::VectorXd comp2 = factor.solve(rhs);
	//x = comp1 + comp2;	
}