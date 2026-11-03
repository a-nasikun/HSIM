// cuSOLVER backend of hsim::solveDenseGeneralized (compiled only with HSIM_WITH_CUDA).
#include "hsim/dense_eigensolver.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <stdexcept>
#include <string>

namespace hsim::detail {

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

// Generalized symmetric-definite eigenproblem with cusolverDnDsygvd (adapted from the cuSOLVER examples).
DenseEigenpairs solveDenseCuSolver(const Eigen::MatrixXd& S, const Eigen::MatrixXd& M)
{
	const int m = static_cast<int>(S.rows());
	const int lda = m;
	const size_t numEntries = static_cast<size_t>(lda) * static_cast<size_t>(m);

	CusolverDnHandle handle;
	DeviceBuffer<double> dA(numEntries), dB(numEntries), dW(static_cast<size_t>(m));
	DeviceBuffer<int> dInfo(1);

	checkCuda(cudaMemcpy(dA.get(), S.data(), sizeof(double) * numEntries, cudaMemcpyHostToDevice), "cudaMemcpy (S)");
	checkCuda(cudaMemcpy(dB.get(), M.data(), sizeof(double) * numEntries, cudaMemcpyHostToDevice), "cudaMemcpy (M)");

	const cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;     // A x = lambda B x
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // eigenvalues and eigenvectors
	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	int lwork = 0;
	checkCusolver(cusolverDnDsygvd_bufferSize(handle.get(), itype, jobz, uplo, m, dA.get(), lda, dB.get(), lda, dW.get(), &lwork),
		"cusolverDnDsygvd_bufferSize");
	DeviceBuffer<double> dWork(static_cast<size_t>(lwork));
	checkCusolver(cusolverDnDsygvd(handle.get(), itype, jobz, uplo, m, dA.get(), lda, dB.get(), lda, dW.get(), dWork.get(), lwork, dInfo.get()),
		"cusolverDnDsygvd");

	DenseEigenpairs result;
	int info = 0;
	result.vectors.resize(m, m);
	result.values.resize(m);
	checkCuda(cudaMemcpy(result.vectors.data(), dA.get(), sizeof(double) * numEntries, cudaMemcpyDeviceToHost), "cudaMemcpy (eigenvectors)");
	checkCuda(cudaMemcpy(result.values.data(), dW.get(), sizeof(double) * static_cast<size_t>(m), cudaMemcpyDeviceToHost), "cudaMemcpy (eigenvalues)");
	checkCuda(cudaMemcpy(&info, dInfo.get(), sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy (info)");

	result.success = (info == 0) && result.values.allFinite() && result.vectors.allFinite();
	return result;
}

} // namespace hsim::detail
