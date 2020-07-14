#include "MLEigenSolver.h"
#include <Eigen/PardisoSupport>
#include <omp.h>
#include <random>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>
#include <cusparse_v2.h>

namespace MuLES
{
	
	/**
	*  @brief		Read a (triangle) mesh specified by the input file. 
	*  @details
	*   Currently it supports 'obj' and 'off' file format only (could be extended to other type format, e.g. .ply, etc.).
	*  @param meshFile  Location of the mesh
	*/
	void MLEigenSolver::readMesh(const string &meshFile)
	{
		V.resize(0, 0);
		F.resize(0, 0);

		if (meshFile.substr(meshFile.find_last_of(".") + 1) == "off") {
			igl::readOFF(meshFile, V, F);
			if (V.rows() == 0 || V.cols() == 0) {
				std::cout << "Error! It cannot be an empty mesh." << std::endl;
				return;
			}
			printf("A new mesh with %d vertices (%d faces) is loaded.\n", V.rows(), F.rows());
		}
		else if (meshFile.substr(meshFile.find_last_of(".") + 1) == "obj") {
			igl::readOBJ(meshFile, V, F);
			if (V.rows() == 0 || V.cols() == 0) {
				cout << "Error! It cannot be an empty mesh." << endl;
				return;
			}
			printf("A new mesh with %d vertices (%d faces) is loaded.\n", V.rows(), F.rows());
		}
		else {
			cout << "Error! File type can be either .OFF or .OBJ only." << endl;
			return;
		}
	}

	/**
	*  @brief
	*   Set up the system to have required levels.
	*  @param n Number of specified levels of the system.
	*/
	void MLEigenSolver::setInitialSystem(int n)
	{
		level = n;
		maxNeighDist.resize(level);
		U.resize(level);
		Sbar.resize(level);
		Mbar.resize(level);
		SubspaceSize.resize(level);
		geoDist.resize(level);
		clusters.resize(level);
	}

	/**
	*  @brief
	*   Set up the vertex-vertex adjacency matrix, generally 3 neighbors for triangle mesh (or 2 on the boundary).
	*/
	void MLEigenSolver::initiateAdM()
	{
		cout << "> Creating neighborhood structure of the mesh. \n";
		AdM.clear();
		AdM.resize(V.rows());

		for (int i = 0; i < F.rows(); i++)
		{
			for (int j = 0; j < F.cols(); j++) {
				/* vertex-to-vertex neighborhood */
				for (int k = j + 1; k < F.cols(); k++) {
					//TODO: Verify that the robustness
					AdM[F(i, j)].insert(F(i, k));
					AdM[F(i, k)].insert(F(i, j));
				}
			}
		}

	}

	/**
	*  @brief
	*   Set-up the corresponding mass matrix M.
	*  @details
	*	For Laplace--Beltrami eigenproblem, 
	*	we generally use the lumped mass matrix (i.e. diagonal mass matrix).
	*  @see	MuLES::MLSolverFunctions::constructMassMatrix()
	*/
	void MLEigenSolver::constructMassMatrix() {}
	
	/**
	*  @brief	Set-up the corresponding stiffness matrix S. 
	*  @details
	*	The widely adopted Laplacian option for discretization of Laplace--Beltrami operator in geometry processing is the cotangent weight.
	*/
	void MLEigenSolver::constructStiffnessMatrix() {}

	/**
	*  @brief	Set-up level of the hierarchy description
	*  @details
	*	It specifies the number of samples (degree of freedom) on each level of the hierarcy.
	*   The number of levels is determined by (1) the number of vertices of the mesh, 
	*   (2) the number of sought eigenpairs, and (3) the decay rate/ratio between one level to the other.
	*  @param	sampSize	[Return] A vector specifying the number of degree of freedom on each level.
	*  @param	numEigen	The number of sought eigenvalues and corresponding eigenvectors.
	*  @param	levelRatio	The decay rate stating how the DoFs decreases from one level to the next level.
	*/
	void MLEigenSolver::setSamplingLevelsConstantRate(Eigen::VectorXi& sampSize, int numEigen, double levelRatio)
	{
		double	spRatio = 1.0;
		int		sizeSubsp =std::max(1000, 2*numEigen);

		vector<int> sampleLevels;
		sampleLevels.reserve(10);
		//int SAMPLE_CUT_OFF = max(100, sizeSubsp);			//500
		int SAMPLE_CUT_OFF = (sizeSubsp);			//500
		int SAMPLE_LOW_BOUND = 2 * SAMPLE_CUT_OFF;
		int numSample = V.rows();
		std::cout << "cut-out" << SAMPLE_CUT_OFF << "| Low bound " << SAMPLE_LOW_BOUND << ": " << std::endl;

		/* automated */
		do {
			numSample = (int)round(numSample / levelRatio);
			if (numSample < SAMPLE_CUT_OFF) break;
			sampleLevels.push_back(numSample);
			std::cout << "Inserted sample: " << numSample << std::endl;
		} while (numSample > SAMPLE_LOW_BOUND);

		if (sampleLevels.size() < 1)
		{
			std::cout << "Error! The provided sample is not enough! \n";
			return;
		}

		sampSize.resize(sampleLevels.size());
		for (int ii = 0; ii < sampleLevels.size(); ++ii)
			sampSize(ii) = sampleLevels[ii];
	}

	void MLEigenSolver::setSamplingLevelsAuto(Eigen::VectorXi& sampSize, int numEigen, double levelRatio)
	{
		if (numEigen > V.rows()) {
			std::cout << "Error! The number of sought eigenvalues should not exceed the number of degree of freedom of the geometry.\n";
			return;
		}
		if (V.rows() < 2000) {
			sampSize.resize(1);
			sampSize(0) = V.rows();
			std::cout << "Since the number of vertices is relatively small, the system will attempt solving it directly in the GPU. \n";
			return;
		}

		const double EXP_RAT = levelRatio;
		int numVert = V.rows();
		int dimSubSpace = max(2*numEigen, 1000);
		double ratInit = (double)numVert / (double)dimSubSpace;
		double ratio = log(ratInit)/log(EXP_RAT);
		std::cout << "The ratio of vert to subspace is " << ratInit << " yielding " << ratio << " ratio setup \n";

		if (ratio < 1.0) {
			sampSize.resize(1);
			sampSize(0) = dimSubSpace;
			std::cout << "The ratio is " << ratio << " thus we'll just have 1 level with " << sampSize(0) << " DoFs\n";
		}
		else {
			int numLevels = (int) floor(ratio);
			double remainder = ratio - numLevels;
			std::cout << "Number of levels: " << numLevels << "+1, with remainder of " << remainder << std::endl;

			// When the remainder is high, add another level for the coarsest level
			if (remainder >= 0.8) {
				numLevels += 1;
				sampSize.resize(numLevels);
				std::cout << "CASE 1: remainder >= 0.8 \n";
				for (int i = 0; i < (numLevels - 1); ++i) {
					sampSize(i) = (int)ceil((double)numVert / pow(EXP_RAT,i+1));
				}
				sampSize(numLevels - 1) = (int)ceil((double)sampSize(numLevels-2) / pow(EXP_RAT, remainder));
			}
			// When the remainder is small, attach it to the level next to finest level
			else if (remainder < 0.2) {
				std::cout << "CASE 3: remainder < 0.2 \n";
				sampSize.resize(numLevels);
				sampSize(0) = (int)ceil((double)numVert / pow(EXP_RAT, 1+remainder));
				for (int i = 1; i < numLevels; ++i) {
					sampSize(i) = (int)ceil((double)sampSize(i - 1) / EXP_RAT);
				}
			}
			else {
				std::cout << "CASE 2: 0.2 <= remainder < 0.8 \n";
				Eigen::VectorXd addRatio(numLevels);
				double sumRatio = 0.0;
				for (int i = 0; i < numLevels; ++i) {
					addRatio(i) = pow(2.0, numLevels-i-1);
					sumRatio += addRatio(i);
				}

				sampSize.resize(numLevels);
				sampSize(0) = (int)ceil((double)numVert / pow(EXP_RAT, 1.0+ addRatio(0)*remainder/sumRatio));
				for (int i = 1; i < numLevels; ++i) {
					sampSize(i) = (int)ceil((double)sampSize(i-1) / pow(EXP_RAT, 1.0+addRatio(i)*remainder/sumRatio));
				}
			}
		}

		printf("Level %d has %d entries \n", 0, numVert);
		for (int i = 0; i < sampSize.rows(); ++i) {
			printf("Level %d has %d entries \n", i+1, sampSize(i));
		}
	}

	void MLEigenSolver::setSamplingLevelsSameRate(Eigen::VectorXi& sampSize, int numEigen, double levelRatio)
	{
		if (numEigen > V.rows()) {
			std::cout << "Error! The number of sought eigenvalues should not exceed the number of degree of freedom of the geometry.\n";
			return;
		}
		if (V.rows() < 2000) {
			sampSize.resize(1);
			sampSize(0) = V.rows();
			std::cout << "Since the number of vertices is relatively small, the system will attempt solving it directly in the GPU. \n";
			return;
		}

		const double EXP_RAT = levelRatio;
		int numVert = V.rows();
		int dimSubSpace = max(2 * numEigen, 1000);
		double ratInit = (double)numVert / (double)dimSubSpace;
		double ratio = log(ratInit) / log(EXP_RAT);
		std::cout << "The ratio of vert to subspace is " << ratInit << " yielding " << ratio << " ratio setup \n";

		if (ratio < 1.0) {
			sampSize.resize(1);
			sampSize(0) = dimSubSpace;
			std::cout << "The ratio is " << ratio << " thus we'll just have 1 level with " << sampSize(0) << " DoFs\n";
		}
		else {
			int numLevels = (int)floor(ratio);
			sampSize.resize(numLevels);
			double levelRatio = ratio / (double)numLevels;
			sampSize(0) = (int)ceil((double)numVert / pow(EXP_RAT, levelRatio));

			for (int i = 1; i < numLevels; ++i) {
				sampSize(i) = (int)ceil((double)sampSize(i-1)/ pow(EXP_RAT, levelRatio));
			}

			//double remainder = ratio - numLevels;
			//std::cout << "Number of levels: " << numLevels << "+1, with remainder of " << remainder << std::endl;



			// When the remainder is high, add another level for the coarsest level
			//if (remainder >= 0.8) {
			//	numLevels += 1;
			//	sampSize.resize(numLevels);
			//	std::cout << "CASE 1: remainder >= 0.8 \n";
			//	for (int i = 0; i < (numLevels - 1); ++i) {
			//		sampSize(i) = (int)ceil((double)numVert / pow(EXP_RAT, i + 1));
			//	}
			//	sampSize(numLevels - 1) = (int)ceil((double)sampSize(numLevels - 2) / pow(EXP_RAT, remainder));
			//}
			//// When the remainder is small, attach it to the level next to finest level
			//else if (remainder < 0.2) {
			//	std::cout << "CASE 3: remainder < 0.2 \n";
			//	sampSize.resize(numLevels);
			//	sampSize(0) = (int)ceil((double)numVert / pow(EXP_RAT, 1 + remainder));
			//	for (int i = 1; i < numLevels; ++i) {
			//		sampSize(i) = (int)ceil((double)sampSize(i - 1) / EXP_RAT);
			//	}
			//}
			//else {
			//	std::cout << "CASE 2: 0.2 <= remainder < 0.8 \n";
			//	Eigen::VectorXd addRatio(numLevels);
			//	double sumRatio = 0.0;
			//	for (int i = 0; i < numLevels; ++i) {
			//		addRatio(i) = pow(2.0, numLevels - i - 1);
			//		sumRatio += addRatio(i);
			//	}
			//
			//	sampSize.resize(numLevels);
			//	sampSize(0) = (int)ceil((double)numVert / pow(EXP_RAT, 1.0 + addRatio(0) * remainder / sumRatio));
			//	for (int i = 1; i < numLevels; ++i) {
			//		sampSize(i) = (int)ceil((double)sampSize(i - 1) / pow(EXP_RAT, 1.0 + addRatio(i) * remainder / sumRatio));
			//	}
			//}
		}

		printf("Level %d has %d entries \n", 0, numVert);
		for (int i = 0; i < sampSize.rows(); ++i) {
			printf("Level %d has %d entries \n", i + 1, sampSize(i));
		}
	}
	
	/**
	*  @brief	 Construct the number of samples in every level via farthest point sampling. 
	*  @details	
	*   Starting from the coarsest level, it computes the the first set of sampled DoFs (i.e. vertices).
	*   Then for every level, it adds more samples according the 
	*  @see MuLES::constructFarthestPointSample() \n MuLES::MLEigenSolver::setSamplingLevels() 
	*  @param	samples	A constact vector stating the number of samples on every level of the hierarchy.
	*/
	void MLEigenSolver::constructSampleHierarchy(const Eigen::VectorXi& samples)
	{
		printf("____Sampling on %d levels \n", samples.size());
		for (int i = 0; i < samples.size(); ++i) {
			SubspaceSize[i] = samples(i);
			printf("______Level %d -> Subspace size: %d\n", i, SubspaceSize[i]);
		}
		sampleSize = SubspaceSize[0];

		SampleMap.resize(V.rows());
		MuLES::constructFarthestPointSample(V, AdM, Sample, SubspaceSize, geoDist);
		cout << "___Assigning level \n";
		sampleLevel.resize(V.rows());

		SampleMap.setConstant(-1);
		sampleLevel.setConstant(-1);

		// for level n-1
		for (int i = 0; i < SubspaceSize[level - 1]; ++i) {
			sampleLevel(Sample(i)) = level - 1;
			SampleMap(Sample(i)) = i;
		}

		for (int i = level - 1; i > 0; --i) {
			for (int j = SubspaceSize[i]; j < SubspaceSize[i - 1]; ++j) {
				sampleLevel(Sample(j)) = i - 1;
				SampleMap(Sample(j)) = j;
			}
		}
	}

	/**
	*  @brief	Construct a multilevel basis for every level on each of the hierarchy. 
	*  @details
	*   The basis is constructed using (1) farthest point sampling, followed by (2) region growing strategy.
	*  @param	support The expected average overlaps of disks on every degree of freedom on each level.
	*  @param	axis	When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset	When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*/
	void MLEigenSolver::constructBasis(double support, int axis, double offset) {}

	/**
	*  @brief	Construct the basis for 2-level hierarchical system
	*  @details In this set-up only 1 basis (restriction and prolongation) matrix is required.
	*	In general, we take the coarsest level DoFs for this construction.
	*  @param	support The expected average overlaps of disks on every degree of freedom on each level.
	*  @param	axis	When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset	When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*/
	void MLEigenSolver::constructBasisCoarseOnly(double support, int axis, double offset) {}

	/**
	*  @brief
	*   Build a radial basis functions (i.e. geodesic disks) on every samples for every level on each of the hierarchy.
	*  @param	k			The level that RBFs needs to be constructed.
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	isClamped	Tells whether clamping is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*  @param	support		The expected average overlaps of disks on every degree of freedom on each level.
	*/
	void MLEigenSolver::constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support) {}

	/**
	*  @details
	*   Our implementation of Xian's piecewise constant weight function for hierarchical problem.
	*	Originally this method is used for fast linear solver used in deformation of elastics body.
	*  @param	support		The expected average overlaps of disks on every degree of freedom on each level.
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.	
	*/
	void MLEigenSolver::constructBasisNonSmooth(double support, int axis, double offset) {}

	/**
	*  @brief
	*   Construct a growing radial geodesic disk for every sample vertex on each level.
	*  @param	s1			Index (ID) of source vertex, 0 to |V|-1.
	*  @param	D			Vector representing distance from the source vertex.
	*  @param	a			First polynomial coefficient for the unique polynomial function, mapping the distance function to Gaussian-like polynomial function.
	*  @param	b			Second Polynomial coefficient for the unique polynomial function, mapping the distance function to Gaussian-like polynomial function.
	*  @param	maxDist		Radius of the geodesic disk.
	*  @param	col			Index of the sample vertex, determining the column index of the basis matrix.
	*  @param	level		Level on which the geodesic disk is computed.
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*  @param	UTriplet	Eigen::Triplet<T> representing the \code{.cpp}Eigen::SparseMatrix<T> \endcode for the basis on that particular level.
	*/
	void MLEigenSolver::constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int level, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet) {}

	/**
	*  @brief	Perform partition of unity on every row of the basis matrix in every level.
	*  @details
	*	This ensures that a constant functions is mapped as constant function accross hierarchy.
	*/
	void MLEigenSolver::formPartitionOfUnityMultiLevel()
	{
		cout << "Partition of unity \n";
		/* Temporary variables */
		vector<Eigen::Triplet<double>>		diagTrip;
		Eigen::SparseMatrix<double>			DiagNormMatrix;
		Eigen::VectorXd						rowSum;

		for (int i = 0; i < level; i++)
		{
			rowSum.resize(U[i].rows());
			rowSum.setZero();
			DiagNormMatrix.resize(U[i].rows(), U[i].rows());
			diagTrip.reserve(U[i].nonZeros());

			/* Getting the sum of every non-zero elements in a row */
			for (int k = 0; k < U[i].outerSize(); ++k) {
				for (Eigen::SparseMatrix<double>::InnerIterator it(U[i], k); it; ++it) {
					rowSum(it.row()) += it.value();
				}
			}

			/* Get the diagonal matrix, reciprocal of the total sum per row */
			for (int k = 0; k < rowSum.size(); k++) {
				// if the sum of the row is very small, then do not do anything on this row
				if (abs(rowSum(k)) < (10.0*std::numeric_limits<double>::epsilon())) {
					diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0));
					//cout << "Row: " << k << ", value:" << rowSum(k) << endl;
				}
				diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0 / rowSum(k)));
			}
			DiagNormMatrix.setFromTriplets(diagTrip.begin(), diagTrip.end());

			/* Form partition of unity */
			U[i] = DiagNormMatrix * U[i];

			diagTrip.clear();
			diagTrip.shrink_to_fit();
		}
	}

	/**
	*  @brief	Perform partition of unity for 2-level system.
	*  @details
	*	This ensures that a constant functions is mapped as constant function accross hierarchy.
	*/
	void MLEigenSolver::formPartitionOfUnity()
	{
		cout << "> Partition of unity \n";
		/* Temporary variables */
		vector<Eigen::Triplet<double>>		diagTrip;
		Eigen::SparseMatrix<double>			DiagNormMatrix;
		Eigen::VectorXd						rowSum = Eigen::VectorXd::Zero(Basis.rows());
		DiagNormMatrix.resize(Basis.rows(), Basis.rows());
		diagTrip.reserve(Basis.nonZeros());

		/* Getting the sum of every non-zero elements in a row */
		for (int k = 0; k < Basis.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(Basis, k); it; ++it) {
				rowSum(it.row()) += it.value();
			}
		}

		/* Get the diagonal matrix, reciprocal of the total sum per row */
		for (int i = 0; i < rowSum.size(); i++) {
			diagTrip.push_back(Eigen::Triplet<double>(i, i, 1.0 / rowSum(i)));
		}
		DiagNormMatrix.setFromTriplets(diagTrip.begin(), diagTrip.end());

		/* Form partition of unity */
		Basis = DiagNormMatrix * Basis;
	}

	/**
	*  @brief	Solve the eigenproblem in the coarsest level of the hierarchy.
	*  @details
	*	The projection to the subspace is done via the restriction operator (R=U^T).
	*	This function gives out two output: \n
	*   <ul>
	*		<li> `EigVectsReducedNonLifted`	Eigenvectors in the reduced space.
	*		<li> `EigVectsReduced`			Eigenvectors projected (i.e. lifted) to the reference space. 
	*   </ul>
	*/
	void MLEigenSolver::solveRestrictedProblem()
	{
		double eps = 1e-16;
		Eigen::SparseMatrix<double> Id(Basis.cols(), Basis.cols());
		Id.setIdentity();


		cout << "> Creating the reduced system matrices.\n";
		Eigen::SparseMatrix<double> S_, M_;			//reduced stiffness and mass matrix
		S_ = Basis.transpose()*S*Basis;
		M_ = Basis.transpose()*M*Basis;

		printf("__S red %.4f (%d nnz)\n", (double)S_.nonZeros() / (double)S_.rows(), S_.nonZeros());
		//cout << S_.topLeftCorner(10, 10) << endl;
		printf("__M red %.4f (%d nnz)\n", (double)M_.nonZeros() / (double)M_.rows(), M_.nonZeros());
		//cout << M_.topLeftCorner(10, 10) << endl;

		cout << "> Computing the reduced eigenproblem (in GPU). \n";

		int status;
		int iter = 0;

		do {
			if (iter == 0)
				status = computeEigenGPU(S_, M_, EigVectsReducedNonLifted, EigValsReduced);
			else {
				eps *= 10;
				status = computeEigenGPU(S_ + eps*Id, M_, EigVectsReducedNonLifted, EigValsReduced);
				cout << "Shift: " << eps << endl;
			}
			iter++;
		} while (status != 0 && iter < 10);

		std::cout << "Succesful and lifted to fine resolution \n";
		EigVectsReduced = Basis * EigVectsReducedNonLifted;
	}

	/**
	*  @brief
	*   Construct a selector matrix for clamped eigen problem. 	
	*  @param	SelMat		The desired selector matrix 
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*/
	void MLEigenSolver::constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis, double offset) {}

	/**
	*  @brief
	*   Main call to subspaceiteration method (SIM). 
	*  @details
	*	It will then call the specific type of SIM according to what user selects. 
	*  @param	type		Desired type of SIM to perform. One of MuLES::MLEigenSolver::SIMType enum types.
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors)
	*  @param	maxIter		Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			Convergence threshold, as relative residual value.
	*						\begin{equation}
	*							{\frac{{\left | S \phi_i - \lambda_i M \phi_i \right |}_{2}}
	*							{{\left | S \phi_i \right |}_2}}< \varepsilon_q .
	*						\end{equation}
	*  @param	isithShift	A boolean value, determining whether a shift is incorporated to the system.
	*  @param	shiftType	If shift is used, it tells which shifting strategy is used.
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.	
	*/
	void MLEigenSolver::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset) { }

	/**
	*  @brief
	*   Perform the subspace iteration method (SIM), with various possible settings.
	*  @details
	*	This is the standard implementation of SIM, with various possible definition of starting vectors,
	*	but without shifting strategy. It also has the convergence control via relative residual and maximum iterations
	*  @param	type		Desired type of SIM to perform. One of MuLES::MLEigenSolver::SIMType enum types.
	*  @param	SL			Stifness matrix (Laplacian operator) used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	ML			The (lumped) Mass matrix used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors)
	*  @param	maxIter		Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			Convergence threshold, as relative residual value.
	*						\begin{equation}
	*							{\frac{{\left | S \phi_i - \lambda_i M \phi_i \right |}_{2}}
	*							{{\left | S \phi_i \right |}_2}}< \varepsilon_q .
	*						\end{equation}
	*  @param	isithShift	A boolean value, determining whether a shift is incorporated to the system.
	*/
	void MLEigenSolver::performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift) {}

	/**
	*  @brief
	*   Main call to subspaceiteration method (SIM) with shifting strategy.
	*  @details
	*	This is the standard implementation of SIM with shifting strategy, with various possible definition of starting vectors.
	*	It also has the convergence control via relative residual and maximum iterations
	*  @param	type		Desired type of SIM to perform. One of MuLES::MLEigenSolver::SIMType enum types.
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors)
	*  @param	maxIter		Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			Convergence threshold, as relative residual value.
	*						\begin{equation}
	*							{\frac{{\left | S \phi_i - \lambda_i M \phi_i \right |}_{2}}
	*							{{\left | S \phi_i \right |}_2}}< \varepsilon_q .
	*						\end{equation}
	*  @param	isithShift	A boolean value, determining whether a shift is incorporated to the system.
	*  @param	shiftType	If shift is used, it tells which shifting strategy is used.
	*/
	void MLEigenSolver::performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

	/**
	*  @brief
	*   Performing a multilevel subspace iteration method.
	*  @details
	*	Our hierarchical subspace iteration method, without any shifting strategy. 
	*  @param	SL			Stifness matrix (Laplacian operator) used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	ML			The (lumped) Mass matrix used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	k			Level of the hierarchy
	*  @param	XkL			Approximated eigenvectors on the level k
	*  @param	AkL			Approximated eigenvalues on the level k
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors)
	*  @param	maxIter		Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			Convergence threshold, as relative residual value.
	*						\begin{equation}
	*							{\frac{{\left | S \phi_i - \lambda_i M \phi_i \right |}_{2}}
	*							{{\left | S \phi_i \right |}_2}}< \varepsilon_q .
	*						\end{equation}
	*  @param	isithShift	A boolean value, determining whether a shift is incorporated to the system.
	*/
	void MLEigenSolver::performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift) {}

	/**
	*  @brief
	*   Performing a multilevel subspace iteration method with regular shift.
	*  @details
	*	Our hierarchical subspace iteration method, with regular shifting strategy, i.e. shifting is performed after every 4-5 iterations.
	*  @param	SL			Stifness matrix (Laplacian operator) used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	ML			The (lumped) Mass matrix used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	k			Level of the hierarchy
	*  @param	XkL			Approximated eigenvectors on the level k
	*  @param	AkL			Approximated eigenvalues on the level k
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors)
	*  @param	maxIter		Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			Convergence threshold, as relative residual value.
	*						\begin{equation}
	*							{\frac{{\left | S \phi_i - \lambda_i M \phi_i \right |}_{2}}
	*							{{\left | S \phi_i \right |}_2}}< \varepsilon_q .
	*						\end{equation}
	*  @param	isithShift	A boolean value, determining whether a shift is incorporated to the system.
	*  @param	shiftType	If shift is used, it tells which shifting strategy is used.
	*/
	void MLEigenSolver::performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

	/**
	*  @brief
	*   Performing a multilevel subspace iteration method with shifting performed once at the beginning of the iteration.
	*  @details
	*	Our hierarchical subspace iteration method, with regular shifting strategy, i.e. shifting is performed after every 4-5 iterations.
	*  @param	SL			Stifness matrix (Laplacian operator) used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	ML			The (lumped) Mass matrix used for this solver. Described explicitly since it might be called not in the finest level
	*  @param	k			Level of the hierarchy
	*  @param	XkL			Approximated eigenvectors on the level k
	*  @param	AkL			Approximated eigenvalues on the level k
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors)
	*  @param	maxIter		Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			Convergence threshold, as relative residual value.
	*						\begin{equation}
	*							{\frac{{\left | S \phi_i - \lambda_i M \phi_i \right |}_{2}}
	*							{{\left | S \phi_i \right |}_2}}< \varepsilon_q
	*						\end{equation}
	*  @param	isithShift	A boolean value, determining whether a shift is incorporated to the system.
	*  @param	shiftType	If shift is used, it tells which shifting strategy is used.
	*/
	void MLEigenSolver::performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

	void MLEigenSolver::solve(int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset, double levelRatio, double basisSupport) {}

	void MLEigenSolver::performMultiLevelSIM_withShiftOnceConverged(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

	void MLEigenSolver::subspaceIterationMethodHelper(int k, Eigen::SparseMatrix<double>& SK, Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd XbarK, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int solverStatus, int numEigs, int q, int& iterCount, int maxIter, double tol, Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& factorization, bool isWithShift, bool isConvergedNotUsed, int lastConvergedId){}

	/**
	*  @brief
	*   Check the convergence of current iteration.
	*  @details
	*	Convergence check via relative residual. 
	*	\begin{equation}
	*		{\frac{{\left | S \phi_i - \lambda_i M \phi_i \right |}_{2}}
	*		{{\left | S \phi_i \right |}_2}}< \varepsilon_q
	*	\end{equation}
	*  @param	Q			Eigenvectors in the Ritz reduced space
	*  @param	A			Eigenvalues in the Ritz reduced space
	*  @param	MK			Mass matrix in the Ritz subspace 
	*  @param	SK			Stiffness matrix in the Ritz subspace 
	*  @param	convRate	Vector storing the convergence rate of each pair of eigenvalue and eigenvector in Ritz subspace	
	*/
	void MLEigenSolver::checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, int num, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const
	{
		convRate.resize(A.rows());

		//Eigen::MatrixXd KQ = SK*Q.topLeftCorner(Q.rows(), num);
		//Eigen::MatrixXd AMQ = (MK*Q.topLeftCorner(Q.rows(), num)).array().rowwise() * (A.head(num)).transpose().array();
		//Eigen::MatrixXd AMQ = MK*Q.topLeftCorner(Q.rows(), num);
		//Eigen::MatrixXd Diff = KQ - AMQ;

		const int		NUM_OF_THREADS = omp_get_num_procs();
		//const int		NUM_OF_THREADS = 2;
		omp_set_num_threads(NUM_OF_THREADS);
		int tid, ntids, ipts, istart, iproc;
		int iCol = 0;
#pragma omp parallel private(tid,ntids,ipts,istart,iCol)
		{
			iproc = omp_get_num_procs();
			tid = omp_get_thread_num();
			ntids = omp_get_num_threads();
			ipts = (int)ceil((double)num / (double)ntids);
			istart = tid * ipts;
			if (tid == ntids - 1) ipts = num - istart;
			if (ipts <= 0) ipts = 0;

			Eigen::VectorXd Kq, AMq, diff;

			for (iCol = istart; iCol < (istart + ipts); ++iCol) {
				if (iCol >= num) continue;
				Kq = SK * Q.col(iCol);
				AMq = A(iCol)*MK*Q.col(iCol);
				diff = Kq - AMq;
				//diff = KQ.col(i) - A(i)*AMQ.col(i);

				double n1 = diff.transpose()*MK*diff;
				double n2 = Kq.transpose()*MK*Kq;
				//double n1 = Diff.col(i).transpose()*MK*Diff.col(i);
				//double n2 = KQ.col(i).transpose()*MK*KQ.col(i);
				convRate(iCol) = sqrt(n1 / n2);
			}
		}
		//printf("eigVals: %d | S: %dx%d | M: %dx%d \n", A.rows(), SK.rows(), SK.cols(), MK.rows(), MK.cols());
	}

	/**
	*  @brief
	*   Check the convergence of current iteration.
	*  @details
	*	The error bound of an eigenvalue expressed through its corresponding eigenvector in the Ritz subspace eigenbasis
	*	\begin{equation}
	*		{\left( 1 - \frac{{\lambda_i}^2}{q_i^T q_i} \right)}^{1/2} < \varepsilon_q
	*	\end{equation}
	*  @param	Q			Eigenvectors in the Ritz reduced space
	*  @param	A			Eigenvalues in the Ritz reduced space
	*  @param	MK			Mass matrix in the Ritz subspace
	*  @param	SK			Stiffness matrix in the Ritz subspace
	*  @param	convRate	Vector storing the convergence rate of each pair of eigenvalue and eigenvector in Ritz subspace
	*/
	void MLEigenSolver::checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, int num, bool isWithShift, double mu, Eigen::VectorXd& convRate) const
	{
		convRate.resize(A.rows());

		for (int i = 0; i < num; ++i)
		{
			double a2 = A(i)*A(i);
			double qTq = Q.col(i).transpose()*Q.col(i);
			if (isWithShift) qTq += mu * (2 * A(i) - mu);
			double rat = a2 / qTq;
			double conv = abs(1.0 - rat);
			convRate(i) = sqrt(conv);
		}
	}
		
	/**
	*  @brief
	*   Compute a comparison of eigenvalues and eigenvectors using other methods (Matlab, Spectra, etc.)
	*  @details
	*	This serves as benchmark to our multilevel eigensolver. 
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors)
	*  @param	solverType	Selected solver (Matlab, Spectra, or CUDA's cusolver)
	*  @param	maxIters	(If applicable) Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			(If applicable) Convergence threshold, depending on the implementation of each solver, highly probably is different compared to ours.	
	*/
	void MLEigenSolver::computeRefEigenProblem(int numEigs, const EigenSolverType& solverType, int maxIters, double tol)
	{
		switch (solverType)
		{
		case EigenSolverType::EIG_SOLVER_MATLAB:
			std::cout << "Compute in Matlab \n";
			computeEigenMatlab(S, M, numEigs, EigVectsRef, EigValsRef);
			//computeEigenMatlab2(S, M, numEigs, EigVectsRef, EigValsRef);
			break;
		case EigenSolverType::EIG_SOLVER_SPECTRA:
			computeEigenSpectraSparse(S, M, numEigs, EigVectsRef, EigValsRef, 1, maxIters, tol);
			break;
		case EigenSolverType::EIG_SOLVER_CUDA:
			if (M.rows() > 7500) {
				cout << "We think the problem is too big to be solved in GPU! \nConsider other solver for this problem.\n";
				return;
			}
			computeEigenGPU(S, M, EigVectsRef, EigValsRef);
			break;
		default:
			break;
		}
	}

	/**
	*  @brief	Compute a comparison of eigenvalues and eigenvectors with some clampling (e.g. some areas are excluded from computation)
	*  @details
	*	This serves as benchmark to our multilevel eigensolver.
	*  @param	numEigs		Number of sought eigenpairs (eigenvalues and eigenvectors).
	*  @param	solverType	Selected solver (Matlab, Spectra, or CUDA's cusolver).
	*  @param	maxIters	(If applicable) Maximum number of iterations before giving up, and return the value from last iterations.
	*  @param	tol			(If applicable) Convergence threshold, depending on the implementation of each solver, highly probably is different compared to ours.
	*  @param	axis		When clamping is used (by default not), it specifies which axis direction is used for clamping.
	*  @param	offset		When clamping is used (by default not), it determines the ratio w.r.t selected axis for clamping.	
	*/
	void MLEigenSolver::computeRefEigenProblemWithSelector(int numEigs, const EigenSolverType& solverType, int axis, double offset)
	{
		cout << "____ Creating selector matrix \n";
		double xMin;
		double xMax;
		double xScale;
		double xBound;		// within this bound, the system will be discarded

		xMin = V.col(axis).minCoeff();
		xMax = V.col(axis).maxCoeff();
		xScale = offset;
		xBound = xMin + xScale*(xMax - xMin);

		/* Create selector matrix */
		vector<Eigen::Triplet<double>> STriplet;
		int vCounter = 0;

		/* Selector matrix for eigen-Functions */
		if (F.cols() == 3 && S.rows() == V.rows()) {
			STriplet.reserve(round((1.0 - xScale)*V.rows()));
			for (int i = 0; i < V.rows(); ++i) {
				if (V(i, axis) >= xBound) {
					STriplet.push_back(Eigen::Triplet<double>(i, vCounter, 1.0));
					vCounter++;
				}
			}
			SelMatrix.resize(V.rows(), vCounter);
		}
		/* Selector Matrix for vibration modes of elestics */
		else if (F.cols() == 4 && S.rows() == 3 * V.rows()) {
			STriplet.reserve(3 * round((1.0 - xScale)*V.rows()));
			for (int i = 0; i < V.rows(); ++i) {
				if (V(i, axis) >= xBound) {							// only select entries corresponding to selected vertices
					for (int j = 0; j < 3; j++) {
						STriplet.push_back(Eigen::Triplet<double>(3 * i + j, 3 * vCounter + j, 1));
					}
					vCounter++;
				}
			}
			SelMatrix.resize(3 * V.rows(), 3 * vCounter);
		}
		SelMatrix.setFromTriplets(STriplet.begin(), STriplet.end());
		STriplet.clear(); STriplet.shrink_to_fit();

		/* Reduced problem */
		Eigen::SparseMatrix<double>		SLoc;
		Eigen::SparseMatrix<double>		MLoc;
		Eigen::MatrixXd					EigVectsLoc;
		cout << "____Setting up the reduced problem \n";
		SLoc = SelMatrix.transpose()*S*SelMatrix;
		MLoc = SelMatrix.transpose()*M*SelMatrix;

		cout << "____Computing the eigenproblem with selector. \n";
		switch (solverType)
		{
		case EigenSolverType::EIG_SOLVER_MATLAB:
			computeEigenMatlab2(SLoc, MLoc, numEigs, EigVectsLoc, EigValsRef);
			break;
		case EigenSolverType::EIG_SOLVER_SPECTRA:
			computeEigenSpectraSparse(SLoc, MLoc, numEigs, EigVectsLoc, EigValsRef, 0, 50, 1e-2);
			break;
		case EigenSolverType::EIG_SOLVER_CUDA:
			if (MLoc.rows() > 7500) {
				cout << "We think the problem is too big to be solved in GPU! \nConsider other solver for this problem.\n";
				return;
			}
			computeEigenGPU(SLoc, MLoc, EigVectsLoc, EigValsRef);
			break;

		default:
			break;
		}

		/* Lift it up to the full space */
		cout << "____Lifting to the original size\n";
		EigVectsRef = SelMatrix*EigVectsLoc;
		printf("Size of eigenvectors: %dx%d \n", EigVectsRef.rows(), EigVectsRef.cols());

	}

	/* Visualization tools */

	/**
	*  @brief	Visualize the geodesic distance over varying level.
	*  @param	viewer		GUI for model interaction.
	*  @param	k			Level on the hierarchy.
	*/
	void MLEigenSolver::visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k) const
	{
		Eigen::MatrixXd VColor;
		igl::jet(geoDist[k], true, VColor);
		viewer.data().set_colors(VColor);
	}

	/**
	*  @brief	Visualize the basis, on varying level.
	*  @param	viewer		GUI for model interaction.
	*  @param	id			ID/index of basis to show (corresponds to column of the basis matrix), mapped to full resolution using the *Prolongation* operator.
	*  @param	k			Level on the hierarchy.
	*  @param	scale		(If applicable) Ratio for scaling of the basis.
	*/
	void MLEigenSolver::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k, double scale) const {}

	/**
	*  @brief	Visualize the selected eigenvector corresponding to a certain eigenvalue.
	*  @param	viewer		GUI for model interaction.
	*  @param	id			ID/index of eigenvector to show (i.e. the ID of the corresponding eigenvalue).
	*  @param	eigType		Type of the eigenvector to show, from MuLES::MLEigenSolver::EigenVectorType enum type.
	*  @param	scale		(If applicable) Ratio for scaling of the basis.
	*/
	void MLEigenSolver::visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale) const {}
	
	/**
	*  @brief	Visualize samples at level 1 (right after the fine resolution mesh).
	*  @param	viewer		GUI for model interaction.
	*/
	void MLEigenSolver::visualizeClusterOfSamplesLevel0(igl::opengl::glfw::Viewer &viewer)
	{
		Eigen::MatrixXd VColor;
		Eigen::VectorXd vcolor(V.rows());
		for (int i = 0; i < V.rows(); i++)
		{
			vcolor(i) = (double)clusters[0](i) / (double)V.rows();
		}

		igl::jet(vcolor, true, VColor);
		viewer.data().set_colors(VColor);
	}

	/**
	*  @brief	Visualize samples at level `k` on the hierarchy.
	*  @param	viewer		GUI for model interaction.
	*  @param	k			Level of sample in the hierarchy to show.
	*/
	void MLEigenSolver::visualizeClusterOfSamplesLevelK(igl::opengl::glfw::Viewer &viewer, int k)
	{
		viewer.data().points.resize(0, 6);
		Eigen::MatrixXd VColor;
		Eigen::VectorXd fcolor(SubspaceSize[k]);


		//cout << "Set color for level " << k << endl;
		for (int i = 0; i < SubspaceSize[k]; i++) fcolor(i) = (double)i; // / (double)clusters[k].size();

		igl::jet(fcolor, true, VColor);

		//cout << "Iterate to set color for each sample vertex \n";
		for (int i = 0; i < SubspaceSize[k - 1]; i++)
		{
			//printf("Point %d (Sample %d) belongs to cluster %d \n", Sample(i), i, clusters[k](i));
			viewer.data().add_points(V.row(Sample(i)), VColor.row(clusters[k](i)));
		}
	}

	/**
	*  @brief	Getter for the vertex. Request the matrix storing the vertex coordinates.
	*/
	Eigen::MatrixXd MLEigenSolver::getV() const { return V; }
		
	/**
	*  @brief	Getter for the faces. 
	*  @details
	*	Request the matrix representing the discretization of the model (face in terms of surface, tetrahedron for volume).
	*	Each row contains 3 integer ids corresponding to the vertex location of the face. 
	*/
	Eigen::MatrixXi MLEigenSolver::getF() const { return F; }

	/**
	*  @brief	Setter for the vertex. Provide the matrix storing the vertex coordinates.
	*  @param	Vin		Matrix storing the vertex coordinates. Generally in the size of `nx3`, where `n` is the number of vertices.
	*/
	void MLEigenSolver::setV(const Eigen::MatrixXd& Vin) { V = Vin; }

	/**
	*  @brief	Setter for the Face. Provide the matrix storing the vertex IDs corresponds to each face.
	*  @param	Fin		Matrix storing the vertex IDs (integer). Generally in the size of `nx3`, where `n` is the number of faces.
	*/
	void MLEigenSolver::setF(const Eigen::MatrixXi& Fin) { F = Fin; }

	/**  @brief	Obtain the stiffness matrix for the Laplacian. */
	Eigen::SparseMatrix<double> MLEigenSolver::getStiffnessMatrix() const { return S; }

	/**  @brief	Obtain the Mass matrix for the Laplacian. */
	Eigen::SparseMatrix<double> MLEigenSolver::getMassMatrix() const { return M; }

	/**  @brief	Obtain the (combined) Bais matrix. */
	Eigen::SparseMatrix<double> MLEigenSolver::getBasisMatrix() const { return Basis; }

	/**  @brief	Obtain the value of `isClampBasis` flag. */
	bool MLEigenSolver::getClampBasis() const { return isClampedBasis; }

	/**  @brief	Set the value of `isClampBasis` flag. */
	void MLEigenSolver::setClampBasis(bool isClamped) { isClampedBasis = isClamped; }

	/** @brief	Obtain number of levels in the hierarchy. */
	int MLEigenSolver::getLevel() const { return level; }

	/**  @brief	Obtain the size of the samples. */
	int MLEigenSolver::getSampleSize() const { return sampleSize; }

	/**  @brief	Obtain the value of `Sample` vector, containing information of number of samples on each level. */
	const Eigen::VectorXi& MLEigenSolver::getSample() const { return Sample; }

	/**  @brief	Obtain the value of Subspace size on each level */
	const vector<int>& MLEigenSolver::getSubspaceSize() const { return SubspaceSize; }

	/**  @brief	Set the value of shifting parameter used (the one proposed by Zhao) */
	void MLEigenSolver::setZhaoShiftParam(const double shiftParam) { zhaoShiftParam = shiftParam; }

	/**  @brief	Obtain the value of shifting parameter used (the one proposed by Zhao) */
	double MLEigenSolver::getZhaoShiftParam() const { return zhaoShiftParam; }

	/**  @brief	Set the ratio of iterating vectors w.r.t numbers of eigenvalues (default is 2.0) */
	void MLEigenSolver::setQParam(const double value) { qParam = value; }

	/**  @brief	Obtain the ratio of iterating vectors w.r.t numbers of eigenvalues (default is 2.0) */
	double MLEigenSolver::getQParam() const { return qParam; }

	void MLEigenSolver::setNotIterateOverConvergedFlag(bool flag) { isConvergedVectsNotUsed = flag; }
	bool MLEigenSolver::getNotIterateOverConvergedFlag() const { return isConvergedVectsNotUsed; };
	void MLEigenSolver::setParallelSolverFlag(bool flag) { isParallelSolverEnabled=flag; }
	bool MLEigenSolver::getParallelSolverFlag() const { return isParallelSolverEnabled; }


	/* [Getters of the eigen-values + vectors] */
	/**  @brief	Obtain the entire eigenvectors (obtained from reference solver). */
	const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsRef() const { return EigVectsRef; }

	/** @brief	Obtain the a selected eigenvector (obtained from reference solver). 
	*	param	id	ID/index of eigenvector to show	*/
	const Eigen::VectorXd& MLEigenSolver::getEigenVectorsRef(int id) const
	{
		if (id >= EigVectsRef.cols())	return EigVectsRef.col(EigValsRef.rows() - 1);
		else if (id < 0)						return EigVectsRef.col(0);

		return EigVectsRef.col(id);
	}

	/** @brief	Obtain the entire eigenvalues (obtained from reference solver), represented as a vector. */
	const Eigen::VectorXd& MLEigenSolver::getEigenValuesRef() const { return EigValsRef; }

	/** @brief	Obtain the selected eigenvalue (obtained from reference solver)
	*	param	id	ID/index of eigenvector to show	*/
	const double MLEigenSolver::getEigenValueRef(int id) const
	{
		if (id >= EigValsRef.rows())	return EigValsRef(EigValsRef.rows() - 1);
		else if (id < 0)						return EigValsRef(0);

		return EigValsRef(id);
	}

	/**  @brief	Obtain the entire *approximated* eigenvectors (obtained in the reduced subspace, then lifted to the reference space). */
	const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsReduced() const { return EigVectsReduced; }

	/** @brief	Obtain the a selected *approximated* eigenvector (obtained in the reduced subspace, then lifted to the reference space).
	*	param	id	ID/index of eigenvector to show	*/
	const Eigen::VectorXd& MLEigenSolver::getEigenVectorsReduced(int id) const
	{
		if (id >= EigVectsReduced.cols())	return EigVectsReduced.col(EigValsReduced.rows() - 1);
		else if (id < 0)						return EigVectsReduced.col(0);

		return EigVectsReduced.col(id);
	}

	/** @brief	Obtain the entire *approximated* eigenvalues (obtained in the reduced subspace, then lifted to the reference space), represented as a vector. */
	const Eigen::VectorXd& MLEigenSolver::getEigenValuesReduced() const { return EigValsReduced; }

	/** @brief	Obtain the selected *approximated* eigenvalue (obtained in the reduced subspace)
	*	param	id	ID/index of eigenvalue to show	*/
	const double MLEigenSolver::getEigenValueReduced(int id) const
	{
		if (id >= EigValsReduced.rows())	return EigValsReduced(EigValsReduced.rows() - 1);
		else if (id < 0)						return EigValsReduced(0);

		return EigValsReduced(id);
	}

	/**  @brief	Obtain the entire eigenvectors (obtained our our SIM or other type of SIM). */
	const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsSIM() const { return EigVectsSIM; }

	/** @brief	Obtain the a selected eigenvector (obtained our our SIM or other type of SIM).
	*	param	id	ID/index of eigenvector to show	*/
	const Eigen::VectorXd& MLEigenSolver::getEigenVectorsSIM(int id) const
	{
		if (id >= EigVectsSIM.cols())		return EigVectsSIM.col(EigValsSIM.rows() - 1);
		else if (id < 0)						return EigVectsSIM.col(0);
		return EigVectsSIM.col(id);
	}

	/** @brief	Obtain the entire eigenvalues (obtained our our SIM or other type of SIM), represented as a vector. */
	const Eigen::VectorXd& MLEigenSolver::getEigenValuesSIM() const { return EigValsSIM; }

	/** @brief	Obtain the selected eigenvalue (obtained our our SIM or other type of SIM)
	*	param	id	ID/index of value to show	*/
	const double MLEigenSolver::getEigenValueSIM(int id) const
	{
		if (id >= EigValsSIM.rows())		return EigValsSIM(EigValsSIM.rows() - 1);
		else if (id < 0)						return EigValsSIM(0);

		return EigValsSIM(id);
	}
	//template void MLEigenSolver::setSamplingLevels(Eigen::PlainObjectBase<double>& sampSize, int numEigen, double levelRatio);

	void MLEigenSolver::cublasGEMMtest()
	{
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;	
		
		int nRow = 200000, nMid = 500, nCol = 500;
		Eigen::MatrixXd XkMat(nRow, nCol); XkMat.setZero();
		Eigen::MatrixXd XMat = Eigen::MatrixXd::Random(nRow, nMid);
		Eigen::MatrixXd QMat = Eigen::MatrixXd::Random(nMid, nCol);

		cout << "ON <<C>>PU ";
		t1 = chrono::high_resolution_clock::now();
		XkMat = XMat*QMat;
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;
		std::cout << "Xk \n" << XkMat.topLeftCorner(10,10) << std::endl;
		XkMat.setZero();
		std::cout << "Set Zero \n" << XkMat.topLeftCorner(10, 10) << std::endl;

		cout << "ON <<G>>PU ";
		t1 = chrono::high_resolution_clock::now();

		cudaError_t cudaStat;		// cudaMalloc status
		cublasStatus_t stat;		// CUBLAS functions status
		cublasHandle_t BLAShandle;		// CUBLAS context


		double *h_Xk;
		double *h_X;
		double *h_Q;
		int mX = XMat.rows();
		int nX = XMat.cols();
		int mQ = QMat.rows();
		int nQ = QMat.cols();

		// Xk = X*Q;
		h_Xk =(double*)malloc(mX*nQ * sizeof(double)); // host memory for a
		h_X = (double*)malloc(mX*nX * sizeof(double)); // host memory for b
		h_Q = (double*)malloc(mQ*nQ * sizeof(double)); // host memory for c
		///h_Xk = XkMat.data();
		h_X = XMat.data();
		h_Q = QMat.data();

		// For the device (i.e. GPU)
		double *d_Xk;
		double *d_X;
		double *d_Q;
		cudaStat = cudaMalloc((void **)&d_Xk, mX*nQ * sizeof(double));
		cudaStat = cudaMalloc((void **)&d_X,  mX*nX * sizeof(double));
		cudaStat = cudaMalloc((void **)&d_Q,  mQ*nQ * sizeof(double));

		stat = cublasCreate(&BLAShandle);
		stat = cublasSetMatrix(mX, nX, sizeof(double), h_X, mX, d_X, mX);
		stat = cublasSetMatrix(mQ, nQ, sizeof(double), h_Q, mQ, d_Q, mQ);
		stat = cublasSetMatrix(mX, nQ, sizeof(double), h_Xk, mX, d_Xk, mX);
		double alpha = 1.0;
		double beta = 0.0;

		stat = cublasDgemm(BLAShandle, CUBLAS_OP_N, CUBLAS_OP_N, mX, nQ, nX, &alpha, d_X, mX, d_Q, mQ, &beta, d_Xk, mX);
		
		stat = cublasGetMatrix(mX, nQ, sizeof(double), d_Xk, mX, h_Xk, mX);
		XkMat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_Xk, mX, nQ, Eigen::OuterStride<>(mX));

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;

		//std::cout << "X \n" << XMat << std::endl;
		//std::cout << "Q \n" << QMat << std::endl;
		//std::cout << "Xk \n" << XkMat << std::endl;
		std::cout << "Xk \n" << XkMat.topLeftCorner(10, 10) << std::endl;

		if (d_Xk) cudaFree(d_Xk);
		if (d_X) cudaFree(d_X);
		if (d_Q) cudaFree(d_Q);
		if (BLAShandle) cublasDestroy(BLAShandle);
		cudaDeviceReset();
	}

	void MLEigenSolver::cusparseCSRMMtest()
	{
		

		cusparseHandle_t	cuSpHandle;
		cusparseStatus_t	cuSpStat;
		cudaError_t			cudaStat1 = cudaSuccess;

		Eigen::SparseMatrix<double, Eigen::RowMajor> SS = S;
		/* Some random X matrix */
		std::cout << "Set up random init matrix X and SX \n";
		Eigen::MatrixXd X = Eigen::MatrixXd::Random(S.rows(), 100);
		Eigen::MatrixXd SX = Eigen::MatrixXd::Random(S.rows(), 100);

		std::cout << "Obtain structure of sparse matrix S. \n";
		cusparseMatDescr_t descrS = 0, descrX = 0;
		const int m_S = SS.rows();
		const int n_S = SS.cols();
		const int lda_S = SS.rows();
		const int nnz_S = SS.nonZeros();
		const double *h_csrVal_S = (double*) malloc(nnz_S*sizeof(double));
		const int *h_csrRowPtr_S = (int*) malloc((m_S + 1) * sizeof(int));
		const int *h_csrColInd_S = (int*) malloc(nnz_S *sizeof(int));
		S.makeCompressed();
		h_csrVal_S = SS.valuePtr();
		h_csrColInd_S = SS.innerIndexPtr();
		h_csrRowPtr_S = SS.outerIndexPtr();

		const int m_X = X.rows();
		const int n_X = X.cols();
		const int lda_X = X.rows();

		std::cout << "Allocate space for the result of SX=S*X. \n";
		const int m_SX = SX.rows();
		const int n_SX = SX.cols();
		const int lda_SX = SX.rows();
		double *h_SX = (double*)malloc(m_SX*n_SX * sizeof(double));

		std::cout << "Obtain parameters from sparse matrix X (in CSR format?). \n";
		double *d_csrVal_S; cudaMalloc(&d_csrVal_S, nnz_S * sizeof(double));
		int *d_csrRowPtr_S; cudaMalloc(&d_csrRowPtr_S, (m_S+1) * sizeof(int));
		int *d_csrColInd_S; cudaMalloc(&d_csrColInd_S, nnz_S * sizeof(int));

		std::cout << "Copy X and SX to GPU :";
		double *d_X; cudaMalloc(&d_X, m_X*n_X * sizeof(double));
		cudaStat1 = cudaMemcpy(d_X, X.data(), m_X*n_X * sizeof(double), cudaMemcpyHostToDevice);
		std::cout << "d_X " << cudaStat1 << "| ";
		double *d_SX; cudaMalloc(&d_SX, m_SX*n_SX * sizeof(double));
		cudaStat1 = cudaMemcpy(d_SX, SX.data(), m_SX*n_SX * sizeof(double), cudaMemcpyHostToDevice);
		std::cout << "d_SX " << cudaStat1 << "\n ";

		std::cout << "Value \n" << S.valuePtr() << std::endl;
		std::cout << "row \n" << h_csrRowPtr_S << std::endl;
		std::cout << "col \n" << h_csrColInd_S << std::endl;

		std::cout << "Copy S to GPU ";
		cudaStat1 = cudaMemcpy(d_csrVal_S,		h_csrVal_S,		nnz_S * sizeof(double), cudaMemcpyHostToDevice);
		std::cout << "val_S " << cudaStat1 << "| ";
		cudaStat1 = cudaMemcpy(d_csrRowPtr_S,	h_csrRowPtr_S,	(m_S+1)* sizeof(int), cudaMemcpyHostToDevice);
		std::cout << "row ptr " << cudaStat1 << "| ";
		cudaStat1 = cudaMemcpy(d_csrColInd_S,	h_csrColInd_S,	nnz_S * sizeof(int), cudaMemcpyHostToDevice);
		std::cout << "cold indices " << cudaStat1 << "\n";
		
		double alpha = 1.0;
		double beta = 0.0;
		
		size_t bufferSize;
		void* buffer = nullptr;

		std::cout << "Multiplication in GPU \n";
		//cuSpStat = cusparseDcsrmm(cuSpHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_S, n_X, n_S, nnz_S, &alpha, descrS, d_csrVal_S, d_csrRowPtr_S, d_csrColInd_S, d_X, m_X, &beta, d_SX, m_SX);
		std::cout << "mult " << cuSpStat << "\n";

		std::cout << "Copy back to CPU \n";
		cudaStat1 = cudaMemcpy(h_SX, d_SX, m_SX*n_SX * sizeof(double), cudaMemcpyDeviceToHost);
		std::cout << "SX to CPU " << cudaStat1 << "\n";

		std::cout << "format it to Eigen Matrix \n";
		SX = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_SX, n_S, m_X, Eigen::OuterStride<>(n_S));

		std::cout << "S\n" << S.topLeftCorner(10, 10) << std::endl << std::endl;
		std::cout << "X\n" << X.topLeftCorner(10, 10) << std::endl << std::endl;
		std::cout << "SX\n" << SX.topLeftCorner(10, 10) << std::endl << std::endl;

		/*Obtain sparse matrix*/

		
	}

	void MLEigenSolver::cusparseSpMMtest()
	{
		chrono::high_resolution_clock::time_point	t0, t1, t2;
		chrono::duration<double>					duration;
		std::cout << "ON <<G>>PU ";
		std::cout << "___Set up + allocating memory: ";
		t1 = chrono::high_resolution_clock::now();
		t0 = t1;
		
		cusparseHandle_t	cuSpHandle;
		cusparseStatus_t	cuSpStat;
		cudaError_t			cudaStat1 = cudaSuccess;

		// C = S*B where A is a sparse and B is a dense matrix
		// Setup the sparse matrix S CPU
		S.makeCompressed();
		int m_S = S.rows();
		int n_S = S.cols();
		int nnz_S = S.nonZeros();


		// Set-up the dense matrix B and C in CPU
		int m_B = S.rows();
		int n_B = 100;
		int m_C = S.rows();
		int n_C = n_B;
		double* h_C = (double*)malloc(m_C * n_C * sizeof(double));


		//std::cout << "S: \n" << S.topLeftCorner(10, 10) << std::endl;
		//std::cout << "B: \n" << BMat.topLeftCorner(10, 10) << std::endl;
		//std::cout << "C: \n" << CMat.topLeftCorner(10, 10) << std::endl;

		// Allocate space in GPU for A, B and C

		double* d_csrVal_S; cudaMalloc(&d_csrVal_S, nnz_S * sizeof(double));
		int* d_csrRowPtr_S; cudaMalloc(&d_csrRowPtr_S, (m_S + 1) * sizeof(int));
		int* d_csrColInd_S; cudaMalloc(&d_csrColInd_S, nnz_S * sizeof(int));
		double* d_B; cudaMalloc(&d_B, m_B * n_B * sizeof(double));
		double* d_C; cudaMalloc(&d_C, m_C * n_C * sizeof(double));

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "in " << duration.count() << " seconds" << std::endl;

		Eigen::MatrixXd BMat = Eigen::MatrixXd::Random(m_B, n_B);
		Eigen::MatrixXd CMat(m_C, n_C); CMat.setZero();

		std::cout << "___Copying to memory: ";
		t1 = chrono::high_resolution_clock::now();
		// Copy A, B and C to GPU 
		cudaStat1 = cudaMemcpy(d_csrVal_S, S.valuePtr(), nnz_S * sizeof(double), cudaMemcpyHostToDevice);
		cudaStat1 = cudaMemcpy(d_csrRowPtr_S, S.outerIndexPtr(), (m_S + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaStat1 = cudaMemcpy(d_csrColInd_S, S.innerIndexPtr(), nnz_S * sizeof(int), cudaMemcpyHostToDevice);

		cudaStat1 = cudaMemcpy(d_B, BMat.data(), m_B * n_B * sizeof(double), cudaMemcpyHostToDevice);
		//cudaStat1 = cudaMemcpy(d_C, CMat.data(), m_C * n_C * sizeof(double), cudaMemcpyHostToDevice);
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "in " << duration.count() << " seconds" << std::endl;

		std::cout << "___Creatign matrices in GPU: ";
		t1 = chrono::high_resolution_clock::now();
		cusparseSpMatDescr_t matS;
		cusparseDnMatDescr_t matB;
		cusparseDnMatDescr_t matC;
		void* dBuffer1 = NULL, *dBuffer2 = NULL;
		size_t bufferSize1 = 0, bufferSize2 = 0;
		cusparseCreate(&cuSpHandle);
		cuSpStat = cusparseCreateCsr(&matS, m_S, n_S, nnz_S, d_csrRowPtr_S, d_csrColInd_S, d_csrVal_S, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
		cuSpStat = cusparseCreateDnMat(&matB, m_B, n_B, m_B, d_B, CUDA_R_64F, CUSPARSE_ORDER_COL);
		cuSpStat = cusparseCreateDnMat(&matC, m_C, n_C, m_C, d_C, CUDA_R_64F, CUSPARSE_ORDER_COL);
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "in " << duration.count() << " seconds" << std::endl;

		double alpha = 1.0;
		double beta = 0.0;

		std::cout << "___Multiplying in GPU: ";
		t1 = chrono::high_resolution_clock::now();
		// Multiply matrix C = A * B in GPU
		cusparseOperation_t opS = CUSPARSE_OPERATION_NON_TRANSPOSE;
		cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
		cuSpStat = cusparseSpMM_bufferSize(cuSpHandle, opS, opB, &alpha, matS, matB, &beta, matC, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, &bufferSize1);
		cudaMalloc((void**)&dBuffer1, bufferSize1);
		cuSpStat = cusparseSpMM(cuSpHandle, opS, opB, &alpha, matS, matB, &beta, matC, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, dBuffer1);
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "in " << duration.count() << " seconds" << std::endl;

		std::cout << "___Copying back to CPU ";
		t1 = chrono::high_resolution_clock::now();
		// Copy C to CPU
		cudaStat1 = cudaMemcpy(h_C, d_C, m_C * n_C * sizeof(double), cudaMemcpyDeviceToHost);
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "in " << duration.count() << " seconds" << std::endl;

		// Format C to Eigen::MatrixXd
		std::cout << "___Format to Eigen ";
		t1 = chrono::high_resolution_clock::now();
		CMat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_C, m_C, n_C, Eigen::OuterStride<>(m_S));
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;
		
		duration = t2 - t0;
		cout << "in " << duration.count() << " seconds" << endl;

		std::cout << "GPU" << CMat.topLeftCorner(10, 10) << std::endl;

		t1 = chrono::high_resolution_clock::now();
		CMat = S * BMat;
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;
		std::cout << "CPU" << CMat.topLeftCorner(10, 10) << std::endl;

		if (d_B) cudaFree(d_B);
		if (d_C) cudaFree(d_C);
		if (d_csrColInd_S) cudaFree(d_csrColInd_S);
		if (d_csrVal_S) cudaFree(d_csrVal_S);
		if (d_csrRowPtr_S) cudaFree(d_csrRowPtr_S);
		if (cuSpHandle) cusparseDestroy(cuSpHandle);
		cudaDeviceReset();
	}
}