#include "MLEigenSolver.h"
#include <Eigen/PardisoSupport>
#include <omp.h>
#include <random>

#include <igl/exact_geodesic.h>
#include <igl/isolines.h>
#include <igl/jet.h>
#include <igl/avg_edge_length.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <cusolver_common.h> 
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
namespace HSIM
{
	void MLEigenSolver::doEdgeFlip(unsigned int n){ }

	void MLEigenSolver::checkBoundary() {}
	
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

		scaleMesh(V, F, 2.0);
		projectToSphere(V, F, 1.0);

		checkBoundary();
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
		cout << "____> Creating neighborhood structure of the mesh. \n";
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
	*  @see	HSIM::MLSolverFunctions::constructMassMatrix()
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

	void MLEigenSolver::setSamplingLevelsAuto(Eigen::VectorXi& sampSize, int numEigen, double levelRatio, bool isFixedLowBound)
	{
		if (numEigen > V.rows()) {
			std::cout << "Error! The number of sought eigenvalues should not exceed the number of degree of freedom of the geometry.\n";
			return;
		}
		if (V.rows() < 2000) {
			sampSize.resize(1);
			sampSize(0) = V.rows();
			std::cout << "____Since the number of vertices is relatively small, the system will attempt solving it directly in the GPU. \n";
			return;
		}

		const double EXP_RAT = levelRatio;
		int lowBound;
		if		(isFixedLowBound)	{ lowBound = 1000; }
		else						{ lowBound = 2 * numEigen; }

		int numVert = V.rows();
		int dimSubSpace = max(2*numEigen, lowBound);
		double ratInit = (double)numVert / (double)dimSubSpace;
		double ratio = log(ratInit)/log(EXP_RAT);
		//std::cout << "The ratio of vert to subspace is " << ratInit << " yielding " << ratio << " ratio setup \n";

		if (ratio < 1.0) {
			sampSize.resize(1);
			sampSize(0) = dimSubSpace;
			//std::cout << "The ratio is " << ratio << " thus we'll just have 1 level with " << sampSize(0) << " DoFs\n";
		}
		else {
			int numLevels = (int)floor(ratio);
			double remainder = ratio - numLevels;
			//numLevels += 1;
			//std::cout << "Number of levels: " << numLevels << "+1, with remainder of " << remainder << std::endl;

				//std::cout << "CASE 2: 0.2 <= remainder < 0.8 \n";
			Eigen::VectorXd addRatio(numLevels);
			double sumRatio = 0.0;
			for (int i = 0; i < numLevels; ++i) {
				addRatio(i) = pow(2.0, numLevels - i - 1);
				sumRatio += addRatio(i);
			}

			sampSize.resize(numLevels);
			sampSize(0) = (int)ceil((double)numVert / pow(EXP_RAT, 1.0 + addRatio(0) * remainder / sumRatio));
			for (int i = 1; i < numLevels; ++i) {
				sampSize(i) = (int)ceil((double)sampSize(i - 1) / pow(EXP_RAT, 1.0 + addRatio(i) * remainder / sumRatio));
			}
		}

		printf("____Level %d has %d entries \n", 0, numVert);
		for (int i = 0; i < sampSize.rows(); ++i) {
			printf("____Level %d has %d entries \n", i+1, sampSize(i));
		}
	}

	void MLEigenSolver::setSamplingLevelsSameRate(Eigen::VectorXi& sampSize, int numEigen, double levelRatio, bool isFixedLowBound)
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
		int lowBound;
		if (isFixedLowBound) { lowBound = 1000; }
		else { lowBound = 2 * numEigen; }
		int numVert = V.rows();
		int dimSubSpace = max(2 * numEigen, lowBound);
		double ratInit = (double)numVert / (double)dimSubSpace;
		double ratio = log(ratInit) / log(EXP_RAT);
		//std::cout << "The ratio of vert to subspace is " << ratInit << " yielding " << ratio << " ratio setup \n";

		if (ratio < 1.0) {
			sampSize.resize(1);
			sampSize(0) = dimSubSpace;
			//std::cout << "The ratio is " << ratio << " thus we'll just have 1 level with " << sampSize(0) << " DoFs\n";
		}
		else {
			int numLevels = (int)floor(ratio);
			sampSize.resize(numLevels);
			double levelRatio = ratio / (double)numLevels;
			sampSize(0) = (int)ceil((double)numVert / pow(EXP_RAT, levelRatio));

			for (int i = 1; i < numLevels; ++i) {
				sampSize(i) = (int)ceil((double)sampSize(i-1)/ pow(EXP_RAT, levelRatio));
			}
		}

		printf("____Level %d has %d entries \n", 0, numVert);
		for (int i = 0; i < sampSize.rows(); ++i) {
			printf("____Level %d has %d entries \n", i + 1, sampSize(i));
		}
	}

	void MLEigenSolver::setSamplingLevelFixedNum(Eigen::VectorXi& sampSize, int numEigen, double levelRatio, bool isFixedLowBound)
	{
		int numLevelDif = 0;
		if (V.rows() > 100000) { numLevelDif = 2;}
		else {					 numLevelDif = 1;}

		int lowBound;
		if (isFixedLowBound) { lowBound = 1000; }
		else { lowBound = 2 * numEigen; }

		int dimSubSpace = max(2 * numEigen, lowBound);
		double ratio = (double)V.rows() / dimSubSpace;
		ratio = log(ratio) / log(levelRatio);

		sampSize.resize(numLevelDif);
		for (int i = 0; i < numLevelDif; ++i) {
			sampSize(i) = ceil((double)V.rows() / (pow(pow(levelRatio, ratio / numLevelDif),i+1)));
		}

		printf("____Level %d has %d entries \n", 0, V.rows());
		for (int i = 0; i < sampSize.rows(); ++i) {
			printf("____Level %d has %d entries \n", i + 1, sampSize(i));
		}
	}
	
	void MLEigenSolver::setSamplingLevelFixedLevels(Eigen::VectorXi& sampSize, int numEigen, int numLevels, bool isFixedLowBound)
	{
		int numLevelDif = numLevels-1;

		int lowBound;
		if (isFixedLowBound) { lowBound = 1000; }
		else { lowBound = 2 * numEigen; }

		int dimSubSpace = max(2 * numEigen, lowBound);
		double ratio = (double)V.rows() / dimSubSpace;
		ratio = pow(ratio, 1.0 / (double)numLevelDif);

		sampSize.resize(numLevelDif);
		for (int i = 0; i < numLevelDif; ++i) {
			sampSize(i) = ceil((double)V.rows() / (pow(ratio, i + 1)));
		}

		printf("____Level %d has %d entries \n", 0, V.rows());
		for (int i = 0; i < sampSize.rows(); ++i) {
			printf("____Level %d has %d entries \n", i + 1, sampSize(i));
		}
	}
	/**
	*  @brief	 Construct the number of samples in every level via farthest point sampling. 
	*  @details	
	*   Starting from the coarsest level, it computes the the first set of sampled DoFs (i.e. vertices).
	*   Then for every level, it adds more samples according the 
	*  @see HSIM::constructFarthestPointSample() \n HSIM::MLEigenSolver::setSamplingLevels() 
	*  @param	samples	A constact vector stating the number of samples on every level of the hierarchy.
	*/
	void MLEigenSolver::constructSampleHierarchy(double support, const Eigen::VectorXi& samples) {}

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
	void MLEigenSolver::constructSTVDDisk(STVD& stvd, const int s1, Eigen::VectorXd& D, const double a, const double b, double maxDist, const int col, int level, int axis, double offset, vector<Eigen::Triplet<double>>& UTriplet) {}

	void MLEigenSolver::computeDistanceFromBoundary(Eigen::VectorXd& D) {}
	void MLEigenSolver::mapDistanceToFunctionOnBoundary(const Eigen::VectorXd& D) {}

	/**
	*  @brief	Perform partition of unity on every row of the basis matrix in every level.
	*  @details
	*	This ensures that a constant functions is mapped as constant function accross hierarchy.
	*/
	void MLEigenSolver::formPartitionOfUnityMultiLevel()
	{
		cout << "____> Partition of unity \n";
		/* Temporary variables */
		vector<Eigen::Triplet<double>>		diagTrip;
		Eigen::SparseMatrix<double>			DiagNormMatrix;
		Eigen::VectorXd						rowSum;

		if (hasBoundary && boundaryType == BoundaryCondition::BOUNDARY_DIRICHLET) {
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
					if (i == 0) {
						if (funcBoundary[0](k) <= maxNeighDist[0] / 2.0) {
							diagTrip.push_back(Eigen::Triplet<double>(k, k, funcBoundary[0](k) / rowSum(k)));
						}
						else {
							diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0/rowSum(k)));
						}
					}

					if (abs(rowSum(k)) < (10000.0 * std::numeric_limits<double>::epsilon())) {
						diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0));
						//cout << "Row: " << k << ", value:" << rowSum(k) << endl;
					}
					else 
					{
						double scaleCoef;
						if (i == 0) {
							scaleCoef = funcBoundary[0](k);
						}
						else {
							scaleCoef = funcBoundary[i](k);
						}

						diagTrip.push_back(Eigen::Triplet<double>(k, k, scaleCoef / rowSum(k)));
					}
				}
				DiagNormMatrix.setFromTriplets(diagTrip.begin(), diagTrip.end());

				/* Form partition of unity */
				U[i] = DiagNormMatrix * U[i];

				diagTrip.clear();
				diagTrip.shrink_to_fit();
			}
		}
		else {
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
					//if (i == 0 && boundaryVertexCheck[k]) {
					//	printf("Boundary %d with total sum of %.12f \n", k, rowSum(k));
					//}
					if (abs(rowSum(k)) < (10.0 * std::numeric_limits<double>::epsilon())) {
						diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0));
						//cout << "Row: " << k << ", value:" << rowSum(k) << endl;
					}
					else {
						diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0 / rowSum(k)));
					}
				}
				DiagNormMatrix.setFromTriplets(diagTrip.begin(), diagTrip.end());

				/* Form partition of unity */
				U[i] = DiagNormMatrix * U[i];

				diagTrip.clear();
				diagTrip.shrink_to_fit();
			}
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

		// Check the convergence if necessary
		//Eigen::VectorXd conv4;
		//printf("EVec: %dx%d || EVals: %d || M: %dx%d || S: %dx%d \n", EigVectsReduced.rows(), EigVectsReduced.cols(), EigValsReduced.rows(), M.rows(), M.cols(), S.rows(), S.cols());
		//checkConvergenceCat2(EigVectsReduced, EigValsReduced, EigValsReduced.rows(), M, Minv, S, conv4);
		//printVector(conv4);
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
	*  @param	type		Desired type of SIM to perform. One of HSIM::MLEigenSolver::SIMType enum types.
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
	*  @param	type		Desired type of SIM to perform. One of HSIM::MLEigenSolver::SIMType enum types.
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
	*  @param	type		Desired type of SIM to perform. One of HSIM::MLEigenSolver::SIMType enum types.
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

	void MLEigenSolver::SIMinCPU(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol, int maxIter, int k) {}

	void MLEigenSolver::SIMinGPU(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol, int maxIter, int k) {}

	void MLEigenSolver::solve(int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset, double levelRatio, double basisSupport) {}

	void MLEigenSolver::solve(int numEigs, double tol, int maxIter, double levelRatio, double basisSupport, bool isWithShift) {}

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
	void MLEigenSolver::checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, int num, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& MKinv, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const
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

				double n1 = diff.transpose()*MKinv*diff;
				double n2 = Kq.transpose()*MKinv*Kq;
				//double n1 = diff.norm();
				//double n2 = Kq.norm();
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
	void MLEigenSolver::checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::MatrixXd& M, int num, bool isWithShift, double mu, Eigen::VectorXd& convRate) const
	{
		std::cout << "MU: " << mu << std::endl;		

		convRate.resize(A.rows());
		for (int i = 0; i < A.rows(); ++i) {
			double a2 = A(i) * A(i);
			double qTq = Q.col(i).transpose() * Q.col(i);
			if (isWithShift) qTq += mu * (2 * A(i) - mu);
			double rat = a2 / qTq;
			double con = abs(1.0 - rat);
			convRate(i) = sqrt(con);
		}
	}

	int MLEigenSolver::sturmCheck(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::VectorXd& eigvals)
	{
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> linSolver;
		double mu = eigvals(eigvals.size() - 1);
		linSolver.compute(SL - mu*ML);
		int negEntryD = (linSolver.vectorD().array() < 0.0).count();
		std::cout << "____computed modes:" << eigvals.rows() << "| Sturm's check: " << negEntryD << std::endl;
		return negEntryD;
	}
	
	void MLEigenSolver::solveBMG(int numEigs, double tol, int maxIter, double levelRatio, double basisSupport, bool isWithShift) {}

	void MLEigenSolver::BMGiter(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol, int maxIter, int k) {}
	
	///void MLEigenSolver::solveInversePowerMethod(unsigned int numEigs, double tol, int maxIter, bool isWithShift)
	///{
	///	/* [1] Set up the System matrices */
	///	constructMassMatrix();
	///	initiateAdM();
	///	constructStiffnessMatrix();
	///
	///	std::cout << "Soving using inverse power method, one by one \n ";
	///	std::cout << "--can be SUPER SLOW! \n";
	///	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> SpSolver;
	///	SpSolver.compute(S);
	///	Eigen::LDLT<Eigen::MatrixXd> DnSolver;
	///	Eigen::MatrixXd V;
	///	Eigen::VectorXd v, vTemp, vConst;
	///	Eigen::VectorXd eigVal(1);
	///	Eigen::VectorXd convRate(1);
	///
	///	EigVectsSIM.resize(S.rows(), numEigs);
	///	EigValsSIM.resize(numEigs);
	///	v.resize(S.rows());
	///	v.setConstant(1.0);
	///	double norm_ = v.transpose() * M * v;
	///	v = v / sqrt(norm_);
	///	EigVectsSIM.col(0) = v;
	///	EigValsSIM(0) = 0;
	///	//std::cout << "Eigval 0: " << EigValsSIM(0) << std::endl;
	///	//std::cout << "Eigvectors " << EigVectsSIM.col(0).head(10).transpose() << std::endl;
	///
	///	Eigen::MatrixXd X;
	///	Eigen::VectorXd vproj, q;
	///	for (int iEig = 1; iEig < numEigs; ++iEig) {
	///		printf("Solving eigenproblem # %d \n", iEig);
	///		///printf("Eig %d => construct constraints | ", iEig);
	///		X = EigVectsSIM.leftCols(iEig);
	///		Eigen::MatrixXd A = X.transpose() * M;
	///		///printf(" of size:%dx%d \n", A.rows(), A.cols());
	///		Eigen::SparseMatrix<double> K(S.rows() + A.rows(), S.rows() + A.rows());
	///		std::vector<Eigen::Triplet<double>> KTriplet;
	///		KTriplet.reserve(S.nonZeros() + 2 * A.nonZeros());
	///		///std::cout << "Const (1):\n" << A.bottomRows(iEig).leftCols(10) << std::endl;
	///
	///		//std::cout << "Prepare LHS \n";
	///		for (int k = 0; k < M.outerSize(); ++k) {
	///			for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
	///				KTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
	///			}
	///		}
	///
	///		for (int kRow = 0; kRow < A.rows(); ++kRow) {
	///			for (int kCol = 0; kCol < A.cols(); ++kCol) {
	///				KTriplet.push_back(Eigen::Triplet<double>(M.rows() + kRow, kCol, A(kRow, kCol)));
	///				KTriplet.push_back(Eigen::Triplet<double>(kCol, M.rows() + kRow, A(kRow, kCol)));
	///			}
	///		}
	///		K.setFromTriplets(KTriplet.begin(), KTriplet.end());
	///		KTriplet.clear(); KTriplet.shrink_to_fit();
	///		///std::cout << "Const (2):\n" << A.bottomRows(iEig).leftCols(10) << std::endl;
	///
	///		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> spSolverK;
	///		spSolverK.compute(K);
	///
	///		Eigen::VectorXd x(S.rows()); x.setConstant(0.05);
	///		Eigen::VectorXd g = x;
	///		Eigen::VectorXd h = A * x;
	///		Eigen::VectorXd b(g.rows() + h.rows());
	///		b << g, h;
	///		Eigen::VectorXd sol;
	///
	///		unsigned int iterCount = 0;
	///		bool isConverged = false;
	///
	///		generateRandomMatrix(S.rows(), 1, V);
	///		v = V.col(0);
	///		norm_ = v.transpose() * M * v;
	///		v = v / sqrt(norm_);
	///
	///		// special case for 1m triangles
	///		std::vector<double> tol_list{ 1e-1, 1e-2, 1e-4, 1e-6 };
	///
	///		do {
	///			///printf("__produced smooth vectors \n");
	///			vTemp = M * v;
	///			v = SpSolver.solve(vTemp);
	///
	///			///printf("__project to constraints vectors \n");
	///			b.head(g.rows()) = M * (x - v);
	///			sol = spSolverK.solve(b);
	///
	///			q = -1.0 * sol.head(g.rows()) + x;
	///			v = q;
	///
	///			//double a = max(v.maxCoeff(), abs(v.minCoeff()));				
	///			//v = v / a;
	///			//eigVal(0) = v.transpose() * v;
	///			norm_ = v.transpose() * M * v;
	///			v = v / sqrt(norm_);
	///
	///			Eigen::VectorXd eVectTemp = v;
	///			eigVal(0) = eVectTemp.transpose() * S * eVectTemp;
	///
	///			checkConvergenceCat2(eVectTemp, eigVal, 1, M, Minv, S, convRate);
	///			//checkConvergenceCat2(v, eigVal, 1, Id, Id, Sh, convRate);
	///			//printf("EigVal %d | --Iter %d | convergence check: %.5f | is converged? %s \n", iEig, iterCount, convRate(0), isConverged ? "true" : "false");
	///
	///
	///			iterCount++;
	///			///if (convRate(0) < tol || iterCount > maxIter) isConverged = true;
	///
	///
	///			if (convRate(0) < tol_list[3]) {
	///				EigValsSIM(iEig) = eigVal(0);
	///				EigVectsSIM.col(iEig) = v;
	///				string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[3]) + ".txt";
	///				writeDenseMatrixToFile(EigValsSIM, filename, true);
	///				filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[3]) + ".txt";
	///				writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
	///			}
	///			else if (convRate(0) < tol_list[2]) {
	///				EigValsSIM(iEig) = eigVal(0);
	///				EigVectsSIM.col(iEig) = v;
	///				string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[2]) + ".txt";
	///				writeDenseMatrixToFile(EigValsSIM, filename, true);
	///				filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[2]) + ".txt";
	///				writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
	///			}
	///			else if (convRate(0) < tol_list[1]) {
	///				EigValsSIM(iEig) = eigVal(0);
	///				EigVectsSIM.col(iEig) = v;
	///				string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[1]) + ".txt";
	///				writeDenseMatrixToFile(EigValsSIM, filename, true);
	///				filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[1]) + ".txt";
	///				writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
	///			}
	///			else if (convRate(0) < tol_list[0]) {
	///				EigValsSIM(iEig) = eigVal(0);
	///				EigVectsSIM.col(iEig) = v;
	///				string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[0]) + ".txt";
	///				writeDenseMatrixToFile(EigValsSIM, filename, true);
	///				filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[0]) + ".txt";
	///				writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
	///			}
	///
	///			//if (iterCount > maxIter) isConverged = true;
	///			if (convRate(0) < tol_list[3] || iterCount > maxIter) isConverged = true;
	///			printf("EigVal %d || %.8f | []Iter %d | convergence check: %.10f | is converged? %s \n", iEig, eigVal(0), iterCount, convRate(0), isConverged ? "true" : "false");
	///		} while (!isConverged);
	///
	///		EigValsSIM(iEig) = eigVal(0);
	///		EigVectsSIM.col(iEig) = v;
	///
	///		///string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol) + ".txt";
	///		///writeDenseMatrixToFile(EigValsSIM, filename, true);
	///		///filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol) + ".txt";
	///		///writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
	///	}
	///}

	void MLEigenSolver::solveInversePowerMethod(unsigned int numEigs, double tol, int maxIter, bool isWithShift)
	{		
		/* [1] Set up the System matrices */
		constructMassMatrix();
		initiateAdM();
		constructStiffnessMatrix();
		
		std::cout << "Soving using inverse power method, one by one \n ";
		std::cout << "--can be SUPER SLOW! \n";

		Eigen::SparseMatrix<double> S1, M1;
		Eigen::SparseMatrix<double> SelMat;

		if (hasBoundary && boundaryType == BoundaryCondition::BOUNDARY_DIRICHLET) {
			printf("This mesh has boundary, and will be treated accordingly \n");
			std::vector<Eigen::Triplet<double>> SelTriplet;
			SelTriplet.reserve(V.rows());
			unsigned int iCounter = 0;
			for (unsigned int i = 0; i < V.rows(); ++i) {
				if (!boundaryVertexCheck[i]) {
					SelTriplet.push_back(Eigen::Triplet<double>(i, iCounter, 1.0));
					iCounter++;
				}
			}
			SelMat.resize(V.rows(), iCounter);
			SelMat.setFromTriplets(SelTriplet.begin(), SelTriplet.end());
			M1 = SelMat.transpose() * M * SelMat;
			S1 = SelMat.transpose() * S * SelMat;
		}
		else {
			M1 = M;
			S1 = S;
		}

		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> SpSolver;
		SpSolver.compute(S1);
		Eigen::LDLT<Eigen::MatrixXd> DnSolver;
		Eigen::MatrixXd V;
		Eigen::VectorXd v, vTemp, vConst;
		Eigen::VectorXd eigVal(1);
		Eigen::VectorXd convRate(1);

		EigVectsSIM.resize(S1.rows(), numEigs);
		EigValsSIM.resize(numEigs);
		v.resize(S1.rows());
		v.setConstant(1.0);
		double norm_ = v.transpose() * M1 * v;
		v = v/sqrt(norm_);
		EigVectsSIM.col(0) = v;
		EigValsSIM(0) = 0;
		//std::cout << "Eigval 0: " << EigValsSIM(0) << std::endl;
		//std::cout << "Eigvectors " << EigVectsSIM.col(0).head(10).transpose() << std::endl;

		Eigen::MatrixXd X;
		Eigen::VectorXd vproj, q;
		for (int iEig = 1; iEig < numEigs; ++iEig) {
			printf("Solving eigenproblem # %d \n", iEig);
			///printf("Eig %d => construct constraints | ", iEig);
			X = EigVectsSIM.leftCols(iEig);
			Eigen::MatrixXd A = X.transpose() * M1;
			///printf(" of size:%dx%d \n", A.rows(), A.cols());
			Eigen::SparseMatrix<double> K(S1.rows() + A.rows(), S1.rows() + A.rows());
			std::vector<Eigen::Triplet<double>> KTriplet;
			KTriplet.reserve(S1.nonZeros() + 2 * A.nonZeros());
			///std::cout << "Const (1):\n" << A.bottomRows(iEig).leftCols(10) << std::endl;

			//std::cout << "Prepare LHS \n";
			for (int k = 0; k < M1.outerSize(); ++k) {
				for (Eigen::SparseMatrix<double>::InnerIterator it(M1, k); it; ++it) {
					KTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
				}
			}

			for (int kRow = 0; kRow < A.rows(); ++kRow) {
				for (int kCol = 0; kCol < A.cols(); ++kCol) {
					KTriplet.push_back(Eigen::Triplet<double>(M1.rows() + kRow, kCol, A(kRow, kCol)));
					KTriplet.push_back(Eigen::Triplet<double>(kCol, M1.rows() + kRow, A(kRow, kCol)));
				}
			}			
			K.setFromTriplets(KTriplet.begin(), KTriplet.end());
			KTriplet.clear(); KTriplet.shrink_to_fit();
			///std::cout << "Const (2):\n" << A.bottomRows(iEig).leftCols(10) << std::endl;

			Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> spSolverK;
			spSolverK.compute(K);

			Eigen::VectorXd x(S1.rows()); x.setConstant(0.05);
			Eigen::VectorXd g = x;
			Eigen::VectorXd h = A * x;
			Eigen::VectorXd b(g.rows() + h.rows());
			b << g, h;
			Eigen::VectorXd sol;

			unsigned int iterCount = 0;
			bool isConverged = false;

			generateRandomMatrix(S1.rows(), 1, V);
			v = V.col(0);
			norm_ = v.transpose() * M1 * v;
			v = v / sqrt(norm_);

			// special case for 1m triangles
			//std::vector<double> tol_list{ 1e-1, 1e-2, 1e-4, 1e-6 };

			do {				
				///printf("__produced smooth vectors \n");
				vTemp = M1 * v;
				v = SpSolver.solve(vTemp);

				///printf("__project to constraints vectors \n");
				b.head(g.rows()) = M1 * (x - v);
				sol = spSolverK.solve(b);

				q = -1.0 * sol.head(g.rows()) + x;
				v = q;

				//double a = max(v.maxCoeff(), abs(v.minCoeff()));				
				//v = v / a;
				//eigVal(0) = v.transpose() * v;
				norm_ = v.transpose() * M1 * v;
				v = v / sqrt(norm_);

				Eigen::VectorXd eVectTemp =  v;
				eigVal(0) = eVectTemp.transpose() * S1 * eVectTemp;

				checkConvergenceCat2(eVectTemp, eigVal, 1, M1, M1, S, convRate);
				//checkConvergenceCat2(v, eigVal, 1, Id, Id, Sh, convRate);
				//printf("EigVal %d | --Iter %d | convergence check: %.5f | is converged? %s \n", iEig, iterCount, convRate(0), isConverged ? "true" : "false");


				iterCount++;
				if (convRate(0) < tol || iterCount > maxIter) isConverged = true;


				//if (convRate(0) < tol_list[3]) {
				//	EigValsSIM(iEig) = eigVal(0);
				//	EigVectsSIM.col(iEig) = v;
				//	string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[3]) + ".txt";
				//	//writeDenseMatrixToFile(EigValsSIM, filename, true);
				//	filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[3]) + ".txt";
				//	//writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
				//} else if (convRate(0) < tol_list[2]) {
				//	EigValsSIM(iEig) = eigVal(0);
				//	EigVectsSIM.col(iEig) = v;
				//	string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[2]) + ".txt";
				//	//writeDenseMatrixToFile(EigValsSIM, filename, true);
				//	filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[2]) + ".txt";
				//	//writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
				//} else if (convRate(0) < tol_list[1]) {
				//	EigValsSIM(iEig) = eigVal(0);
				//	EigVectsSIM.col(iEig) = v;
				//	string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[1]) + ".txt";
				//	//writeDenseMatrixToFile(EigValsSIM, filename, true);
				//	filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[1]) + ".txt";
				//	//writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
				//} else if (convRate(0) < tol_list[0]) {
				//	EigValsSIM(iEig) = eigVal(0);
				//	EigVectsSIM.col(iEig) = v;
				//	string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[0]) + ".txt";
				//	//writeDenseMatrixToFile(EigValsSIM, filename, true);
				//	filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol_list[0]) + ".txt";
				//	//writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
				//}

				//if (iterCount > maxIter) isConverged = true;
				//if (convRate(0) < tol_list[3] || iterCount > maxIter) isConverged = true;
				printf("EigVal %d || %.8f | []Iter %d | convergence check: %.10f | is converged? %s \n", iEig, eigVal(0), iterCount, convRate(0), isConverged ? "true" : "false");
			} while (!isConverged);

			EigValsSIM(iEig) = eigVal(0);
			EigVectsSIM.col(iEig) = v;

			///string filename = "../../Data/eigvals-analysis/sphere_EigVals_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol) + ".txt";
			///writeDenseMatrixToFile(EigValsSIM, filename, true);
			///filename = "../../Data/eigvals-analysis/sphere_EigVects_IPM_" + to_string(V.rows()) + "_" + to_string(numEigs) + "_" + to_string(tol) + ".txt";
			///writeDenseMatrixToFile(Eigen::MatrixXd(EigVectsSIM.leftCols(iEig)), filename, true);
		}
		if (hasBoundary && boundaryType == BoundaryCondition::BOUNDARY_DIRICHLET) {
			printf("Multiply with selectro for to map to orignal mesh \n");
			EigVectsSIM = SelMat * EigVectsSIM;
		}
	}

	//void MLEigenSolver::solveInversePowerMethod(unsigned int numEigs, double tol, int maxIter, bool isWithShift)
	//{
	//	/* [1] Set up the System matrices */
	//	constructMassMatrix();
	//	initiateAdM();
	//	constructStiffnessMatrix();
	//
	//	Eigen::SparseMatrix<double> Sh = MhNeg * S * MhNeg;
	//	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> SpSolver;
	//	SpSolver.compute(Sh);
	//	Eigen::MatrixXd V;
	//	Eigen::VectorXd v;
	//	Eigen::VectorXd eigVal(1);
	//	Eigen::VectorXd convRate(1);
	//	Eigen::SparseMatrix<double> Id(Sh.rows(), Sh.rows()); Id.setIdentity();
	//
	//	EigVectsSIM.resize(Sh.rows(), numEigs);
	//	EigValsSIM.resize(numEigs);
	//	v.resize(Sh.rows());
	//	v.setConstant(1.0);
	//	v.normalize();
	//	EigVectsSIM.col(0) = v;			
	//
	//	Eigen::MatrixXd XTS;
	//	Eigen::MatrixXd XTSX;
	//	Eigen::VectorXd vproj, q;
	//	Eigen::LDLT<Eigen::MatrixXd> DnSolver;
	//	for (int iEig = 1; iEig < numEigs; ++iEig) {
	//		SpSolver.compute(Sh - EigValsSIM(iEig - 1) * Id);
	//		// orthogonal Constraints
	//		XTS = EigVectsSIM.leftCols(iEig).transpose();
	//		XTSX = XTS * EigVectsSIM.leftCols(iEig);
	//		std::cout << "Gram Mat: \n" << XTSX << std::endl;
	//
	//		DnSolver.compute(XTSX);
	//
	//		// Random matrix
	//		generateRandomMatrix(Sh.rows(), 1, V);
	//		v = V.col(0);
	//		v.normalize();
	//
	//
	//		unsigned int iterCount = 0;
	//		bool isConverged = false;
	//
	//		do {
	//			v.normalize();
	//			v = SpSolver.solve(v);
	//			double a = max(v.maxCoeff(), abs(v.minCoeff()));
	//
	//			eigVal(0) = 1.0 / a;
	//			v = v/(a-EigValsSIM(iEig-1));
	//
	//			vproj = XTS * v;
	//			q = DnSolver.solve(vproj);
	//			std::cout << "Dense solution: " << q.transpose() << std::endl;
	//			v = v - EigVectsSIM.leftCols(iEig) * q;
	//			Eigen::VectorXd eVectTemp = MhNeg * v;
	//			checkConvergenceCat2(eVectTemp, eigVal, 1, M, Id, S, convRate);
	//			printf("EigVal %d | []Iter %d | convergence check: %.5f | is converged? %s \n", iEig, iterCount, convRate(0), isConverged ? "true" : "false");
	//			checkConvergenceCat2(v, eigVal, 1, Id, Id, Sh, convRate);
	//			printf("EigVal %d | --Iter %d | convergence check: %.5f | is converged? %s \n", iEig, iterCount, convRate(0), isConverged ? "true" : "false");
	//
	//
	//			iterCount++;
	//			if (convRate(0) < tol || iterCount > maxIter) isConverged = true;
	//
	//
	//		} while (!isConverged);
	//		
	//		EigValsSIM(iEig) = eigVal(0);
	//		EigVectsSIM.col(iEig) = MhNeg * v;
	//	}		
	//
	//	std::cout << "M-norm: " << EigVectsSIM.transpose() * M * EigVectsSIM << std::endl;
	//}
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
		/*Creating transformation of S1Q1=AMQ1 to S2Q2=AQ2
		**where S2=M^-1/2 * S1 * M^-1/2 */

		Eigen::SparseMatrix<double> MhalfNeg(M.rows(), M.cols());
		Eigen::SparseMatrix<double> Strans(S.rows(), S.cols());
		vector<Eigen::Triplet<double>> MTriplet;
		MTriplet.reserve(M.nonZeros());

		Eigen::VectorXd IdVect(S.rows());
		IdVect.setOnes();
		Eigen::SparseMatrix<double> Mtrans;

		for (int k = 0; k < M.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
				double val = 1.0 / sqrt(it.value());
				MTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), val));
			}
		}
		MhalfNeg.setFromTriplets(MTriplet.begin(), MTriplet.end());
		MTriplet.clear();
		MTriplet.shrink_to_fit();
		Strans = MhalfNeg * S * MhalfNeg;
		Mtrans = MhalfNeg * M * MhalfNeg;
		double val = IdVect.transpose() * Mtrans * IdVect;
		std::cout << "Identity check: " << val << std::endl;
		std::cout << "M diag: " << M.diagonal().head(50).transpose() << std::endl;
		std::cout << "M^{-1/2} diag: " << MhalfNeg.diagonal().head(50).transpose() << std::endl;
		std::cout << "M trans: " << Mtrans.diagonal().head(50).transpose() << std::endl;

		int numIterUsed;
		switch (solverType)
		{
		case EigenSolverType::EIG_SOLVER_MATLAB:
			std::cout << "Compute in Matlab \n";
			computeEigenMatlab(S, M, numEigs, EigVectsRef, EigValsRef);
			//computeEigenMatlab2(S, M, numEigs, EigVectsRef, EigValsRef);
			break;
		case EigenSolverType::EIG_SOLVER_SPECTRA:
			//computeEigenSpectraSparse(S, M, numEigs, EigVectsRef, EigValsRef, numIterUsed, 1, maxIters, tol);
			computeEigenSpectraSparse(Strans, numEigs, EigVectsRef, EigValsRef, numIterUsed, 1, maxIters, tol);
			EigVectsRef = MhalfNeg * EigVectsRef;
			std::cout << "Number of iterations: " << numIterUsed << std::endl;
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
		int numIterUsed;
		switch (solverType)
		{
		case EigenSolverType::EIG_SOLVER_MATLAB:
			computeEigenMatlab2(SLoc, MLoc, numEigs, EigVectsLoc, EigValsRef);
			break;
		case EigenSolverType::EIG_SOLVER_SPECTRA:
			computeEigenSpectraSparse(SLoc, MLoc, numEigs, EigVectsLoc, EigValsRef, numIterUsed, 0, 500, 1e-2);
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
	*  @param	eigType		Type of the eigenvector to show, from HSIM::MLEigenSolver::EigenVectorType enum type.
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

	void MLEigenSolver::visualizeHierarchy(igl::opengl::glfw::Viewer &viewer, int k, int sourceID)
	{
		Eigen::VectorXi VS, FS, VT, FT;
		Eigen::MatrixXd IsoV, IsoE;
		bool isUsingHeat = true;
		//VS.resize(1);
		//VS << sourceID;
		//
		Eigen::VectorXd dist; 
		if (!isUsingHeat) {
			dist.resize(V.rows());
			dist.setConstant(std::numeric_limits<double>::max());
			ComputeDijkstra(V, Sample(sourceID), AdM, dist);
		}
		else {
			GeodesicFromHeat(V, F, S, M, Sample(sourceID), dist);
			//VS.resize(1);
			//VS << Sample(sourceID);
			//VT.setLinSpaced(V.rows(), 0, V.rows() - 1);
			//igl::exact_geodesic(V, F, VS, FS, VT, FT, dist);
		}

		Eigen::MatrixXd FCol(F.rows(), F.cols());
		FCol.setConstant(0.93333333);		

		std::cout << "Dist: " << dist.head(10).transpose() << std::endl;
		std::cout << "Cutoff" << maxNeighDist[k] << std::endl;		

		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < F.cols(); ++j) {
				if (dist(F(i, j)) < maxNeighDist[k]) {
					if(k==1)
						//FCol.row(i) << 251.0/255.0, 143.0/255.0, 103.0/255.0;		// red-sih
						FCol.row(i) << 247.0/255.0, 202.0/255.0, 205.0/255.0;		// red-sih
						//FCol.row(i) << 248.0/255.0, 173.0/255.0, 157.0/255.0;		// red-sih
					else 
						//FCol.row(i) << 233.0/255.0, 255.0/255.0, 249.0/255.0;
						FCol.row(i) << 158.0/255.0, 216.0/255.0, 219.0/255.0;
						//FCol.row(i) << 108.0/255.0, 212.0/255.0, 255.0/255.0;		// blue-sih					
				}
			}
		}
		std::cout << "Level:" << k << "| maxDist:" << maxNeighDist[k] << std::endl;
		viewer.data().set_colors(FCol);		
		viewer.data().points.resize(0, 6);
		if(k==1)
			viewer.data().add_points(V.row(Sample(sourceID)), Eigen::RowVector3d(221.0/255.0, 4.0/255.0, 38.0/255.0));
		else 
			viewer.data().add_points(V.row(Sample(sourceID)), Eigen::RowVector3d(29.0/255.0, 51.0/255.0, 84.0/255.0));
		

		//igl::isolines(V, F, dist, 10, IsoV, IsoE);
	}

	void MLEigenSolver::visualizeHierarchy(igl::opengl::glfw::Viewer &viewer, int levels, int sample1, int sample2)
	{
		Eigen::VectorXi VS, FS, VT, FT;
		Eigen::MatrixXd IsoV, IsoE;
		bool isUsingHeat = true;

		Eigen::MatrixXd FCol(F.rows(), F.cols());
		FCol.setConstant(0.93333333);

		/* Domain viz for level 2 */
		Eigen::VectorXd dist;
		if (!isUsingHeat) {
			dist.resize(V.rows());
			dist.setConstant(std::numeric_limits<double>::max());
			ComputeDijkstra(V, Sample(sample2), AdM, dist);
		}
		else {
			GeodesicFromHeat(V, F, S, M, Sample(sample2), dist);
			//VS.resize(1);
			//VS << Sample(sourceID);
			//VT.setLinSpaced(V.rows(), 0, V.rows() - 1);
			//igl::exact_geodesic(V, F, VS, FS, VT, FT, dist);
		}

		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < F.cols(); ++j) {
				if (dist(F(i, j)) < maxNeighDist[1]) {
					//FCol.row(i) << 251.0/255.0, 143.0/255.0, 103.0/255.0;		// red-sih
					//FCol.row(i) << 247.0 / 255.0, 202.0 / 255.0, 205.0 / 255.0;		// red-sih
					//FCol.row(i) << 248.0/255.0, 173.0/255.0, 157.0/255.0;		// red-sih		
					//FCol.row(i) << 224.0/255.0, 119.0/255.0, 125.0/255.0;		// red-sih	
					//FCol.row(i) << 254.0/255.0, 95.0/255.0, 85.0/255.0;		// red-sih	
					FCol.row(i) << 239.0/255.0, 169.0/255.0, 174.0/255.0;		// red-sih	
				}
			}
		}
		std::cout << "Level 1 | maxDist:" << maxNeighDist[1] << std::endl;
		viewer.data().add_points(V.row(Sample(sample2)), Eigen::RowVector3d(221.0 / 255.0, 4.0 / 255.0, 38.0 / 255.0));
		

		/* LEVEL 0 */
		if (!isUsingHeat) {
			dist.resize(V.rows());
			dist.setConstant(std::numeric_limits<double>::max());
			ComputeDijkstra(V, Sample(sample1), AdM, dist);
		}
		else {
			GeodesicFromHeat(V, F, S, M, Sample(sample1), dist);
			//VS.resize(1);
			//VS << Sample(sourceID);
			//VT.setLinSpaced(V.rows(), 0, V.rows() - 1);
			//igl::exact_geodesic(V, F, VS, FS, VT, FT, dist);
		}

		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < F.cols(); ++j) {
				if (dist(F(i, j)) < maxNeighDist[0]) {
					//FCol.row(i) << 233.0/255.0, 255.0/255.0, 249.0/255.0;
					//FCol.row(i) << 158.0 / 255.0, 216.0 / 255.0, 219.0 / 255.0;
					FCol.row(i) << 108.0/255.0, 212.0/255.0, 255.0/255.0;		// blue-sih					
				}
			}
		}
		std::cout << "Level: 0| maxDist:" << maxNeighDist[0] << std::endl;
		viewer.data().set_colors(FCol);
		viewer.data().add_points(V.row(Sample(sample1)), Eigen::RowVector3d(29.0 / 255.0, 51.0 / 255.0, 84.0 / 255.0));


		//igl::isolines(V, F, dist, 10, IsoV, IsoE);
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

	void  MLEigenSolver::setLevel(int n)
	{
		level = n;
	}

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
	const MLEigenSolver::BoundaryCondition MLEigenSolver::getBoundaryType() const { return this->boundaryType; }
	void MLEigenSolver::setBoundaryType(const BoundaryCondition& boundary) {  this->boundaryType = boundary;  }

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

	void MLEigenSolver::cusparseDcsrm2(const Eigen::SparseMatrix<double> SM, const Eigen::MatrixXd& B, Eigen::MatrixXd& X)
	{
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;
		std::cout << "Solving in GPU! \n";
		t1 = chrono::high_resolution_clock::now();

		cusparseHandle_t			cuSpHandle;
		cusparseStatus_t			cuSpStat;
		cudaError_t					cudaStat = cudaSuccess;
		cusparseOperation_t			opNoTrans = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;
		cusparseOperation_t			opTrans = cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE;
		csrsm2Info_t				info = NULL;
		const cusparseSolvePolicy_t policy = cusparseSolvePolicy_t::CUSPARSE_SOLVE_POLICY_NO_LEVEL;
		size_t						lworkInBytes = 0;
		char*						d_work = NULL;
		cudaStream_t				stream = NULL;

		//Eigen::SparseMatrix<double> SMTri = SM.triangularView<Eigen::Lower>();
		//std::cout << "S: " << SM.nonZeros() << "|| STri: " << SMTri.nonZeros() << std::endl;
		int m_S = SM.rows();
		int n_S = SM.cols();
		int nnz_S = SM.nonZeros();
		int m_X = B.rows();
		int n_X = B.cols();	
		int m_B = B.rows();
		int n_B = B.cols();

		// Buffer
		const double alpha = 1.0;
		const double beta = 0.0;
		const int algo = 0; /* non-block version */
		cusparseMatDescr_t matSM;
		cudaStat = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		cusparseCreate(&cuSpHandle);
		cuSpStat = cusparseSetStream(cuSpHandle, stream);
		cusparseCreateCsrsm2Info(&info);

		cuSpStat = cusparseCreateMatDescr(&matSM);
		cusparseSetMatIndexBase(matSM, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(matSM, cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(matSM, cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER);
		cusparseSetMatDiagType(matSM, cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT);

		double* h_B = (double*)malloc(m_B * n_B * sizeof(double));
		double* h_X = (double*)malloc(m_X * n_X * sizeof(double));

		// Set-up SM on GPU (Sparse)
		double* d_csrVal_S; cudaMalloc(&d_csrVal_S, nnz_S * sizeof(double));
		int* d_csrRowPtr_S; cudaMalloc(&d_csrRowPtr_S, (m_S + 1) * sizeof(int));
		int* d_csrColInd_S; cudaMalloc(&d_csrColInd_S, nnz_S * sizeof(int));
		// Set-up X on GPU (Dense)
		double* d_X; cudaMalloc(&d_X, m_X * n_X * sizeof(double));
		double* d_B; cudaMalloc(&d_B, m_X * n_X * sizeof(double));


		// Copy to GPU
		cudaStat = cudaMemcpy(d_csrVal_S, SM.valuePtr(), nnz_S * sizeof(double), cudaMemcpyHostToDevice);
		cudaStat = cudaMemcpy(d_csrRowPtr_S, SM.outerIndexPtr(), (m_S + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaStat = cudaMemcpy(d_csrColInd_S, SM.innerIndexPtr(), nnz_S * sizeof(int), cudaMemcpyHostToDevice);
		cudaStat = cudaMemcpy(d_B, B.data(), m_B * n_B * sizeof(double), cudaMemcpyHostToDevice);

		cuSpStat = cusparseDcsrsm2_bufferSizeExt(cuSpHandle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m_S, n_X, nnz_S, &alpha, matSM, d_csrVal_S, d_csrRowPtr_S, d_csrColInd_S, d_B, m_B, info, policy, &lworkInBytes);
		std::cout << "allocate memory: " << cuSpStat << std::endl;

		// Analyze
		cudaMalloc((void**)&d_work, lworkInBytes);
		cuSpStat = cusparseDcsrsm2_analysis(cuSpHandle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m_S, n_X, nnz_S, &alpha, matSM, d_csrVal_S, d_csrRowPtr_S, d_csrColInd_S, d_B, m_X, info, policy, d_work);
		std::cout << "Factorize: " << cuSpStat << std::endl;
		// Solve
		cuSpStat = cusparseDcsrsm2_solve(cuSpHandle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m_S, n_X, nnz_S, &alpha, matSM, d_csrVal_S, d_csrRowPtr_S, d_csrColInd_S, d_B, m_X, info, policy, d_work);
		std::cout << "Solve system: " << cuSpStat << std::endl;

		cudaStat = cudaMemcpy(h_X, d_B, m_X * n_X * sizeof(double), cudaMemcpyDeviceToHost);

		// Format C to Eigen::MatrixXd
		Eigen::MatrixXd CMat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_X, m_X, n_X, Eigen::OuterStride<>(m_X));		
		std::cout << "GPU (X from (S-mu*M)X=M*X0: \n" << CMat.topLeftCorner(10, 10) << std::endl;

		X = CMat;

		if (h_B)			free(h_B);
		if (h_X)			free(h_X);
		if (d_X)			cudaFree(d_X);
		if (d_B)			cudaFree(d_B);
		if(d_csrVal_S   )	cudaFree(d_csrVal_S);
		if (d_csrRowPtr_S)	cudaFree(d_csrRowPtr_S);
		if (d_csrColInd_S)	cudaFree(d_csrColInd_S);
		if(cuSpHandle)		cusparseDestroy(cuSpHandle);
		cudaDeviceReset();

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "...in " << duration.count() << " seconds." << endl;
	}

	void MLEigenSolver::cusolverSP(const Eigen::SparseMatrix<double> SM, const Eigen::MatrixXd& B, Eigen::MatrixXd& X)
	{
		cusolverSpHandle_t		cuSolverHandle;
		cusolverStatus_t		cuSolverStat = cusolverStatus_t::CUSOLVER_STATUS_SUCCESS;
		cusparseStatus_t		cuSpStat;
		cudaError_t				cudaStat;
		cudaStream_t			stream = NULL;
	
		int m_S = SM.rows();
		int n_S = SM.cols();
		int nnz_S = SM.nonZeros();
		int m_X = B.rows();
		int n_X = B.cols();
		int m_B = B.rows();
		int n_B = B.cols();
	
		double* h_B = (double*)malloc(m_B * n_B * sizeof(double));
		double* h_X = (double*)malloc(m_X * n_X * sizeof(double));
	
		// Set-up SM on GPU (Sparse)
		std::cout << "allocating memory: ";
		double* d_csrVal_S; cudaStat = cudaMalloc(&d_csrVal_S, nnz_S * sizeof(double));			std::cout << cudaStat << "| ";
		int* d_csrRowPtr_S; cudaStat = cudaMalloc(&d_csrRowPtr_S, (m_S + 1) * sizeof(int));		std::cout << cudaStat << "| ";
		int* d_csrColInd_S; cudaStat = cudaMalloc(&d_csrColInd_S, nnz_S * sizeof(int));			std::cout << cudaStat << "| ";
		// Set-up X on GPU (Dense)
		double* d_X; cudaStat = cudaMalloc(&d_X, m_X * n_X * sizeof(double));					std::cout << cudaStat << "| ";
		double* d_B; cudaStat = cudaMalloc(&d_B, m_X * n_X * sizeof(double));					std::cout << std::endl;
	
		cuSolverStat = cusolverSpCreate(&cuSolverHandle);
		std::cout << "Creating handle: " << cuSolverStat << std::endl;
		cudaStat = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		std::cout << "Creating stream: " << cudaStat << std::endl;
	
		cuSolverStat = cusolverSpSetStream(cuSolverHandle, stream);
		assert(CUSOLVER_STATUS_SUCCESS == status);
	
	
		// Copy to GPU
		std::cout << "Copy to GPU: ";
		cudaStat = cudaMemcpy(d_csrVal_S, SM.valuePtr(), nnz_S * sizeof(double), cudaMemcpyHostToDevice);				std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpy(d_csrRowPtr_S, SM.outerIndexPtr(), (m_S + 1) * sizeof(int), cudaMemcpyHostToDevice);		std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpy(d_csrColInd_S, SM.innerIndexPtr(), nnz_S * sizeof(int), cudaMemcpyHostToDevice);			std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpy(d_B, B.data(), m_B * n_B * sizeof(double), cudaMemcpyHostToDevice);						std::cout << std::endl;		
	
		cusparseMatDescr_t matSM;
		cuSpStat = cusparseCreateMatDescr(&matSM);
		cuSpStat = cusparseSetMatIndexBase(matSM, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
		cuSpStat = cusparseSetMatType(matSM, cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL);
		cuSpStat = cusparseSetMatFillMode(matSM, cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER);
		cuSpStat = cusparseSetMatDiagType(matSM, cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT);
		std::cout << "Creating matrix description: " << cuSpStat << std::endl;
	
	
		int isSym;
		cuSolverStat = cusolverSpXcsrissymHost(cuSolverHandle, m_S, n_X, matSM, SM.outerIndexPtr(), SM.outerIndexPtr(), SM.innerIndexPtr(), &isSym);
		std::cout << "Symmetry? " << isSym << std::endl;
	
		double tol = 1e-2;
		int reorder = 2;
		int* d_sing = NULL;
		cudaStat = cudaMalloc(&d_sing, 1 * sizeof(int));
		int* h_sing = (int*)malloc(sizeof(int));
	
		//cuSolverStat = cusolverSpDcsrlsvcholHost(cuSolverHandle, m_S, nnz_S, matSM, SM.valuePtr(), SM.outerIndexPtr(), SM.innerIndexPtr(), B.data(), tol, reorder, h_X, d_sing);		
		cuSolverStat = cusolverSpDcsrlsvchol(cuSolverHandle, m_S, nnz_S, matSM, d_csrVal_S, d_csrRowPtr_S, d_csrColInd_S, d_B, tol, reorder, d_X, d_sing);
		std::cout << "Solving linear system: " << cuSolverStat << std::endl;
	
		//cudaStat = cudaMemcpy(h_sing, d_sing, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "Num of singularities: " << *d_sing << std::endl;
	
		cudaStat = cudaMemcpy(h_X, d_X, m_X * n_X * sizeof(double), cudaMemcpyDeviceToHost);
		std::cout << "Copy to CPU: " << cudaStat << std::endl;
	
		X.resize(m_X, n_X);
		X = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_X, m_X, n_X, Eigen::OuterStride<>(m_X));
		std::cout << "GPU (X from (S-mu*M)X=M*X0: \n" << X.topLeftCorner(10, 10) << std::endl;
	}


	void printMatrix(int m, int n, const double* A, int lda, const char* name)
	{
		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				double Areg = A[row + col * lda];
				printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
			}
		}
	}

	void MLEigenSolver::cuSolverCholTest(Eigen::SparseMatrix<double> SM, Eigen::MatrixXd& B, Eigen::MatrixXd& X)
	{
		SM.makeCompressed();
		struct  testOpts {
			char* sparse_mat_filename;   // by switch -F<filename>
			const char* testFunc; // by switch -R<name>
			const char* reorder; // by switch -P<name>
			int lda; // by switch -lda<int>
		};
	
		struct testOpts opts;
		opts.testFunc = "chol";
		opts.reorder = "symamd";
		opts.lda = SM.rows();
		cusolverSpHandle_t handle = NULL;
		cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
		cudaStream_t stream = NULL;
		cusparseMatDescr_t descrA = NULL;
		cusolverStatus_t cuSolverStat = cusolverStatus_t::CUSOLVER_STATUS_SUCCESS;
		cusparseStatus_t		cuSpStat;
		cudaError_t				cudaStat;
		
	
		int rowsA	= SM.rows(); /* number of rows of A */
		int colsA	= SM.cols(); /* number of columns of A */
		int nnzA	= SM.nonZeros(); /* number of nonzeros of A */
		int baseA	= 0; /* base index in CSR format */
	
		/* CSR(A) from I/O */
		int* h_csrRowPtrA = NULL; (int*)malloc((colsA+1) * sizeof(int));	 //S.outerIndexPtr();
		int* h_csrColIndA = NULL; (int*)malloc(nnzA* sizeof(int));			//S.innerIndexPtr();
		double* h_csrValA = NULL; (double*)malloc(nnzA* sizeof(double));	//S.valuePtr();
	
		double* h_z ; (double*)malloc(rowsA * sizeof(double));  /* z = B \ (Q*b) */
		double* h_x ; (double*)malloc(rowsA * sizeof(double));  /* x = A \ b */
		double* h_b ; (double*)malloc(rowsA * sizeof(double));  // B.col(0).data(); /* b = ones(n,1) */
		double* h_Qb; (double*)malloc(rowsA * sizeof(double));  /* Q*b */
		double* h_r ; (double*)malloc(rowsA * sizeof(double));  /* r = b - A*x */
	
		int* h_Q = NULL; /* <int> n */
						 /* reorder to reduce zero fill-in */
						 /* Q = symrcm(A) or Q = symamd(A) */
		/* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
		int* h_csrRowPtrB = NULL; (int*)malloc((colsA + 1) * sizeof(int));  /* <int> n+1 */
		int* h_csrColIndB = NULL; (int*)malloc(nnzA * sizeof(int));			/* <int> nnzA */
		double* h_csrValB = NULL; (double*)malloc(nnzA * sizeof(double));	/* <double> nnzA */
		int* h_mapBfromA  = NULL; (int*)malloc(nnzA * sizeof(int));			/* <int> nnzA */
	
		size_t size_perm = 0;
		void* buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */
	
	/* device copy of A: used in residual evaluation */
		int* d_csrRowPtrA = NULL;
		int* d_csrColIndA = NULL;
		double* d_csrValA = NULL;
	
		/* device copy of B: used in B*z = Q*b */
		int* d_csrRowPtrB = NULL;
		int* d_csrColIndB = NULL;
		double* d_csrValB = NULL;
	
		int* d_Q = NULL; /* device copy of h_Q */
		double* d_z = NULL; /* z = B \ Q*b */
		double* d_x = NULL; /* x = A \ b */
		double* d_b = NULL; /* a copy of h_b */
		double* d_Qb = NULL; /* a copy of h_Qb */
		double* d_r = NULL; /* r = b - A*x */
	
		double tol = std::numeric_limits<double>::epsilon()/1000;
		const int reorder = 0; /* no reordering */
		int singularity = 0; /* -1 if A is invertible under tol. */
	
		/* the constants are used in residual evaluation, r = b - A*x */
		const double minus_one = -1.0;
		const double one = 1.0;
	
		double b_inf = 0.0;
		double x_inf = 0.0;
		double r_inf = 0.0;
		double A_inf = 0.0;
		int errors = 0;
		int issym = 0;
	
		double start, stop;
		double time_solve_cpu;
		double time_solve_gpu;
	
		cuSolverStat	= cusolverSpCreate(&handle);			std::cout << "Create handle: " << cuSolverStat << std::endl;
		cuSpStat		= cusparseCreate(&cusparseHandle);		std::cout << "Create handle: " << cuSpStat << std::endl;
	
	 	cudaStat = cudaStreamCreate(&stream);
		std::cout << "Create stream: " << cudaStat << std::endl;
	
		/* bind stream to cusparse and cusolver*/
		cuSolverStat	= cusolverSpSetStream(handle, stream);				std::cout << "set handle + stream : " << cuSolverStat << std::endl;
		cuSpStat		= cusparseSetStream(cusparseHandle, stream);		std::cout << "set handle + stream : " << cuSpStat << std::endl;
	
		std::cout << "Matrix stat: ";
		cusparseCreateMatDescr(&descrA);
		cuSpStat = cusparseSetMatIndexBase(descrA, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);  std::cout<<cuSpStat << ": ";
		cuSpStat = cusparseSetMatType(descrA, cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL);	std::cout<<cuSpStat << ": ";
		cuSpStat = cusparseSetMatFillMode(descrA, cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER);	std::cout<<cuSpStat << ": ";
		cuSpStat = cusparseSetMatDiagType(descrA, cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT);	std::cout<<cuSpStat << "\n ";
	
		h_z = (double*)malloc(sizeof(double) * colsA);
		h_x = (double*)malloc(sizeof(double) * colsA);
		h_b = (double*)malloc(sizeof(double) * rowsA);
		h_Qb = (double*)malloc(sizeof(double) * rowsA);
		h_r = (double*)malloc(sizeof(double) * rowsA);
		h_b = B.col(0).data();
	
	
		h_Q = (int*)malloc(sizeof(int) * colsA);
		h_csrRowPtrB = (int*)malloc(sizeof(int) * (rowsA + 1));
		h_csrColIndB = (int*)malloc(sizeof(int) * nnzA);
		h_csrValB = (double*)malloc(sizeof(double) * nnzA);
		h_mapBfromA = (int*)malloc(sizeof(int) * nnzA);
	
		h_csrRowPtrA = (int*)malloc(sizeof(int) * (rowsA + 1));
		h_csrColIndA = (int*)malloc(sizeof(int) * nnzA);
		h_csrValA = (double*)malloc(sizeof(double) * nnzA);
		h_csrRowPtrA = SM.outerIndexPtr();
		h_csrColIndA = SM.innerIndexPtr();
		h_csrValA = SM.valuePtr();		
	
	
		std::cout << "Allocating memory in GPU" << std::endl;
		cudaStat = cudaMalloc((void**)&d_csrRowPtrA, sizeof(int)* (rowsA + 1));        std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_csrColIndA, sizeof(int)* nnzA);		       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_csrValA, sizeof(double)* nnzA);		       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_csrRowPtrB, sizeof(int)* (rowsA + 1));	       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_csrColIndB, sizeof(int)* nnzA);		       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_csrValB, sizeof(double)* nnzA);		       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_Q, sizeof(int)* colsA);				       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_z, sizeof(double)* colsA);				       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_x, sizeof(double)* colsA);				       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_b, sizeof(double)* rowsA);				       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_Qb, sizeof(double)* rowsA);			       std::cout << cudaStat << "| ";
		cudaStat = cudaMalloc((void**)&d_r, sizeof(double)* rowsA);				       std::cout << cudaStat << std::endl;
	
		cuSolverStat = cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrRowPtrA+1, h_csrColIndA, &issym);
		//cuSolverStat = cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA, SM.outerIndexPtr(), SM.outerIndexPtr() + 1, SM.innerIndexPtr(), &issym);
		std::cout << issym << "|| Check symmetry: " << cuSolverStat << std::endl;
	
		if (!issym) {
			printf("Error: A has no symmetric pattern, please use LU or QR \n");
		}
	
		printf("step 2: reorder the matrix A to minimize zero fill-in\n");
		printf("        if the user choose a reordering by -P=symrcm, -P=symamd or -P=metis\n");
		cuSolverStat = cusolverSpXcsrsymrcmHost(handle, rowsA, nnzA,descrA, h_csrRowPtrA, h_csrColIndA,h_Q);
		std::cout << "Reordereing the matrix: " << cuSolverStat << std::endl;
	
		printf("step 2.2: B = A(Q,Q) \n");
	
		memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
		memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);
	
		std::cout << "Reordering the RHS vectors" << std::endl;
		 cusolverSpXcsrperm_bufferSizeHost(
			handle, rowsA, colsA, nnzA,
			descrA, h_csrRowPtrB, h_csrColIndB,
			h_Q, h_Q,
			&size_perm);
	
		if (buffer_cpu)
		{
			free(buffer_cpu);
		}
		buffer_cpu = (void*)malloc(sizeof(char) * size_perm);
	
		/* h_mapBfromA = Identity */
		for (int j = 0; j < nnzA; j++)
		{
			h_mapBfromA[j] = j;
		}
		cuSolverStat = cusolverSpXcsrpermHost(
			handle, rowsA, colsA, nnzA,
			descrA, h_csrRowPtrB, h_csrColIndB,
			h_Q, h_Q,
			h_mapBfromA,
			buffer_cpu);
	
		std::cout << "Permutation matrix: " << cuSolverStat << std::endl;
	
		/* B = A( mapBfromA ) */
		for (int j = 0; j < nnzA; j++)
		{
			h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
		}		
	
		/* h_Qb = b(Q) */
		for (int row = 0; row < rowsA; row++) {
			h_Qb[row] = h_b[h_Q[row]];
		}
	
		printf("step 4: prepare data on device\n");
		cudaStat = cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice, stream); std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice, stream);		   std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice, stream);		   std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice, stream); std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_csrColIndB, h_csrColIndB, sizeof(int) * nnzA, cudaMemcpyHostToDevice, stream);		   std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double) * nnzA, cudaMemcpyHostToDevice, stream);		   std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice, stream);					   std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_Qb, h_Qb, sizeof(double) * rowsA, cudaMemcpyHostToDevice, stream);					   std::cout << cudaStat << "| ";
		cudaStat = cudaMemcpyAsync(d_Q, h_Q, sizeof(int) * rowsA, cudaMemcpyHostToDevice, stream);						   std::cout << cudaStat << "| ";
																														   std::cout << cudaStat << std::endl;
		printf("step 7: solve A*x = b on GPU\n");
	
		/* solve B*z = Q*b */
		cusolverSpDcsrlsvchol(
				handle, rowsA, nnzA,
				descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
				d_Qb, tol, reorder, d_z, &singularity);
	
		
		cudaDeviceSynchronize();
		std::cout << "Singularity: " << singularity << std::endl;
		if (0 <= singularity)
		{
			printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
		}
		/* Q*x = z */
		cusparseDsctr(cusparseHandle,
			rowsA,
			d_z,
			d_Q,
			d_x,
			CUSPARSE_INDEX_BASE_ZERO);
		cudaDeviceSynchronize();
	
		cudaMemcpyAsync(h_x, d_x, sizeof(double) * rowsA, cudaMemcpyDeviceToHost, stream);
		Eigen::VectorXd result(S.rows());
		result = Eigen::Map<Eigen::VectorXd>(h_x, S.rows());
		std::cout << result.head(15).transpose() << std::endl;
	
		printf("step 8: evaluate residual r = b - A*x (result on GPU)\n");
		cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA, cudaMemcpyDeviceToDevice, stream);
	
		cusparseSpMatDescr_t matA = NULL;
		cusparseCreateCsr(
			&matA, rowsA, colsA, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
		cusparseDnVecDescr_t vecx = NULL;
		cusparseDnVecDescr_t vecAx = NULL;
		cusparseCreateDnVec(&vecx, colsA, d_x, CUDA_R_64F);
		cusparseCreateDnVec(&vecAx, rowsA, d_r, CUDA_R_64F);
	
		/* Allocate workspace for cuSPARSE */
		size_t bufferSize = 0;
		cusparseSpMV_bufferSize(
			cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
			&one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
		void* buffer = NULL;
		cudaMalloc(&buffer, bufferSize);
	
		cusparseSpMV(
			cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
			&one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &buffer);
	
		cudaMemcpyAsync(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost, stream);
		cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost, stream);
		/* wait until h_x and h_r are ready */
		cudaDeviceSynchronize();
	
		
		if (handle)				{ cusolverSpDestroy(handle); }
		if (cusparseHandle)		{ cusparseDestroy(cusparseHandle); }
		if (stream)				{ cudaStreamDestroy(stream); }
		if (descrA)				{ cusparseDestroyMatDescr(descrA); }
		if (matA)				{ cusparseDestroySpMat(matA); }
		if (vecx)				{ cusparseDestroyDnVec(vecx); }
		if (vecAx)				{ cusparseDestroyDnVec(vecAx); }
	
		if (h_csrValA)			{ free(h_csrValA); }
		if (h_csrRowPtrA)		{ free(h_csrRowPtrA); }
		if (h_csrColIndA)		{ free(h_csrColIndA); }
		if (h_z)				{ free(h_z); }
		if (h_x)				{ free(h_x); }
		if (h_b)				{ free(h_b); }
		if (h_Qb)				{ free(h_Qb); }
		if (h_r)				{ free(h_r); }
	
		if (h_Q)				{ free(h_Q); }
	
		if (h_csrRowPtrB)		{ free(h_csrRowPtrB); }
		if (h_csrColIndB)		{ free(h_csrColIndB); }
		if (h_csrValB)			{ free(h_csrValB); }
		if (h_mapBfromA)		{ free(h_mapBfromA); }
	
		if (buffer_cpu)			{ free(buffer_cpu); }
	
		if (d_csrValA)			{ cudaFree(d_csrValA); }
		if (d_csrRowPtrA)		{ cudaFree(d_csrRowPtrA); }
		if (d_csrColIndA)		{ cudaFree(d_csrColIndA); }
		if (d_csrValB)			{ cudaFree(d_csrValB); }
		if (d_csrRowPtrB)		{ cudaFree(d_csrRowPtrB); }
		if (d_csrColIndB)		{ cudaFree(d_csrColIndB); }
		if (d_Q)				{ cudaFree(d_Q); }
		if (d_z)				{ cudaFree(d_z); }
		if (d_x)				{ cudaFree(d_x); }
		if (d_b)				{ cudaFree(d_b); }
		if (d_Qb)				{ cudaFree(d_Qb);}
		if (d_r)				{ cudaFree(d_r); }
	
	}


	double vec_norminf(int n, const double* x)
	{
		double norminf = 0;
		for (int j = 0; j < n; j++) {
			double x_abs = fabs(x[j]);
			norminf = (norminf > x_abs) ? norminf : x_abs;
		}
		return norminf;
	}

	void MLEigenSolver::cuLinearSolverTest(Eigen::SparseMatrix<double> SM, Eigen::MatrixXd& B, Eigen::MatrixXd& X)
	{
		////Matrix constructrions
		//int matSize = 100;
		//Eigen::SparseMatrix<double> MM(matSize, matSize);
		//Eigen::VectorXd b(matSize), x(matSize);
		//Eigen::MatrixXd X(matSize,1);
		//b.setConstant(50.0);
		//std::cout <<"b=\n " << b.head(10).transpose() << std::endl;
		//
		//std::vector<Eigen::Triplet<double>> MTriplet;
		//MTriplet.reserve(3 * matSize);
		//for (int i = 0; i < matSize; ++i) {
		//	MTriplet.push_back(Eigen::Triplet<double>(i, i, 10.0));
		//}
		//MM.setFromTriplets(MTriplet.begin(), MTriplet.end());

		// ======================== GPU : BEGIN ======================
		cusolverSpHandle_t handle = NULL;
		cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
		cudaStream_t stream = NULL;
		cusparseMatDescr_t descrA = NULL;

		int rowsA = SM.rows(); /* number of rows of A */
		int colsA = SM.cols(); /* number of columns of A */
		int nnzA = SM.nonZeros(); /* number of nonzeros of A */
		int baseA = 0; /* base index in CSR format */

		/* CSR(A) from I/O */
		int* h_csrRowPtrA = NULL;
		int* h_csrColIndA = NULL;
		double* h_csrValA = NULL;		
		h_csrRowPtrA	= (int*)malloc((rowsA+1)* sizeof(int)); // host memory for a
		h_csrColIndA	= (int*)malloc(nnzA * sizeof(int));
		h_csrValA		= (double*)malloc(nnzA*sizeof(double));	
		h_csrValA		= SM.valuePtr();
		h_csrRowPtrA	= SM.outerIndexPtr();
		h_csrColIndA	= SM.innerIndexPtr();


		double* h_z = NULL; /* z = B \ (Q*b) */
		double* h_x = NULL; /* x = A \ b */
		double* h_b = NULL; /* b = ones(n,1) */
		double* h_Qb = NULL; /* Q*b */
		double* h_r = NULL; /* r = b - A*x */
		//h_b = (double*)malloc(rowsA*sizeof(double));
		//h_b = B.col(0).data();

		int* h_Q = NULL; /* <int> n */
						 /* reorder to reduce zero fill-in */
						 /* Q = symrcm(A) or Q = symamd(A) */
		/* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
		int* h_csrRowPtrB = NULL; /* <int> n+1 */
		int* h_csrColIndB = NULL; /* <int> nnzA */
		double* h_csrValB = NULL; /* <double> nnzA */
		int* h_mapBfromA = NULL;  /* <int> nnzA */

		size_t size_perm = 0;
		void* buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */

	/* device copy of A: used in residual evaluation */
		int* d_csrRowPtrA = NULL;
		int* d_csrColIndA = NULL;
		double* d_csrValA = NULL;

		/* device copy of B: used in B*z = Q*b */
		int* d_csrRowPtrB = NULL;
		int* d_csrColIndB = NULL;
		double* d_csrValB = NULL;

		int* d_Q = NULL; /* device copy of h_Q */
		double* d_z = NULL; /* z = B \ Q*b */
		double* d_x = NULL; /* x = A \ b */
		double* d_b = NULL; /* a copy of h_b */
		double* d_Qb = NULL; /* a copy of h_Qb */
		double* d_r = NULL; /* r = b - A*x */

		double tol = 1.e-12;
		const int reorder = 0; /* no reordering */
		int singularity = 0; /* -1 if A is invertible under tol. */

		/* the constants are used in residual evaluation, r = b - A*x */
		const double minus_one = -1.0;
		const double one = 1.0;

		double b_inf = 0.0;
		double x_inf = 0.0;
		double r_inf = 0.0;
		double A_inf = 0.0;
		int errors = 0;
		int issym = 0;

		double start, stop;
		double time_solve_cpu;
		double time_solve_gpu;

		cusolverSpCreate(&handle);
		cusparseCreate(&cusparseHandle);

		cudaStreamCreate(&stream);
		/* bind stream to cusparse and cusolver*/
		cusolverSpSetStream(handle, stream);
		cusparseSetStream(cusparseHandle, stream);

		/* configure matrix descriptor*/
		cusparseCreateMatDescr(&descrA);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		
		h_z = (double*)malloc(sizeof(double) * colsA);
		h_x = (double*)malloc(sizeof(double) * colsA);
		h_b = (double*)malloc(sizeof(double) * rowsA);
		h_Qb = (double*)malloc(sizeof(double) * rowsA);
		h_r = (double*)malloc(sizeof(double) * rowsA);
		h_b = B.col(0).data();

		h_Q = (int*)malloc(sizeof(int) * colsA);
		h_csrRowPtrB = (int*)malloc(sizeof(int) * (rowsA + 1));
		h_csrColIndB = (int*)malloc(sizeof(int) * nnzA);
		h_csrValB = (double*)malloc(sizeof(double) * nnzA);
		h_mapBfromA = (int*)malloc(sizeof(int) * nnzA);

		assert(NULL != h_z);
		assert(NULL != h_x);
		assert(NULL != h_b);
		assert(NULL != h_Qb);
		assert(NULL != h_r);
		assert(NULL != h_Q);
		assert(NULL != h_csrRowPtrB);
		assert(NULL != h_csrColIndB);
		assert(NULL != h_csrValB);
		assert(NULL != h_mapBfromA);

		cudaMalloc((void**)&d_csrRowPtrA, sizeof(int) * (rowsA + 1));
		cudaMalloc((void**)&d_csrColIndA, sizeof(int) * nnzA);
		cudaMalloc((void**)&d_csrValA, sizeof(double) * nnzA);
		cudaMalloc((void**)&d_csrRowPtrB, sizeof(int) * (rowsA + 1));
		cudaMalloc((void**)&d_csrColIndB, sizeof(int) * nnzA);
		cudaMalloc((void**)&d_csrValB, sizeof(double) * nnzA);
		cudaMalloc((void**)&d_Q, sizeof(int) * colsA);
		cudaMalloc((void**)&d_z, sizeof(double) * colsA);
		cudaMalloc((void**)&d_x, sizeof(double) * colsA);
		cudaMalloc((void**)&d_b, sizeof(double) * rowsA);
		cudaMalloc((void**)&d_Qb, sizeof(double) * rowsA);
		cudaMalloc((void**)&d_r, sizeof(double) * rowsA);

		/* verify if A has symmetric pattern or not */
		cusolverSpXcsrissymHost(
			handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrRowPtrA + 1, h_csrColIndA, &issym);
		
		
		if (!issym)	{
			printf("Error: A has no symmetric pattern, please use LU or QR \n");
		}

		printf("step 2: reorder the matrix A to minimize zero fill-in\n");
		printf("        if the user choose a reordering by -P=symrcm, -P=symamd or -P=metis\n");
		//cusolverSpXcsrsymamdHost(handle, rowsA, nnzA,descrA, h_csrRowPtrA, h_csrColIndA,h_Q);
		//cusolverSpXcsrsymrcmHost(handle, rowsA, nnzA,descrA, h_csrRowPtrA, h_csrColIndA,h_Q);
		cusolverSpXcsrmetisndHost(handle, rowsA, nnzA,descrA, h_csrRowPtrA, h_csrColIndA,NULL, /* default setting. */			h_Q);

		printf("step 2.2: B = A(Q,Q) \n");

		memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
		memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

		cusolverSpXcsrperm_bufferSizeHost(
			handle, rowsA, colsA, nnzA,
			descrA, h_csrRowPtrB, h_csrColIndB,
			h_Q, h_Q,
			&size_perm);

		if (buffer_cpu)
		{
			free(buffer_cpu);
		}
		buffer_cpu = (void*)malloc(sizeof(char) * size_perm);
		assert(NULL != buffer_cpu);

		/* h_mapBfromA = Identity */
		for (int j = 0; j < nnzA; j++)
		{
			h_mapBfromA[j] = j;
		}
		cusolverSpXcsrpermHost(
			handle, rowsA, colsA, nnzA,
			descrA, h_csrRowPtrB, h_csrColIndB,
			h_Q, h_Q,
			h_mapBfromA,
			buffer_cpu);

		/* B = A( mapBfromA ) */
		for (int j = 0; j < nnzA; j++)
		{
			h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
		}

		//printf("step 3: b(j) = 1 + j/n \n");
		//for (int row = 0; row < rowsA; row++)
		//{
		//	h_b[row] = 1.0 + ((double)row) / ((double)rowsA);
		//}

		/* h_Qb = b(Q) */
		for (int row = 0; row < rowsA; row++) {
			h_Qb[row] = h_b[h_Q[row]];
		}

		printf("step 4: prepare data on device\n");
		cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_csrColIndB, h_csrColIndB, sizeof(int) * nnzA, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double) * nnzA, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_Qb, h_Qb, sizeof(double) * rowsA, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_Q, h_Q, sizeof(int) * rowsA, cudaMemcpyHostToDevice, stream);

		cusparseSpMatDescr_t matA = NULL;
		cusparseCreateCsr(
			&matA, rowsA, colsA, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

		cusparseDnVecDescr_t vecx = NULL;
		cusparseCreateDnVec(&vecx, colsA, d_x, CUDA_R_64F);
		cusparseDnVecDescr_t vecAx = NULL;
		cusparseCreateDnVec(&vecAx, rowsA, d_r, CUDA_R_64F);

		/* Allocate workspace for cuSPARSE */
		size_t bufferSize = 0;
		cusparseSpMV_bufferSize(
			cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
			&one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
		void* buffer = NULL;
		cudaMalloc(&buffer, bufferSize);

		cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
			        &one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &buffer);			
		cudaMemcpyAsync(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost, stream);
		/* wait until h_r is ready */
		cudaDeviceSynchronize();
			

		printf("step 7: solve A*x = b on GPU\n");
		
		cusolverSpDcsrlsvchol(
			handle, rowsA, nnzA,
			descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
			d_Qb, tol, reorder, d_z, &singularity);

		cudaDeviceSynchronize();

		cusparseDsctr(cusparseHandle, rowsA, d_z, d_Q, d_x, CUSPARSE_INDEX_BASE_ZERO);
		cudaDeviceSynchronize();

		///printf("step 8: evaluate residual r = b - A*x (result on GPU)\n");
		///cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA, cudaMemcpyDeviceToDevice, stream);
		///
		///cusparseSpMV(
		///	cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
		///	&one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &buffer);

		cudaMemcpyAsync(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost, stream);
		///cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost, stream);
		/* wait until h_x and h_r are ready */
		///cudaDeviceSynchronize();

		///b_inf = vec_norminf(rowsA, h_b);
		///x_inf = vec_norminf(colsA, h_x);
		///r_inf = vec_norminf(rowsA, h_r);

		Eigen::VectorXd X1;
		X1.resize(SM.rows());
		X1 = Eigen::Map<Eigen::VectorXd>(h_x, SM.rows());
		X.resize(SM.rows(), 1);
		X.col(0) = X1;

		// only cholesky and qr have GPU version
		///printf("(GPU) |b - A*x| = %E \n", r_inf);
		///printf("(GPU) |A| = %E \n", A_inf);
		///printf("(GPU) |x| = %E \n", x_inf);
		///printf("(GPU) |b| = %E \n", b_inf);
		///printf("(GPU) |b - A*x|/(|A|*|x| + |b|) = %E \n", r_inf / (A_inf * x_inf + b_inf));


		///cudaDeviceSynchronize();
		// ======================== GPU : END ======================

		//std::cout << "MM" << MM.topLeftCorner(10, 10) << std::endl;
		//std::cout << "b" << b.head(10).transpose() << std::endl;
		//std::cout << "x" << x << std::endl;

		std::cout << "Cleaning up \n";
	
		if (handle) {			cusolverSpDestroy(handle); }
		if (cusparseHandle) {	cusparseDestroy(cusparseHandle); }
		if (stream) {			cudaStreamDestroy(stream); }
		if (descrA) {			cusparseDestroyMatDescr(descrA); }
		if (matA       ) {		cusparseDestroySpMat(matA); }
		if (vecx       ) {		cusparseDestroyDnVec(vecx); }
		if (vecAx      ) {		cusparseDestroyDnVec(vecAx); }

		//if (h_csrValA   ) { free(h_csrValA); }
		//if (h_csrRowPtrA) { free(h_csrRowPtrA); }
		//if (h_csrColIndA) { free(h_csrColIndA); }
		//if (h_z) { free(h_z); }
		//if (h_x) { free(h_x); }
		//if (h_b) { free(h_b); }
		//if (h_Qb) { free(h_Qb); }
		//if (h_r) { free(h_r); }
		//
		//if (h_Q) { free(h_Q); }
		//
		//if (h_csrRowPtrB) { free(h_csrRowPtrB); }
		//if (h_csrColIndB) { free(h_csrColIndB); }
		//if (h_csrValB   ) { free(h_csrValB   ); }
		//if (h_mapBfromA ) { free(h_mapBfromA ); }
		//
		//if (buffer_cpu) { free(buffer_cpu); }

		if (d_csrValA   ) { cudaFree(d_csrValA); }
		if (d_csrRowPtrA) { cudaFree(d_csrRowPtrA); }
		if (d_csrColIndA) { cudaFree(d_csrColIndA); }
		if (d_csrValB   ) { cudaFree(d_csrValB); }
		if (d_csrRowPtrB) { cudaFree(d_csrRowPtrB); }
		if (d_csrColIndB) { cudaFree(d_csrColIndB); }
		if (d_Q) { cudaFree(d_Q); }
		if (d_z) { cudaFree(d_z); }
		if (d_x) { cudaFree(d_x); }
		if (d_b) { cudaFree(d_b); }
		if (d_Qb){ cudaFree(d_Qb); }
		if (d_r) { cudaFree(d_r); }

		std::cout << "Return to main function \n";
	}

	void MLEigenSolver::cusparseLinearSolverLLTest(Eigen::SparseMatrix<double> SM, Eigen::MatrixXd& B, Eigen::MatrixXd& X)
	{
		chrono::high_resolution_clock::time_point	t0, t1, t2;
		chrono::duration<double>					duration;


		std::cout << "[1] Set-up GPU system \n";
		t0 = chrono::high_resolution_clock::now();
		t1 = t0;

		////Matrix constructrions
		//int matSize = 100;
		//Eigen::SparseMatrix<double> SM(matSize, matSize);
		//Eigen::VectorXd b(matSize), x(matSize);
		////Eigen::MatrixXd X(matSize,1);
		//b.setConstant(50.0);
		//std::cout <<"b=\n " << b.head(10).transpose() << std::endl;
		//
		//std::vector<Eigen::Triplet<double>> MTriplet;
		//MTriplet.reserve(3 * matSize);
		//for (int i = 0; i < matSize; ++i) {
		//	MTriplet.push_back(Eigen::Triplet<double>(i, i, 10.0));
		//}
		//SM.setFromTriplets(MTriplet.begin(), MTriplet.end());

		// GPU implementation
		cusolverSpHandle_t cusolverSpH = NULL; // reordering, permutation and 1st LU factorization
		cusparseHandle_t   cusparseH = NULL;   // residual evaluation
		cudaStream_t stream = NULL;
		cusparseMatDescr_t descrA = NULL; // A is a base-0 general matrix
		csrcholInfo_t d_info = NULL; // opaque info structure for LU with parital pivoting

		int rowsA = SM.rows(); // number of rows of A
		int colsA = SM.cols(); // number of columns of A
		int nnzA = SM.nonZeros(); // number of nonzeros of A
		int baseA = 0; // base index in CSR format

		int* h_csrRowPtrA = NULL; // <int> n+1 
		int* h_csrColIndA = NULL; // <int> nnzA 
		double* h_csrValA = NULL; // <double> nnzA 		
		h_csrRowPtrA = (int*)malloc((rowsA + 1) * sizeof(int)); // host memory for a
		h_csrColIndA = (int*)malloc(nnzA * sizeof(int));
		h_csrValA = (double*)malloc(nnzA * sizeof(double));
		h_csrValA = SM.valuePtr();
		h_csrRowPtrA = SM.outerIndexPtr();
		h_csrColIndA = SM.innerIndexPtr();

		double* h_x = NULL; // <double> n,  x = A \ b
		double* h_b = NULL; // <double> n, b = ones(m,1)
		double* h_r = NULL; // <double> n, r = b - A*x

		size_t size_internal = 0;
		size_t size_chol = 0; // size of working space for csrlu
		void* buffer_cpu = NULL; // working space for Cholesky
		void* buffer_gpu = NULL; // working space for Cholesky

		int* d_csrRowPtrA = NULL; // <int> n+1
		int* d_csrColIndA = NULL; // <int> nnzA
		double* d_csrValA = NULL; // <double> nnzA
		double* d_x = NULL; // <double> n, x = A \ b 
		double* d_b = NULL; // <double> n, a copy of h_b
		double* d_r = NULL; // <double> n, r = b - A*x

		// the constants used in residual evaluation, r = b - A*x
		const double minus_one = -1.0;
		const double one = 1.0;
		// the constant used in cusolverSp
		// singularity is -1 if A is invertible under tol
		// tol determines the condition of singularity
		int singularity = 0;
		const double tol = 1.e-14;

		double x_inf = 0.0; // |x|
		double r_inf = 0.0; // |r|
		double A_inf = 0.0; // |A|
		int errors = 0;

		cusolverSpCreate(&cusolverSpH);
		cusparseCreate(&cusparseH);
		cudaStreamCreate(&stream);
		cusolverSpSetStream(cusolverSpH, stream);
		cusparseSetStream(cusparseH, stream);
		cusparseCreateMatDescr(&descrA);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;


		std::cout << "[2] Allocating memory \n";
		t1 = chrono::high_resolution_clock::now();

		h_x = (double*)malloc(sizeof(double) * colsA);
		h_b = (double*)malloc(sizeof(double) * rowsA);
		h_r = (double*)malloc(sizeof(double) * rowsA);
		h_b = B.col(0).data();

		assert(NULL != h_x);
		assert(NULL != h_b);
		assert(NULL != h_r);

		cudaMalloc((void**)&d_csrRowPtrA, sizeof(int) * (rowsA + 1));
		cudaMalloc((void**)&d_csrColIndA, sizeof(int) * nnzA);
		cudaMalloc((void**)&d_csrValA, sizeof(double) * nnzA);
		cudaMalloc((void**)&d_x, sizeof(double) * colsA);
		cudaMalloc((void**)&d_b, sizeof(double) * rowsA);
		cudaMalloc((void**)&d_r, sizeof(double) * rowsA);

		cusparseSpMatDescr_t matA = NULL;
		cusparseCreateCsr(
			&matA, rowsA, colsA, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

		cusparseDnVecDescr_t vecx = NULL;
		cusparseCreateDnVec(&vecx, colsA, d_x, CUDA_R_64F);
		cusparseDnVecDescr_t vecAx = NULL;
		cusparseCreateDnVec(&vecAx, rowsA, d_r, CUDA_R_64F);

		/* Allocate workspace for cuSPARSE */
		size_t bufferSize = 0;
		cusparseSpMV_bufferSize(
			cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
			&one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
		void* buffer = NULL;
		cudaMalloc(&buffer, bufferSize);

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;

		std::cout << "[3] Copying data to GPU \n";
		t1 = chrono::high_resolution_clock::now();

		printf("step 9: create opaque info structure\n");
		cusolverSpCreateCsrcholInfo(&d_info);

		cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice);
		cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
		cudaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice);

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;

		std::cout << "[2] Analysing the structure of matrix \n";
		t1 = chrono::high_resolution_clock::now();

		printf("step 10: analyze chol(A) to know structure of L\n");
		cusolverSpXcsrcholAnalysis(
			cusolverSpH, rowsA, nnzA,
			descrA, d_csrRowPtrA, d_csrColIndA,
			d_info);


		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;

		std::cout << "[2] Allocating workspace for program \n";
		t1 = chrono::high_resolution_clock::now();

		printf("step 11: workspace for chol(A)\n");
		cusolverSpDcsrcholBufferInfo(
			cusolverSpH, rowsA, nnzA,
			descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
			d_info,
			&size_internal,
			&size_chol);

		if (buffer_gpu) {
			cudaFree(buffer_gpu);
		}
		cudaMalloc(&buffer_gpu, sizeof(char) * size_chol);

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;

		
		t1 = chrono::high_resolution_clock::now();
		printf("step 12: compute A = L*L^T \n");
		cusolverSpDcsrcholFactor(
			cusolverSpH, rowsA, nnzA,
			descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
			d_info,
			buffer_gpu);

		printf("step 13: check if the matrix is singular \n");
		cusolverSpDcsrcholZeroPivot(
			cusolverSpH, d_info, tol, &singularity);

		if (0 <= singularity) {
			fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
			//return;
		}

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;


		t1 = chrono::high_resolution_clock::now();
		printf("step 14: solve A*x = b \n");
		cusolverSpDcsrcholSolve(
			cusolverSpH, rowsA, d_b, d_x, d_info, buffer_gpu);

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;

		std::cout << "[2] Copy to CPU memory \n";
		t1 = chrono::high_resolution_clock::now();
		cudaMemcpyAsync(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost, stream);
		//cudaMemcpy(d_r, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice);
		//
		//cusparseSpMV(
		//	cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
		//	&one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &buffer);
		//
		//cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost);

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;

		//r_inf = vec_norminf(rowsA, h_r);

		//printf("(GPU) |b - A*x| = %E \n", r_inf);
		//printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

		std::cout << "[2] Map to Eigen \n";
		t1 = chrono::high_resolution_clock::now();
		Eigen::VectorXd X1;
		X1.resize(SM.rows());
		X1 = Eigen::Map<Eigen::VectorXd>(h_x, SM.rows());
		X.resize(SM.rows(), 1);
		X.col(0) = X1;
		std::cout << "X1 \n" << X1.head(10).transpose() << std::endl;

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;

		std::cout << "[2] Cleaning up \n";
		t1 = chrono::high_resolution_clock::now();
		// Cleaning up
		if (cusolverSpH) { cusolverSpDestroy(cusolverSpH); }
		if (cusparseH  ) { cusparseDestroy(cusparseH); }
		if (stream     ) { cudaStreamDestroy(stream); }
		if (descrA     ) { cusparseDestroyMatDescr(descrA); }
		if (d_info     ) { cusolverSpDestroyCsrcholInfo(d_info); }
		if (matA       ) { cusparseDestroySpMat(matA); }
		if (vecx       ) { cusparseDestroyDnVec(vecx); }
		if (vecAx      ) { cusparseDestroyDnVec(vecAx); }

		//if (h_csrValA   ) { free(h_csrValA); }
		//if (h_csrRowPtrA) { free(h_csrRowPtrA); }
		//if (h_csrColIndA) { free(h_csrColIndA); }
		//
		//if (h_x   ) { free(h_x); }
		//if (h_b   ) { free(h_b); }
		//if (h_r   ) { free(h_r); }

		if (buffer_cpu) { free(buffer_cpu); }
		if (buffer_gpu) { cudaFree(buffer_gpu); }

		if (d_csrValA   ) { cudaFree(d_csrValA); }
		if (d_csrRowPtrA) { cudaFree(d_csrRowPtrA); }
		if (d_csrColIndA) { cudaFree(d_csrColIndA); }
		if (d_x) { cudaFree(d_x); }
		if (d_b) { cudaFree(d_b); }
		if (d_r) { cudaFree(d_r); }

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "    in " << duration.count() << " seconds." << std::endl;
	}

	void MLEigenSolver::cuSparseProjection(Eigen::SparseMatrix<double>& Basis, Eigen::SparseMatrix<double>& Mat, Eigen::SparseMatrix<double>& MatRed)
	{
		cusparseSpGEMMDescr_t		spGEMMDesc = NULL;
		cusparseStatus_t			cuSpStat = cusparseStatus_t::CUSPARSE_STATUS_SUCCESS;
		cusparseOperation_t			NoTrans = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;
		cudaError					cudaStat = cudaError::cudaSuccess;
		cusparseHandle_t			cuSpHandle;
		
		std::cout << "Getting the basis matrix \n";
		Eigen::SparseMatrix<double> U, UT;
		U = Basis;
		UT = Basis.transpose();
		U.makeCompressed();
		UT.makeCompressed();
		Mat.makeCompressed();

		std::cout << "Setup matrix dimensions/size \n";
		/* Variable on size */
		int m_U = U.rows();
		int n_U = U.cols();
		int nnz_U = U.nonZeros();
		int m_UT = n_U;
		int n_UT = m_U;
		int nnz_UT = nnz_U;
		int m_Mat = Mat.rows();
		int n_Mat = Mat.cols();
		int nnz_Mat = Mat.nonZeros();
		int m_MU = Mat.rows();
		int n_MU = U.cols();
		int nnz_MU = 40 * m_MU;
		int m_UTMU = n_MU;
		int n_UTMU = n_MU;
		int nnz_UTMU = 70 * n_MU;

		double alpha = 1.0;
		double beta = 0.0;

		std::cout << "Allocating memory in GPU \n";
		cusparseSpMatDescr_t matUT;
		cusparseSpMatDescr_t matU;
		cusparseSpMatDescr_t matM;
		cusparseSpMatDescr_t matMU;
		cusparseSpMatDescr_t matUTMU;
		cusparseCreate(&cuSpHandle);

		double* d_csrVal_U; cudaMalloc(&d_csrVal_U, nnz_U * sizeof(double));
		int* d_csrRowPtr_U; cudaMalloc(&d_csrRowPtr_U, (m_U + 1) * sizeof(int));
		int* d_csrColInd_U; cudaMalloc(&d_csrColInd_U, nnz_U * sizeof(int));

		double* d_csrVal_UT; cudaMalloc(&d_csrVal_UT, nnz_UT * sizeof(double));
		int* d_csrRowPtr_UT; cudaMalloc(&d_csrRowPtr_UT, (m_UT + 1) * sizeof(int));
		int* d_csrColInd_UT; cudaMalloc(&d_csrColInd_UT, nnz_UT * sizeof(int));

		double* d_csrVal_Mat; cudaMalloc(&d_csrVal_Mat, nnz_Mat * sizeof(double));
		int* d_csrRowPtr_Mat; cudaMalloc(&d_csrRowPtr_Mat, (m_Mat + 1) * sizeof(int));
		int* d_csrColInd_Mat; cudaMalloc(&d_csrColInd_Mat, nnz_Mat * sizeof(int));

		//double* d_csrVal_MU; cudaMalloc(&d_csrVal_MU,nnz_MU * sizeof(double));
		//int* d_csrRowPtr_MU; cudaMalloc(&d_csrRowPtr_MU, (m_MU + 1) * sizeof(int));
		//int* d_csrColInd_MU; cudaMalloc(&d_csrColInd_MU, nnz_MU * sizeof(int));

		double* d_csrVal_UTMU; cudaMalloc(&d_csrVal_UTMU, nnz_UTMU * sizeof(double));
		int* d_csrRowPtr_UTMU; cudaMalloc(&d_csrRowPtr_UTMU, (m_UTMU + 1) * sizeof(int));
		int* d_csrColInd_UTMU; cudaMalloc(&d_csrColInd_UTMU, nnz_UTMU * sizeof(int));

		std::cout << "Copying data to GPU \n";

		cudaStat = cudaMemcpy(d_csrVal_U, U.valuePtr(), nnz_U * sizeof(double), cudaMemcpyHostToDevice);				std::cout << cudaStat << "|";
		cudaStat = cudaMemcpy(d_csrRowPtr_U, U.outerIndexPtr(), (m_U + 1) * sizeof(int), cudaMemcpyHostToDevice);		std::cout << cudaStat << "|";
		cudaStat = cudaMemcpy(d_csrColInd_U, U.innerIndexPtr(), nnz_U * sizeof(int), cudaMemcpyHostToDevice);			std::cout << cudaStat << "\n";

		cudaStat = cudaMemcpy(d_csrVal_UT, UT.valuePtr(), nnz_UT * sizeof(double), cudaMemcpyHostToDevice);				std::cout << cudaStat << "|";
		cudaStat = cudaMemcpy(d_csrRowPtr_UT, UT.outerIndexPtr(), (m_UT + 1) * sizeof(int), cudaMemcpyHostToDevice);	std::cout << cudaStat << "|";
		cudaStat = cudaMemcpy(d_csrColInd_UT, UT.innerIndexPtr(), nnz_UT * sizeof(int), cudaMemcpyHostToDevice);		std::cout << cudaStat << "\n";

		cudaStat = cudaMemcpy(d_csrVal_Mat, Mat.valuePtr(), nnz_Mat * sizeof(double), cudaMemcpyHostToDevice);			std::cout << cudaStat << "|";
		cudaStat = cudaMemcpy(d_csrRowPtr_Mat, Mat.outerIndexPtr(), (m_Mat + 1) * sizeof(int), cudaMemcpyHostToDevice);	std::cout << cudaStat << "|";
		cudaStat = cudaMemcpy(d_csrColInd_Mat, Mat.innerIndexPtr(), nnz_Mat * sizeof(int), cudaMemcpyHostToDevice);		std::cout << cudaStat << "\n";

		std::cout << "Copying matrix MU where MU = M*U \n";
		Eigen::SparseMatrix<double> Id(Basis.rows(), Basis.cols());
		Id.setZero();
		Id.makeCompressed();
		int nnz_ID = Id.nonZeros();
		printf("ID=%dx%d (%d nnzs)\n", Id.rows(), Id.cols(), Id.nonZeros());
		std::cout << "Allocate values arrays: ";	double* d_csrVal_MU; cudaStat = cudaMalloc(&d_csrVal_MU, nnz_ID * sizeof(double));				std::cout << cudaStat;
		std::cout << "| row arrays: ";				int* d_csrRowPtr_MU; cudaStat = cudaMalloc(&d_csrRowPtr_MU, (Id.rows() + 1) * sizeof(int));		std::cout << cudaStat;
		std::cout << "| columns arrays: ";			int* d_csrColInd_MU; cudaStat = cudaMalloc(&d_csrColInd_MU, nnz_ID * sizeof(int));				std::cout << cudaStat << std::endl;

		cudaStat = cudaMemcpy(d_csrVal_MU, Id.valuePtr(), nnz_ID * sizeof(double), cudaMemcpyHostToDevice);					std::cout << "Copying values to GPU : " << cudaStat;
		cudaStat = cudaMemcpy(d_csrRowPtr_MU, Id.outerIndexPtr(), (Basis.rows() + 1) * sizeof(int), cudaMemcpyHostToDevice);		std::cout << "| row arrays: " << cudaStat;
		cudaStat = cudaMemcpy(d_csrColInd_MU, Id.innerIndexPtr(), nnz_ID * sizeof(int), cudaMemcpyHostToDevice);			std::cout << "| columns arrays: " << cudaStat << std::endl;

		std::cout << "Creating matrices in GPU \n";
		cuSpStat = cusparseCreateCsr(&matUT, m_UT, n_UT, nnz_UT, d_csrRowPtr_UT, d_csrColInd_UT, d_csrVal_UT, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);				std::cout << "Creating matrix on GPU (UT)"<< cuSpStat << "|";
		cuSpStat = cusparseCreateCsr(&matU, m_U, n_U, nnz_U, d_csrRowPtr_U, d_csrColInd_U, d_csrVal_U, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);						std::cout << "(U)" << cuSpStat << "|";
		cuSpStat = cusparseCreateCsr(&matM, m_Mat, n_Mat, nnz_Mat, d_csrRowPtr_Mat, d_csrColInd_Mat, d_csrVal_Mat, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);			std::cout << "(M)" << cuSpStat << "|";
		cuSpStat = cusparseCreateCsr(&matMU, m_MU, n_MU, nnz_MU, d_csrRowPtr_MU, d_csrColInd_MU, d_csrVal_MU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);				std::cout << "(MU)" << cuSpStat << "|";
		cuSpStat = cusparseCreateCsr(&matUTMU, m_UTMU, n_UTMU, nnz_UTMU, d_csrRowPtr_UTMU, d_csrColInd_UTMU, d_csrVal_UTMU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);	std::cout << "(UTMU)" << cuSpStat << "|\n";

		void* dBuffer1 = NULL, * dBuffer2 = NULL;
		size_t bufferSize1 = 0, bufferSize2 = 0;

		cuSpStat = cusparseSpGEMM_createDescr(&spGEMMDesc);

		std::cout << "Preparing matrix multiplication \n";
		cusparseSpGEMM_workEstimation(cuSpHandle, NoTrans, NoTrans, &alpha, matM, matU, &beta, matMU, CUDA_R_64F, cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT, spGEMMDesc, &bufferSize1, NULL);
		cudaMalloc((void**)&dBuffer1, bufferSize1);
		cusparseSpGEMM_workEstimation(cuSpHandle, NoTrans, NoTrans, &alpha, matM, matU, &beta, matMU, CUDA_R_64F, cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT, spGEMMDesc, &bufferSize1, dBuffer1);
		std::cout << "buffer size 1: " << bufferSize1 << std::endl;

		std::cout << "Multiplying \n";
		cusparseSpGEMM_compute(cuSpHandle, NoTrans, NoTrans, &alpha, matM, matU, &beta, matMU, CUDA_R_64F, cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT, spGEMMDesc, &bufferSize2, NULL);
		cudaMalloc((void**)&dBuffer2, bufferSize2);
		cusparseSpGEMM_compute(cuSpHandle, NoTrans, NoTrans, &alpha, matM, matU, &beta, matMU, CUDA_R_64F, cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT, spGEMMDesc, &bufferSize2, dBuffer2);
		std::cout << "buffer size 2: " << bufferSize2 << std::endl;

		int64_t d_m_MU, d_n_MU, d_nnz_MU;
		cusparseSpMatGetSize(matMU, &d_m_MU, &d_n_MU, &d_nnz_MU);
		cudaMalloc((void**)&d_csrColInd_MU, d_nnz_MU * sizeof(int));
		cudaMalloc((void**)&d_csrVal_MU, d_nnz_MU * sizeof(double));
		printf("MU:%dx%d <%d nonzeros> \n", d_m_MU, d_n_MU, d_nnz_MU);

		std::cout << "Copying data from GPU to CPU \n";
		cusparseCsrSetPointers(matMU, d_csrRowPtr_MU, d_csrColInd_MU, d_csrVal_MU);

		//cusparseSpGEMM_copy(cuSpHandle, NoTrans, NoTrans, &alpha, matM, matU, &beta, matMU, CUDA_R_64F, cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT, spGEMMDesc);
		cusparseSpGEMM_copy(cuSpHandle, NoTrans, NoTrans, &alpha, matM, matU, &beta, matMU, CUDA_R_64F, cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT, spGEMMDesc);
		
		//double* h_Q = (double*)malloc(m_Q * n_Q * sizeof(double));
		//double* h_A = (double*)malloc(n_Q * sizeof(double));
		//cudaStat = cudaMemcpy(h_Q, d_Q, sizeof(double) * m_Q * n_Q, cudaMemcpyDeviceToHost);
		//cudaStat = cudaMemcpy(h_A, d_A, sizeof(double) * m_MDnSym, cudaMemcpyDeviceToHost);
		//
		//EigVects.resize(m_Q, n_Q);
		//EigVals.resize(n_Q);
		//EigVects = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_Q, m_Q, n_Q, Eigen::OuterStride<>(m_Q));
		//EigVals = Eigen::Map<Eigen::VectorXd>(h_A, n_Q);

		std::cout << "Settign up Eigen sparse matrix \n";
		MatRed.resize(M.rows(), U.cols());
		cudaStat = cudaMemcpy(MatRed.valuePtr(), d_csrVal_MU, nnz_MU * sizeof(double), cudaMemcpyDeviceToHost);
		cudaStat = cudaMemcpy(MatRed.outerIndexPtr(), d_csrRowPtr_Mat, (m_MU + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaStat = cudaMemcpy(MatRed.innerIndexPtr(), d_csrColInd_Mat, nnz_MU * sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "CUDA: MU=\n" << MatRed.topLeftCorner(10, 10) << std::endl;

		/* CPU-based Implementation */
		Eigen::SparseMatrix<double> CPU_MU = Mat * Basis;
		std::cout << "CPU: MU=\n" << CPU_MU.topLeftCorner(10, 10) << std::endl;

		//cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_workEstimation(cusparseHandle_t      handle,
		//		cusparseOperation_t   opA,
		//		cusparseOperation_t   opB,
		//		const void* alpha,
		//		cusparseSpMatDescr_t  matA,
		//		cusparseSpMatDescr_t  matB,
		//		const void* beta,
		//		cusparseSpMatDescr_t  matC,
		//		cudaDataType          computeType,
		//		cusparseSpGEMMAlg_t   alg,
		//		cusparseSpGEMMDescr_t spgemmDescr,
		//		size_t * bufferSize1,
		//		void* externalBuffer1);
		//
		//cusparseStatus_t CUSPARSEAPI
		//	cusparseSpGEMM_compute(cusparseHandle_t      handle,
		//		cusparseOperation_t   opA,
		//		cusparseOperation_t   opB,
		//		const void* alpha,
		//		cusparseSpMatDescr_t  matA,
		//		cusparseSpMatDescr_t  matB,
		//		const void* beta,
		//		cusparseSpMatDescr_t  matC,
		//		cudaDataType          computeType,
		//		cusparseSpGEMMAlg_t   alg,
		//		cusparseSpGEMMDescr_t spgemmDescr,
		//		void* externalBuffer1,
		//		size_t * bufferSize2,
		//		void* externalBuffer2);
		//
		//cusparseStatus_t CUSPARSEAPI
		//	cusparseSpGEMM_copy(cusparseHandle_t      handle,
		//		cusparseOperation_t   opA,
		//		cusparseOperation_t   opB,
		//		const void* alpha,
		//		cusparseSpMatDescr_t  matA,
		//		cusparseSpMatDescr_t  matB,
		//		const void* beta,
		//		cusparseSpMatDescr_t  matC,
		//		cudaDataType          computeType,
		//		cusparseSpGEMMAlg_t   alg,
		//		cusparseSpGEMMDescr_t spgemmDescr,
		//		void* externalBuffer2);
		//
		//cuSpStat = cusparseSpGEMM_destroyDescr(spGEMMDesc);
	}
}