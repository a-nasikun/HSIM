#include "MLEigenSolver.h"
#include <Eigen/PardisoSupport>
#include <omp.h>
#include <random>

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
	void MLEigenSolver::setSamplingLevels(Eigen::VectorXd& sampSize, int numEigen, double levelRatio)
	{
		double	spRatio = 1.0;
		int		sizeSubsp = spRatio * 2 * numEigen;

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
	
	/**
	*  @brief	 Construct the number of samples in every level via farthest point sampling. 
	*  @details	
	*   Starting from the coarsest level, it computes the the first set of sampled DoFs (i.e. vertices).
	*   Then for every level, it adds more samples according the 
	*  @see MuLES::constructFarthestPointSample() \n MuLES::MLEigenSolver::setSamplingLevels() 
	*  @param	samples	A constact vector stating the number of samples on every level of the hierarchy.
	*/
	void MLEigenSolver::constructSampleHierarchy(const Eigen::VectorXd& samples)
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

	/* Subspace Iteration methods */
	void MLEigenSolver::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset) { }

	void MLEigenSolver::performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift) {}

	void MLEigenSolver::performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

	void MLEigenSolver::performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift) {}

	void MLEigenSolver::performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

	void MLEigenSolver::performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

	void MLEigenSolver::checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const
	{
		convRate.resize(A.rows());
		Eigen::VectorXd Kq, AMq, diff;

		for (int i = 0; i < A.rows(); i++) {
			Kq = SK * Q.col(i);
			AMq = A(i)*MK*Q.col(i);
			diff = Kq - AMq;

			double n1 = diff.transpose()*MK*diff;
			double n2 = Kq.transpose()*MK*Kq;
			convRate(i) = sqrt(n1 / n2);
		}
		//printf("eigVals: %d | S: %dx%d | M: %dx%d \n", A.rows(), SK.rows(), SK.cols(), MK.rows(), MK.cols());
	}

	void MLEigenSolver::checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, bool isWithShift, double mu, Eigen::VectorXd& convRate) const
	{
		convRate.resize(A.rows());

		for (int i = 0; i < A.rows(); ++i)
		{
			double a2 = A(i)*A(i);
			double qTq = Q.col(i).transpose()*Q.col(i);
			if (isWithShift) qTq += mu * (2 * A(i) - mu);
			double rat = a2 / qTq;
			double conv = abs(1.0 - rat);
			convRate(i) = sqrt(conv);
		}
	}

	/* Computing the reference eigenproblem */
	void MLEigenSolver::computeRefEigenProblem(int numEigs, const EigenSolverType& solverType, int maxIters, double tol)
	{
		switch (solverType)
		{
		case EigenSolverType::EIG_SOLVER_MATLAB:
			computeEigenMatlab(S, M, numEigs, EigVectsRef, EigValsRef);
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
			computeEigenMatlab(SLoc, MLoc, numEigs, EigVectsLoc, EigValsRef);
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
	void MLEigenSolver::visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k) const
	{
		Eigen::MatrixXd VColor;
		igl::jet(geoDist[k], true, VColor);
		viewer.data().set_colors(VColor);
	}

	void MLEigenSolver::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k, double scale) const {}
	void MLEigenSolver::visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale) const {}
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

	/* Some getters */
	Eigen::MatrixXd MLEigenSolver::getV() const { return V; }
	Eigen::MatrixXi MLEigenSolver::getF() const { return F; }
	void MLEigenSolver::setV(const Eigen::MatrixXd& Vin) { V = Vin; }
	void MLEigenSolver::setF(const Eigen::MatrixXi& Fin) { F = Fin; }
	Eigen::SparseMatrix<double> MLEigenSolver::getStiffnessMatrix() const { return S; }
	Eigen::SparseMatrix<double> MLEigenSolver::getMassMatrix() const { return M; }
	Eigen::SparseMatrix<double> MLEigenSolver::getBasisMatrix() const { return Basis; }
	bool MLEigenSolver::getClampBasis() const { return isClampedBasis; }
	void MLEigenSolver::setClampBasis(bool isClamped) { isClampedBasis = isClamped; }
	int MLEigenSolver::getLevel() const { return level; }
	int MLEigenSolver::getSampleSize() const { return sampleSize; }
	const Eigen::VectorXi& MLEigenSolver::getSample() const { return Sample; }
	const vector<int>& MLEigenSolver::getSubspaceSize() const { return SubspaceSize; }
	void MLEigenSolver::setZhaoShiftParam(const double shiftParam) { zhaoShiftParam = shiftParam; }
	double MLEigenSolver::getZhaoShiftParam() const { return zhaoShiftParam; }

	/* Getters of the eigen-values + vectors*/
	const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsRef() const { return EigVectsRef; }
	const Eigen::VectorXd& MLEigenSolver::getEigenVectorsRef(int id) const
	{
		if (id >= EigVectsRef.cols())	return EigVectsRef.col(EigValsRef.rows() - 1);
		else if (id < 0)						return EigVectsRef.col(0);

		return EigVectsRef.col(id);
	}
	const Eigen::VectorXd& MLEigenSolver::getEigenValuesRef() const { return EigValsRef; }
	const double MLEigenSolver::getEigenValueRef(int id) const
	{
		if (id >= EigValsRef.rows())	return EigValsRef(EigValsRef.rows() - 1);
		else if (id < 0)						return EigValsRef(0);

		return EigValsRef(id);
	}
	const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsReduced() const { return EigVectsReduced; }
	const Eigen::VectorXd& MLEigenSolver::getEigenVectorsReduced(int id) const
	{
		if (id >= EigVectsReduced.cols())	return EigVectsReduced.col(EigValsReduced.rows() - 1);
		else if (id < 0)						return EigVectsReduced.col(0);

		return EigVectsReduced.col(id);
	}
	const Eigen::VectorXd& MLEigenSolver::getEigenValuesReduced() const { return EigValsReduced; }
	const double MLEigenSolver::getEigenValueReduced(int id) const
	{
		if (id >= EigValsReduced.rows())	return EigValsReduced(EigValsReduced.rows() - 1);
		else if (id < 0)						return EigValsReduced(0);

		return EigValsReduced(id);
	}
	const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsSIM() const { return EigVectsSIM; }
	const Eigen::VectorXd& MLEigenSolver::getEigenVectorsSIM(int id) const
	{
		if (id >= EigVectsSIM.cols())		return EigVectsSIM.col(EigValsSIM.rows() - 1);
		else if (id < 0)						return EigVectsSIM.col(0);
		return EigVectsSIM.col(id);
	}
	const Eigen::VectorXd& MLEigenSolver::getEigenValuesSIM() const { return EigValsSIM; }
	const double MLEigenSolver::getEigenValueSIM(int id) const
	{
		if (id >= EigValsSIM.rows())		return EigValsSIM(EigValsSIM.rows() - 1);
		else if (id < 0)						return EigValsSIM(0);

		return EigValsSIM(id);
	}
	//template void MLEigenSolver::setSamplingLevels(Eigen::PlainObjectBase<double>& sampSize, int numEigen, double levelRatio);
}