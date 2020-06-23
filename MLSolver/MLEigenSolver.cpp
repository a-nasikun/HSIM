#include "MLEigenSolver.h"
#include <Eigen/PardisoSupport>
#include <omp.h>

/**
*  @brief Reading a (triangle) mesh specified by the input.
*
*  @details
*   Reading a (triangle) mesh specified by the input 'meshFile', 
*	currently it supports 'obj' and 'off' file format only
*  @param meshFile  Location of the mesh
*/
void MLEigenSolver::readMesh(const string &meshFile)
{
	V.resize(0, 0);
	F.resize(0, 0);

	if (meshFile.substr(meshFile.find_last_of(".") + 1) == "off") {
		igl::readOFF(meshFile, V, F);
		if (V.rows() == 0 || V.cols() == 0) {
			cout << "Error! It cannot be an empty mesh." << endl;
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
*  @brief Setting up the system to have required levels.
*
*  @details
*   Setting up the system to have required levels.
*  @param n Number of specified levels of the system.
*/
void MLEigenSolver::setInitialSystem(int n)
{
	level			= n;
	maxNeighDist.resize(level);
	U.resize(level);
	Sbar.resize(level);
	Mbar.resize(level);
	SubspaceSize.resize(level);
	geoDist.resize(level);
	clusters.resize(level);
}

/**
*  @brief Setting up the adjacency matrix.
*
*  @details
*   Setting up the adjacency matrix.
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
			for (int k = j+1; k < F.cols(); k++) {
				//if (j == k) continue;
				//AM[F(i, j)].insert(F(i, k));
				//TODO: Verify that this is correct!
				AdM[F(i, j)].insert(F(i, k));
				AdM[F(i, k)].insert(F(i, j));
			}
		}
	}

}

void MLEigenSolver::constructMassMatrix() {}
void MLEigenSolver::constructStiffnessMatrix() {}

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

void MLEigenSolver::constructSampleHierarchy(const Eigen::VectorXd& samples)
{
	printf("____Sampling on %d levels \n", samples.size());
	for (int i = 0; i < samples.size(); ++i){
		SubspaceSize[i] = samples(i);
		printf("______Level %d -> Subspace size: %d\n", i, SubspaceSize[i]);
	}
	sampleSize = SubspaceSize[0];

	SampleMap.resize(V.rows());
	constructFarthestPointSample(V, AdM, Sample, SubspaceSize, geoDist);
	cout << "___Assigning level \n";
	sampleLevel.resize(V.rows());

	SampleMap.setConstant(-1);
	sampleLevel.setConstant(-1);

	// for level n-1
	for (int i = 0; i < SubspaceSize[level - 1]; ++i){
		sampleLevel(Sample(i))	= level - 1;
		SampleMap(Sample(i))	= i;
	}

	for (int i = level - 1; i > 0; --i){
		for (int j = SubspaceSize[i]; j < SubspaceSize[i - 1]; ++j){
			sampleLevel(Sample(j))	= i - 1;
			SampleMap(Sample(j))	= j;
		}
	}
}

void MLEigenSolver::constructBasis(double support, int axis, double offset) {}

void MLEigenSolver::constructBasisCoarseOnly(double support, int axis, double offset) {}

void MLEigenSolver::constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support) {}

void MLEigenSolver::constructBasisNonSmooth(double support, int axis, double offset) {}

void MLEigenSolver::constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int level, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter) {}

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
		if(iter==0)
			status = computeEigenGPU(S_, M_, EigVectsReducedNonLifted, EigValsReduced);
		else {
			eps *= 10;
			status = computeEigenGPU(S_+eps*Id, M_, EigVectsReducedNonLifted, EigValsReduced);
			cout << "Shift: " << eps << endl;
		}
		iter++;
	} while (status != 0 && iter<10);

	std::cout << "Succesful and lifted to fine resolution \n";
	//EigVectsReduced = Basis * EigVectsReducedNonLifted;
}

void MLEigenSolver::constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis, double offset) {}

/* Subspace Iteration methods */
void MLEigenSolver::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset) { }

void MLEigenSolver::performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift){} 

void MLEigenSolver::performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

void MLEigenSolver::performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift) {}

void MLEigenSolver::performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

void MLEigenSolver::performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType) {}

void MLEigenSolver::checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const
{
	convRate.resize(A.rows());
	Eigen::VectorXd Kq, AMq, diff;

	for (int i = 0; i < A.rows(); i++) {
		Kq		= SK * Q.col(i);
		AMq		= A(i)*MK*Q.col(i);
		diff	= Kq - AMq;

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
		if ( M.rows()>7500 ) {
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

	xMin	= V.col(axis).minCoeff();
	xMax	= V.col(axis).maxCoeff();
	xScale	= offset;
	xBound	= xMin + xScale*(xMax - xMin);

	/* Create selector matrix */
	vector<Eigen::Triplet<double>> STriplet;
	int vCounter = 0;

	/* Selector matrix for eigen-Functions */
	if (F.cols()==3 && S.rows()==V.rows()) {
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
	else if (F.cols()==4 && S.rows()==3*V.rows()) {
		STriplet.reserve(3 * round((1.0 - xScale)*V.rows()));
		for (int i = 0; i < V.rows(); ++i) {
			if (V(i, axis) >= xBound) {							// only select entries corresponding to selected vertices
				for (int j = 0; j < 3; j++) {
					STriplet.push_back(Eigen::Triplet<double>(3 * i + j, 3 * vCounter + j, 1));
				}
				vCounter++;
			}
		}
		SelMatrix.resize(3*V.rows(), 3*vCounter);
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
		if (MLoc.rows()>7500) {
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

/* Getters of the eigen-values + vectors*/
const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsRef() const			{ return EigVectsRef; }
const Eigen::VectorXd& MLEigenSolver::getEigenVectorsRef(int id) const		
{ 
	if		(id >= EigVectsRef.cols())	return EigVectsRef.col(EigValsRef.rows()-1);
	else if	(id< 0)						return EigVectsRef.col(0);

	return EigVectsRef.col(id); 
}
const Eigen::VectorXd& MLEigenSolver::getEigenValuesRef() const				{ return EigValsRef; }
const double MLEigenSolver::getEigenValueRef(int id) const					
{ 
	if		(id >= EigValsRef.rows())	return EigValsRef(EigValsRef.rows()-1);
	else if	(id< 0)						return EigValsRef(0);

	return EigValsRef(id); 
}
const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsReduced() const		{ return EigVectsReduced; }
const Eigen::VectorXd& MLEigenSolver::getEigenVectorsReduced(int id) const
{
	if (id >= EigVectsReduced.cols())	return EigVectsReduced.col(EigValsReduced.rows()-1);
	else if (id< 0)						return EigVectsReduced.col(0);

	return EigVectsReduced.col(id); 
}
const Eigen::VectorXd& MLEigenSolver::getEigenValuesReduced() const			{ return EigValsReduced; }
const double MLEigenSolver::getEigenValueReduced(int id) const
{
	if (id >= EigValsReduced.rows())	return EigValsReduced(EigValsReduced.rows() - 1);
	else if (id< 0)						return EigValsReduced(0);

	return EigValsReduced(id); 
}
const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsSIM() const			{ return EigVectsSIM; }
const Eigen::VectorXd& MLEigenSolver::getEigenVectorsSIM(int id) const
{
	if (id >= EigVectsSIM.cols())		return EigVectsSIM.col(EigValsSIM.rows()-1);
	else if (id< 0)						return EigVectsSIM.col(0);
	return EigVectsSIM.col(id); 
}
const Eigen::VectorXd& MLEigenSolver::getEigenValuesSIM() const				{ return EigValsSIM; }
const double MLEigenSolver::getEigenValueSIM(int id) const
{
	if (id >= EigValsSIM.rows())		return EigValsSIM(EigValsSIM.rows() - 1);
	else if (id< 0)						return EigValsSIM(0);

	return EigValsSIM(id); 
}

void MLEigenSolver::LBSTest_pos(igl::opengl::glfw::Viewer &viewer, int numBasis, int id, double trans, int transAxis, double angle, int rotAxis)
{
	if (id >= V.rows()) id = 0;
	cout << "Set up angles and translation \n";
	double c1;
	double s1;
	Eigen::MatrixXd Q(3, 4); Q.setZero();
	Eigen::VectorXd qBar(12), q(12*numBasis);
	Eigen::SparseMatrix<double> UU(3*V.rows(), 12*numBasis);
	vector<Eigen::Triplet<double>> UTrip;
	UTrip.reserve(12 * numBasis* V.rows());

	cout << "Set up affine transformaiton matrix \n";
	for (int k = 0; k < numBasis; ++k)
	{
		srand(time(NULL));
		c1 = cos(k*angle / 180.0 * M_PI);
		s1 = sin(k*angle / 180.0 * M_PI);

		//switch (rotAxis)
		switch (k%3)
		{
		case 0:
			Q(0, 0) = 1.0;
			Q(1, 1) = c1;
			Q(1, 2) = -s1;
			Q(2, 1) = s1;
			Q(2, 2) = c1;
			break;
		case 1:
			Q(1, 1) = 1.0;
			Q(2, 2) = c1;
			Q(2, 0) = -s1;
			Q(0, 2) = s1;
			Q(0, 0) = c1;
			break;
		case 2:
			Q(2, 2) = 1.0;
			Q(0, 0) = c1;
			Q(0, 1) = -s1;
			Q(1, 0) = s1;
			Q(1, 1) = c1;
			break;
		default:
			break;
		}
		cout << "Transform affine matrix to vector. \n";
		// translation element
		Q(transAxis, k%3) = (1+k%3) * trans;
		qBar(0) = Q(0, 0);
		qBar(1) = Q(1, 0);
		qBar(2) = Q(2, 0);
		qBar(3) = Q(0, 1);
		qBar(4) = Q(1, 1);
		qBar(5) = Q(2, 1);
		qBar(6) = Q(0, 2);
		qBar(7) = Q(1, 2);
		qBar(8) = Q(2, 2);
		qBar(9) = Q(0, 3);
		qBar(10) = Q(1, 3);
		qBar(11) = Q(2, 3);

		q.block(12 * k, 0, 12, 1) = qBar;
	}


	/* Dijkstra distance */
	cout << "Set up distance \n";
	Eigen::VectorXd dist;
	Eigen::VectorXd handleId(numBasis);
	handleId(0) = id;
	Eigen::MatrixXd weight(V.rows(), numBasis);
	weight.setZero();
	dist.resize(V.rows());
	dist.setConstant(std::numeric_limits<double>::infinity());
	printf("Size: V: %d | AdV: %d | dist: %d \n", V.rows(), AdM.size(), dist.rows());

	double maxDist;

	Eigen::VectorXi::Index maxIndex;
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		if (k < (numBasis - 1)) {
			dist.maxCoeff(&maxIndex);
			handleId(k + 1) = maxIndex;
		}

		if (k == 0) {
			maxDist = dist.maxCoeff() / 2;
		}
	}

	cout << "Set up weight \n";
	double			aCoef = 2.0 / (pow(maxDist, 3));
	double			bCoef = -3.0 / (pow(maxDist, 2));
	cout << "Handles: " << handleId.transpose() << endl;

	dist.setConstant(std::numeric_limits<double>::infinity());
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		for (int i = 0; i < V.rows(); ++i) {
			if (dist(i) <= maxDist)
				weight(i,k) = aCoef*pow(dist(i), 3) + bCoef*pow(dist(i), 2) + 1.0;
		}
		dist.setConstant(std::numeric_limits<double>::infinity());
	}

	/* Partition of unity of the weight */
	cout << "partition of unity\n";
	cout << "_PreV: " << weight.row(0) << endl;
	for (int i = 0; i < V.rows(); ++i) {
		double rowSum = weight.row(i).sum();
		Eigen::VectorXd rowSumVector(numBasis);
		rowSumVector.setConstant(rowSum);
		weight.row(i) = weight.row(i).cwiseQuotient(rowSumVector);
	}
	cout << "_Post: " << weight.row(0) << endl;
	
	for (int k = 0; k < numBasis; ++k) {
		cout << "Set up the vectors of basis \n";
		/* Setting up the subspace UU*/
		for (int i = 0; i < V.rows(); ++i) {
			//if (dist(i) > maxDist) continue;
			/* For the first 3 vertex coordinates (i.e. x, y, and z) */
			for (int j = 0; j < 3; ++j) {
				double val = V(i, j);
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 3 * j + 0, weight(i, k)*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 3 * j + 1, weight(i, k)*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 3 * j + 2, weight(i, k)*val));
			}
			/* For the the translation/last column */
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 9, weight(i, k)));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 10, weight(i, k)));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 11, weight(i, k)));
		}
	}
	cout << "Set up basis matrix \n";
	UU.setFromTriplets(UTrip.begin(), UTrip.end());
	
	/* Setting position */
	Eigen::VectorXd pos;
	Eigen::MatrixXd Pos;
	cout << "Compute new position \n";
	pos = UU*q;

	cout << "Transform the new position \n";
	Pos.resize(V.rows(), V.cols());
	for (int i = 0; i < V.rows(); ++i) {
		for (int j = 0; j < V.cols(); ++j) {
			Pos(i, j) = pos(3*i+j);
		}
	}

	cout << "Init pos: " << V.row(0) << endl;
	cout << "After: " << Pos.row(0) << endl;

	Eigen::MatrixXd VCol;
	igl::jet(-weight.col(0), true, VCol);
	cout << "Show vertices \n";
	viewer.data().set_vertices(Pos);
	viewer.data().compute_normals();
	viewer.data().set_colors(VCol);

	viewer.data().points.resize(0,6);
	for(int i=0; i<numBasis; i++)
		viewer.data().add_points(Pos.row(handleId(i)), Eigen::RowVector3d(1.0, 0.1, 0.1));
}

void MLEigenSolver::LBSTest_vel(igl::opengl::glfw::Viewer &viewer, int numBasis, int id, double trans, int transAxis, double angle, int rotAxis)
{
	if (id >= V.rows()) id = 0;
	cout << "Set up angles and translation \n";
	double c1;
	double s1;
	Eigen::MatrixXd Q(3, 4); Q.setZero();
	Eigen::VectorXd qBar(12), q(12 * numBasis);
	Eigen::SparseMatrix<double> UU(3 * V.rows(), 12 * numBasis);
	vector<Eigen::Triplet<double>> UTrip;
	UTrip.reserve(12 * numBasis* V.rows());

	cout << "Set up affine transformaiton matrix \n";
	for (int k = 0; k < numBasis; ++k)
	{
		srand(time(NULL));
		c1 = cos(k*angle / 180.0 * M_PI);
		s1 = sin(k*angle / 180.0 * M_PI);

		//switch (rotAxis)
		switch (k % 3)
		{
		case 0:
			Q(0, 0) = 1.0;
			Q(1, 1) = c1;
			Q(1, 2) = -s1;
			Q(2, 1) = s1;
			Q(2, 2) = c1;
			break;
		case 1:
			Q(1, 1) = 1.0;
			Q(2, 2) = c1;
			Q(2, 0) = -s1;
			Q(0, 2) = s1;
			Q(0, 0) = c1;
			break;
		case 2:
			Q(2, 2) = 1.0;
			Q(0, 0) = c1;
			Q(0, 1) = -s1;
			Q(1, 0) = s1;
			Q(1, 1) = c1;
			break;
		default:
			break;
		}
		cout << "Transform affine matrix to vector. \n";
		Eigen::MatrixXd Id(3, 4);
		Id.setIdentity();
		Q = Q - Id;


		// translation element
		Q(transAxis, k % 3) = (1 + k % 3) * trans;
		qBar(0) = Q(0, 0);
		qBar(1) = Q(1, 0);
		qBar(2) = Q(2, 0);
		qBar(3) = Q(0, 1);
		qBar(4) = Q(1, 1);
		qBar(5) = Q(2, 1);
		qBar(6) = Q(0, 2);
		qBar(7) = Q(1, 2);
		qBar(8) = Q(2, 2);
		qBar(9) = Q(0, 3);
		qBar(10) = Q(1, 3);
		qBar(11) = Q(2, 3);

		q.block(12 * k, 0, 12, 1) = qBar;
	}


	/* Dijkstra distance */
	cout << "Set up distance \n";
	Eigen::VectorXd dist;
	Eigen::VectorXd handleId(numBasis);
	handleId(0) = id;
	Eigen::MatrixXd weight(V.rows(), numBasis);
	weight.setZero();
	dist.resize(V.rows());
	dist.setConstant(std::numeric_limits<double>::infinity());
	printf("Size: V: %d | AdV: %d | dist: %d \n", V.rows(), AdM.size(), dist.rows());

	double maxDist;

	Eigen::VectorXi::Index maxIndex;
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		if (k < (numBasis - 1)) {
			dist.maxCoeff(&maxIndex);
			handleId(k + 1) = maxIndex;
		}

		if (k == 0) {
			maxDist = dist.maxCoeff() / 2;
		}
	}

	cout << "Set up weight \n";
	double			aCoef = 2.0 / (pow(maxDist, 3));
	double			bCoef = -3.0 / (pow(maxDist, 2));
	cout << "Handles: " << handleId.transpose() << endl;

	dist.setConstant(std::numeric_limits<double>::infinity());
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		for (int i = 0; i < V.rows(); ++i) {
			if (dist(i) <= maxDist)
				weight(i, k) = aCoef*pow(dist(i), 3) + bCoef*pow(dist(i), 2) + 1.0;
		}
		dist.setConstant(std::numeric_limits<double>::infinity());
	}

	/* Partition of unity of the weight */
	cout << "partition of unity\n";
	cout << "_PreV: " << weight.row(0) << endl;
	for (int i = 0; i < V.rows(); ++i) {
		double rowSum = weight.row(i).sum();
		Eigen::VectorXd rowSumVector(numBasis);
		rowSumVector.setConstant(rowSum);
		weight.row(i) = weight.row(i).cwiseQuotient(rowSumVector);
	}
	cout << "_Post: " << weight.row(0) << endl;

	for (int k = 0; k < numBasis; ++k) {
		cout << "Set up the vectors of basis \n";
		/* Setting up the subspace UU*/
		for (int i = 0; i < V.rows(); ++i) {
			//if (dist(i) > maxDist) continue;
			/* For the first 3 vertex coordinates (i.e. x, y, and z) */
			for (int j = 0; j < 3; ++j) {
				double val = V(i, j);
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 3 * j + 0, (weight(i, k))*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 3 * j + 1, (weight(i, k))*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 3 * j + 2, (weight(i, k))*val));
			}
			/* For the the translation/last column */
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 9 , (weight(i, k))));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 10, (weight(i, k))));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 11, (weight(i, k))));
		}
	}
	cout << "Set up basis matrix \n";
	UU.setFromTriplets(UTrip.begin(), UTrip.end());

	/* Setting position */
	Eigen::VectorXd pos;
	Eigen::MatrixXd Pos;
	cout << "Compute new position \n";
	pos = UU*q;

	cout << "Transform the new position \n";
	Pos.resize(V.rows(), V.cols());
	for (int i = 0; i < V.rows(); ++i) {
		for (int j = 0; j < V.cols(); ++j) {
			Pos(i, j) = pos(3 * i + j);
		}
	}

	cout << "Init pos: " << V.row(0) << endl;
	cout << "After: " << V.row(0)+ Pos.row(0) << endl;

	Eigen::MatrixXd VCol;
	igl::jet(-weight.col(0), true, VCol);
	cout << "Show vertices \n";
	viewer.data().set_vertices(Pos+V);
	viewer.data().compute_normals();
	viewer.data().set_colors(VCol);

	viewer.data().points.resize(0, 6);
	for (int i = 0; i<numBasis; i++)
		viewer.data().add_points(Pos.row(handleId(i)), Eigen::RowVector3d(1.0, 0.1, 0.1));
}


void MLEigenSolver::testParallelFactorization(bool isParallel)
{
	std::cout << "Testing parallel factorization \n";	

	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	
	//bool isParallel = false;

	//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factor;	
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;

	/* Generating random matrix as input */
	Eigen::MatrixXd input, output;
	const int numCols = 100;
	generateRandomMatrix(S.rows(), numCols, input);
	output.resize(input.rows(), input.cols());

	/* Normalizing the input */
	for (int i = 0; i < numCols; ++i) {
		double mNorm = input.col(i).transpose() * M * input.col(i);
		mNorm = (double) 1.0 / sqrt(mNorm);
		input.col(i) = mNorm * input.col(i);
	}
	//input = M*input;


	cout << "___Factorizing \n";
	
	/* Factorizing (Serial) */
	if (!isParallel)
	{		
		std::cout << " (Serial)...";
		t1 = chrono::high_resolution_clock::now();
		factor.compute(S);
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;

		std::cout << " Solving...";
		t1 = chrono::high_resolution_clock::now();
		for (int i = 0; i < numCols; ++i) {
			output.col(i) = factor.solve(input.col(i));
		}

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;

		printVector(output.row(0), true, 4);
	}
	else
	{
		std::cout << " (Parallel)...";
		t1 = chrono::high_resolution_clock::now();

		//Eigen::initParallel();
		factor.compute(S);
		//factor.pardisoParameterArray()[33] = 1;
		//auto const& cfactor = factor;
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;

		std::cout << " Solving...";
		t1 = chrono::high_resolution_clock::now();

		const int		NUM_OF_THREADS = omp_get_num_procs()/2;
		//const int		NUM_OF_THREADS = 2;
		omp_set_num_threads(NUM_OF_THREADS);
		//Eigen::setNbThreads(NUM_OF_THREADS);
		int tid, ntids, ipts, istart, iproc;
		int iCol=0;

		auto const& cinput = input; // Not sure if "input" is already const, let's make sure
		auto const& cfactor = factor; // Make sure we have a const reference to "factor"

		//vector<Eigen::PardisoLDLT<Eigen::SparseMatrix<double>>> factorPar(NUM_OF_THREADS);
		
		//factorPar[0] = &(*factor);
		//factorPar.resize(NUM_OF_THREADS);
		//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorParA = factor;

		//Eigen::SparseMatrix<double> MatS;
		//factor.getMatrix(MatS);
		//std::cout << "Matrix: \n" << MatS.topLeftCorner(10, 10) << std::endl;
		//std::cout << "S: \n" << S.topLeftCorner(10, 10) << std::endl;

		//std::copy(factor, factorPar[0]);

#pragma omp parallel private(tid,ntids,ipts,istart,iCol)
//#pragma omp parallel private(factor)
		{
			iproc = omp_get_num_procs();
			tid = omp_get_thread_num();
			ntids = omp_get_num_threads();
			ipts = (int)ceil((double)numCols / (double)ntids);
			istart = tid * ipts;
			if (tid == ntids - 1) ipts = numCols - istart;
			if (ipts <= 0) ipts = 0;
			//factor.compute(S);
		
			std::printf("ID %d with %d items (%d->%d) \n", tid, ipts, istart, istart + ipts);
			Eigen::VectorXd temp(input.rows());
		
			for (iCol = istart; iCol < (istart + ipts); ++iCol) {
				//output.col(iCol) = factor.solve(input.col(iCol));
				//factor._solve_impl(input.col(iCol), output.col(iCol));
				temp = cfactor.solve(cinput.col(iCol));
#pragma omp critical
				{
					output.col(iCol) = temp;
				}
			}
		}

		///int n = 4;
		///omp_set_num_threads(n);
		///Eigen::setNbThreads(n);
		///factor.compute(S);
		///
		///for (int i = 0; i < numCols; ++i) {
		///	output.col(i) = factor.solve(input.col(i));
		///}

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds" << endl;

		printVector(output.row(0), true,4);

	}
	
}