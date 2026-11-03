#include "LBEigenSolver.h"

#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/invert_diag.h>
#include <igl/avg_edge_length.h>
#include <igl/jet.h>

//#include <Eigen/PardisoSupport>

#include <queue>

namespace HSIM {

namespace {
	/* Size q of the iterated subspace for numEigs sought eigenpairs (Bathe's rule, as in the paper). */
	int subspaceSizeFor(int numEigs) { return std::max(2 * numEigs, numEigs + 8); }
}

/**
*  @brief Reading a (triangle) mesh specified by the input.
*
*  @details
*   Reading a (triangle) mesh specified by the input 'meshFile',
*	currently it supports 'obj' and 'off' file format only
*  @param meshFile  Location of the mesh
*/
void LBEigenSolver::readMesh(const string& meshFile)
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

void LBEigenSolver::setup(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes)
{
	/* Initial setup*/
	setV(V);
	setF(F);
	constructMassMatrix();
	initiateAdM();
	constructStiffnessMatrix();
}

/**
*  @brief Setting up the system to have required levels.
*build neighborhood structure
*  @details
*   Setting up the system to have required levels.
*  @param n Number of specified levels of the system.
*/
void LBEigenSolver::setInitialSystem(int n)
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
*  @brief Setting up the adjacency matrix.
*
*  @details
*   Setting up the adjacency matrix.
*/
void LBEigenSolver::initiateAdM()
{
#ifdef VERBOSE
	cout << "> Creating neighborhood structure of the mesh. \n";
#endif
	AdM.clear();
	AdM.resize(V.rows());

	for (int i = 0; i < F.rows(); i++)
	{
		for (int j = 0; j < F.cols(); j++) {
			/* vertex-to-vertex neighborhood */
			for (int k = j + 1; k < F.cols(); k++) {
				//if (j == k) continue;
				//AM[F(i, j)].insert(F(i, k));
				//TODO: Verify that this is correct!
				AdM[F(i, j)].insert(F(i, k));
				AdM[F(i, k)].insert(F(i, j));
			}
		}
	}

}

void LBEigenSolver::computeArea()
{
	/* Constructing Average length of the edges */
	avgEdgeLength = igl::avg_edge_length(V, F);

	Eigen::VectorXd dArea;
	igl::doublearea(V, F, dArea);	
	surfaceArea = dArea.sum() / 2.0;
}

void LBEigenSolver::constructMassMatrix()
{
	#ifdef VERBOSE
		cout << "> Creating mass matrix for the surface mesh \n";
	#endif // VERBOSE	
	/* Construct Mass matrix M*/
	//igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
	igl::invert_diag(M, Minv);

	Eigen::VectorXd dblArea;
	igl::doublearea(V, F, dblArea);
	surfaceArea = dblArea.sum() / 2.0;
}

void LBEigenSolver::constructStiffnessMatrix()
{
	#ifdef VERBOSE
		cout << "> Creating stiffness matrix (Cotangent Laplacian). \n";
	#endif 
	igl::cotmatrix(V, F, S);
	S = -S;
}

/* Main functions for user */
tuple<Eigen::VectorXd, Eigen::MatrixXd> LBEigenSolver::eigs(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes, double tol, int maxIters, double shiftParam)
{
	setup(V, F, numModes);
	buildHierarchy(V, F, numModes);
	zhaoShiftParam = shiftParam;
	performSubspaceIterationMethod(LBEigenSolver::SIMType::SIM_MultiLevel, numModes, maxIters, tol, true, LBEigenSolver::ShiftType::SHIFT_ZHAO, 0, 0.0);
	std::cout << "COMPUTATION IS SUCCESSFUL \n";
	return make_tuple(EigValsSIM, EigVectsSIM);
}

void LBEigenSolver::buildHierarchy()
{

}

void LBEigenSolver::buildHierarchy(int numModes)
{
	buildHierarchy(V, F, numModes);
}

void LBEigenSolver::buildHierarchy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int numModes)
{
	std::cout << "1. Building Hierarchy. \n";
	#ifdef VERBOSE
		std::cout << "Number of eigenvalues to compute = " << numModes << std::endl;
	#endif

	/* Sampling levels */
	Eigen::VectorXd sampSize;
	double levelRatio = 10.0;
	setSamplingLevels(sampSize, numModes, levelRatio);
	numModesHierarchy = numModes;
	setInitialSystem(sampSize.rows());
	constructSampleHierarchy(sampSize);

	/* Creating prolongation operator */
	double numOfSupport = 7.5;
	int defAxis = 0;
	double clampRegion = 0.0;
	setClampBasis(isClampedBasis);
	constructBasis(numOfSupport, defAxis, clampRegion);
	solveRestrictedProblem();
}

/* Construction of the subspace and/or hierarchy */
void LBEigenSolver::setSamplingLevels(Eigen::VectorXd& sampSize, int numEigen, double levelRatio)
{
	if (numEigen < 1) throw std::invalid_argument("setSamplingLevels: the number of eigenpairs must be positive");
	if (levelRatio <= 1.0) throw std::invalid_argument("setSamplingLevels: the level ratio must be larger than 1");

	/* The coarsest level must be able to hold the whole iterated subspace (q vectors). */
	int		sizeSubsp = subspaceSizeFor(numEigen);

	vector<int> sampleLevels;
	sampleLevels.reserve(10);
	//int SAMPLE_CUT_OFF = max(100, sizeSubsp);			//500
	int SAMPLE_CUT_OFF = (sizeSubsp);			//500
	int SAMPLE_LOW_BOUND = 2 * SAMPLE_CUT_OFF;
	int numSample = V.rows();
	#ifdef VERBOSE
		std::cout << "cut-out" << SAMPLE_CUT_OFF << "| Low bound " << SAMPLE_LOW_BOUND << ": " << std::endl;
	#endif

	/* automated */
	do {
		numSample = (int)round(numSample / levelRatio);
		if (numSample < SAMPLE_CUT_OFF) break;
		sampleLevels.push_back(numSample);		
	} while (numSample > SAMPLE_LOW_BOUND);

	if (sampleLevels.size() < 1)
	{
		throw std::invalid_argument("setSamplingLevels: the mesh (" + to_string(V.rows()) + " vertices) is too coarse to compute "
			+ to_string(numEigen) + " eigenpairs with level ratio " + to_string(levelRatio)
			+ "; at least " + to_string((int)ceil(levelRatio * sizeSubsp)) + " vertices are needed");
	}

	sampSize.resize(sampleLevels.size());
	for (int ii = 0; ii < sampleLevels.size(); ++ii)
		sampSize(ii) = sampleLevels[ii];
}

void LBEigenSolver::constructSampleHierarchy(const Eigen::VectorXd& samples)
{
	std::cout << "____ Performing sampling on the surface \n";
	#ifdef VERBOSE
		printf("____Sampling on %d levels \n", samples.size());
	#endif
	for (int i = 0; i < samples.size(); ++i) {
		SubspaceSize[i] = samples(i);
		//printf("______Level %d -> Subspace size: %d\n", i, SubspaceSize[i]);
	}
	sampleSize = SubspaceSize[0];

	SampleMap.resize(V.rows());
	constructFarthestPointSample(V, AdM, Sample, SubspaceSize, geoDist);
	//cout << "___Assigning level \n";
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

void LBEigenSolver::constructBasis(double support, int axis, double offset)
{
	cout << "____ Constructing prolongation operator \n";

	/* Support radius r_k of the basis functions on level k: a disc of radius r_k covers (on average) 'support'
	 * samples, pi r_k^2 = support * A / m_k. NOTE: the original implementation used twice the surface area here
	 * (double-area sum); this factor is kept so that results are identical to the published ones. */
	const double areaFactor = 2.0;
	for (int i = 0; i < level; i++) {
		maxNeighDist[i] = sqrt((areaFactor*surfaceArea*support) / (M_PI*SubspaceSize[i]));
		//printf("Radius %d is %.4f (avg edge: %.4f)\n", i, maxNeighDist[i], avgEdgeLength);
	}

	for (int i = 0; i < level; i++) {
		constructRadialBasisFunctions(i, axis, isClampedBasis, offset, support);
		//printf("basis level %d: done \n", i);
	}

	formPartitionOfUnityMultiLevel();

	Basis.resize(V.rows(), V.rows());
	Basis.setIdentity();
	for (int i = 0; i < level; i++)
		Basis = Basis * U[i];

	#ifdef VERBOSE
		printf("Prolongation sparsity:  %.4f (%d/%d) \n", (double)Basis.nonZeros() / (double)Basis.rows(), Basis.nonZeros(), Basis.rows());
	#endif
}

void LBEigenSolver::constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	#ifdef VERBOSE
		cout << "___Prolongation construction (level " << k <<" )... ";
	#endif
	t1 = chrono::high_resolution_clock::now();

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef = 2.0 / (pow(maxNeighDist[k], 3));
	double			bCoef = -3.0 / (pow(maxNeighDist[k], 2));
	int				i = 0;

	const int		NUM_OF_THREADS = omp_get_max_threads();	// respects OMP_NUM_THREADS, no global side effect
	UTriplet.resize(NUM_OF_THREADS);

	int tid, ntids, ipts, istart;

	vector<int> elementCounter;
	elementCounter.resize(NUM_OF_THREADS);
	int totalElementInsertions = 0;

	/* Manually break the chunck for parallelization */
#pragma omp parallel num_threads(NUM_OF_THREADS) private(tid,ntids,ipts,istart,i)
	{
		tid = omp_get_thread_num();
		ntids = omp_get_num_threads();
		ipts = (int)ceil(1.00*(double)SubspaceSize[k] / (double)ntids);
		istart = tid * ipts;
		if (tid == ntids - 1) ipts = SubspaceSize[k] - istart;
		if (ipts <= 0) ipts = 0;

		// Instantiate D once, used for all Dijkstra
		Eigen::VectorXd				D(V.rows());
		D.setConstant(numeric_limits<double>::infinity());

		if(k==0)
			UTriplet[tid].reserve(2 * round((double)ipts / (double)SubspaceSize[0]) * support * V.rows());
		else 
			UTriplet[tid].reserve(2 * round((double)ipts / (double)SubspaceSize[0]) * support * SubspaceSize[k-1]);

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= SubspaceSize[k]) break;
			/* Dijkstra basis function */
			//ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist[0], i, SubspaceSize[0], 0, isClamped, axis, offset, UTriplet[tid], elementCounter[tid]);
			constructDijkstraDisk(Sample(i), D, aCoef, bCoef, maxNeighDist[k], i, k, axis, offset, UTriplet[tid], elementCounter[tid]);
		}
	} /* End of OpenMP parallelization */

	int totalElements = 0;
	for (int j = 0; j < NUM_OF_THREADS; j++) {
		totalElements += UTriplet[j].size();
		totalElementInsertions += elementCounter[j];
	}

	/* Combining the Triplet */
	AllTriplet.resize(totalElements);
#pragma omp parallel for
	for (int j = 0; j < NUM_OF_THREADS; j++) {
		int tripSize = 0;
		for (int k = 0; k < j; k++) {
			tripSize += UTriplet[k].size();
		}
		std::copy(UTriplet[j].begin(), UTriplet[j].end(), AllTriplet.begin() + tripSize);
	}
	//Basis.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	if(k==0)
		U[0].resize(V.rows(), SubspaceSize[k]);
	else 
		U[k].resize(SubspaceSize[k-1], SubspaceSize[k]);

	U[k].setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	#ifdef VERBOSE
		cout << "in " << duration.count() << " seconds" << endl;
	#endif
	#ifdef VERBOSE
		printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", U[k].rows(), U[k].cols(), (double)(U[k].nonZeros() / (double)U[k].rows()));
	#endif
}

void LBEigenSolver::constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter)
{
	priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistPQueue;

	/* Setting the bound */
	double xBound = -numeric_limits<double>::infinity();
	if (isClampedBasis) {
		double xMin = V.col(axis).minCoeff();
		double xMax = V.col(axis).maxCoeff();
		xBound = xMin + offset*(xMax - xMin);		// within this bound, the elements will be discarded
	}

	int elCounter = 1;

	// Computing distance for initial sample points S
	D(s1) = 0.0f;
	VertexPair vp{ s1,D(s1) };
	DistPQueue.push(vp);

	double distFromCenter = numeric_limits<double>::infinity();

	// For other vertices in mesh
	do {
		if (DistPQueue.size() == 0) break;
		VertexPair vp1 = DistPQueue.top();
		distFromCenter = vp1.distance;
		DistPQueue.pop();

		// Updating the distance for neighbors of vertex of lowest distance in priority queue
		auto const& elem = AdM[vp1.vId];
		for (auto it = elem.begin(); it != elem.end(); ++it) {
			/* Regular dijkstra */
			double dist;
			dist = (V.row(*it) - V.row(vp1.vId)).norm();
			double tempDist = vp1.distance + dist;

			/* Correct distance using Euclidean Distance */
			//double tempDist = (V.row(s1) - V.row(*it)).norm();

			/* updating the distance */
			if (tempDist < D(*it)) {
				D(*it) = tempDist;
				VertexPair vp2{ *it,tempDist };
				DistPQueue.push(vp2);
			}
		}
	} while (distFromCenter <= maxDist);

	if (k == 0) {
		for (int i = 0; i < V.rows(); ++i) {
			double aVal = D(i);
			if (aVal <= maxDist) {
				if (isClampedBasis && V(i, axis) < xBound) continue;

				/* Polynomial mapping (3rd Order) */
				double val = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;					// face
				UTriplet.push_back(Eigen::Triplet<double>(i, col, val));
			}
		}
	}
	else {
		for (int i = 0; i < SubspaceSize[k-1]; ++i) {
			double aVal = D(Sample(i));
			if (aVal <= maxDist) {
				if (isClampedBasis && V(Sample(i), axis) < xBound) continue;

				/* Polynomial mapping (3rd Order) */
				double val = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;
				UTriplet.push_back(Eigen::Triplet<double>(i, col, val));

			}
		}
	}
	D.setConstant(numeric_limits<double>::infinity());
}

void LBEigenSolver::formPartitionOfUnityMultiLevel()
{
	#ifdef VERBOSE
		cout << "Partition of unity \n";
	#endif
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
			if (abs(rowSum(k)) < (10.0 * std::numeric_limits<double>::epsilon())) {
				diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0));
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

void LBEigenSolver::solveRestrictedProblem()
{
	double eps = 1e-16;
	Eigen::SparseMatrix<double> Id(Basis.cols(), Basis.cols());
	Id.setIdentity();

	cout << "____ Computing the approximated eigenvectors.\n";
	

	Eigen::SparseMatrix<double> S_, M_;			//reduced stiffness and mass matrix
	S_ = Basis.transpose() * S * Basis;
	M_ = Basis.transpose() * M * Basis;
	
	#ifdef VERBOSE	
		cout << "> Computing the reduced eigenproblem (in GPU). \n";
	#endif	


	Eigen::MatrixXd SDense(S_);
	Eigen::MatrixXd MDense(M_);

	/* Solve the dense problem; if the solver fails, retry with an increasingly regularised stiffness matrix. */
	int status = computeEigenGPUDense(SDense, MDense, EigVectsReducedNonLifted, EigValsReduced);
	for (int iter = 1; status != 0 && iter < 10; ++iter) {
		eps *= 10;
		Eigen::MatrixXd SReg = SDense;
		SReg.diagonal().array() += eps;
		status = computeEigenGPUDense(SReg, MDense, EigVectsReducedNonLifted, EigValsReduced);
		#ifdef VERBOSE
			cout << "Shift: " << eps << endl;
		#endif
	}
	if (status != 0)
		throw std::runtime_error("solveRestrictedProblem: the dense eigensolver failed on the coarsest level");

	#ifdef VERBOSE
		std::cout << "Succesful and lifted to fine resolution \n";
	#endif

	EigVectsReduced = Basis * EigVectsReducedNonLifted;
}

void LBEigenSolver::constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis, double offset)
{
	SelMat.resize(M.rows(), M.cols());
	vector<Eigen::Triplet<double>> STriplet;
	STriplet.reserve(M.nonZeros());

	/* Setting the bound */
	double xBound = -numeric_limits<double>::infinity();
	if (isClampedBasis) {
		double xMin = V.col(axis).minCoeff();
		double xMax = V.col(axis).maxCoeff();
		xBound = xMin + offset * (xMax - xMin);		// within this bound, the elements will be discarded
	}

	for (int i = 0; i < V.rows(); i++) {
		if (V(i, axis) >= xBound) {							// only select entries corresponding to selected vertices
			// entries for the selector matrix			
			STriplet.push_back(Eigen::Triplet<double>(i, i, 1.0));
		}
	}

	SelMat.setFromTriplets(STriplet.begin(), STriplet.end());
	#ifdef VERBOSE
		printf("__Dim of this selector matrix: %dx%d \n", SelMat.rows(), SelMat.cols());
	#endif

}

/* Subspace Iteration methods */
void LBEigenSolver::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset)
{
	
	cout << "2. Computing the eigenvectors and eigenvalues.\n";
	if (U.size() < 1 || EigVectsReducedNonLifted.rows() < 1 || level < 1)
	{
		if (V.rows() < 1) {
			std::cout << "There is no mesh data. Please load one before computing the eigenvalues of its Laplacian. Thank you.\n";
			return;
		}
		std::cout << "!!! The hierarchy has not been constructed. We'll help you build one. No worries. \n";
		setup(V, F, numEigs);
		buildHierarchy(numEigs);
	}

	
	if (numEigs < 1)
		throw std::invalid_argument("performSubspaceIterationMethod: the number of eigenpairs must be positive");
	if (maxIter < 1)
		throw std::invalid_argument("performSubspaceIterationMethod: maxIter must be positive");
	if (!(zhaoShiftParam >= 0.0 && zhaoShiftParam < 1.0))
		throw std::invalid_argument("performSubspaceIterationMethod: the shift parameter must lie in [0, 1)");
	if (subspaceSizeFor(numEigs) > EigVectsReducedNonLifted.cols())
		throw std::invalid_argument("performSubspaceIterationMethod: the hierarchy was built for " + to_string(numModesHierarchy)
			+ " eigenpairs and cannot be used for " + to_string(numEigs) + "; rebuild it with buildHierarchy()");

	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	/* Initializations */
	#ifdef VERBOSE
		std::cout << "Max itter = " << maxIter << std::endl;
	#endif

	EigValsSIMStack.resize(numEigs, maxIter);
	EigValsSIMStack.setZero();
	convergence2Stacks.resize(numEigs, maxIter);
	convergence2Stacks.setZero();
	convergence3Stacks.resize(numEigs, maxIter);
	convergence3Stacks.setZero();

	Eigen::SparseMatrix<double> MSel, SSel;
	if (isClampedBasis) {
		Eigen::SparseMatrix<double> SelM;
		constructSelectorMatrixFullSize(SelM, axis, offset);
		MSel = SelM.transpose()*M*SelM;
		SSel = SelM.transpose()*S*SelM;
	}
	else {
		MSel = M;
		SSel = S;
	}

	#ifdef VERBOSE
		std::cout << "Lifting the eigenvectors... ";
	#endif

	EigVectsReduced = Basis * EigVectsReducedNonLifted.leftCols(subspaceSizeFor(numEigs));
	#ifdef VERBOSE
		std::cout << "...successful\n";
	#endif
	/* Performing the SIM according to user selected methods */
	timingPerLevel.resize(level);
	itersPerLevel.resize(level);
	performMultiLevelSIM_withShiftOnce(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift, shiftType);


	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
		cout << "...in " << duration.count() << " seconds." << endl;

	bool isWrittenToFile = false;
	/* Write the convergence information to an external file. */
	if (isWrittenToFile) {
		string modelName = "Sphere_" + to_string(V.rows()) + "_" + to_string(F.cols());
		string fileConv = "../../Data/" + modelName + "_EigValsStack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift ? "shift_" + to_string((int)shiftType) + "_" : "") + to_string(tol) + ".txt";
		cout << fileConv << endl;
		writeDenseMatrixToFile(itersPerLevel, fileConv);
		writeDenseMatrixToFile(timingPerLevel, fileConv);
		writeDenseMatrixToFile(EigValsSIMStack, fileConv);
		string fileEigs = "../../Data/" + modelName + "_Convergence2Stack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift ? "shift_" + to_string((int)shiftType) + "_" : "") + to_string(tol) + ".txt";
		writeDenseMatrixToFile(convergence2Stacks, fileEigs);
		fileEigs = "../../Data/" + modelName + "_Convergence3Stack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift ? "shift_" + to_string((int)shiftType) + "_" : "") + to_string(tol) + ".txt";
		writeDenseMatrixToFile(convergence3Stacks, fileEigs);
	}
}

void LBEigenSolver::performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
{
	chrono::high_resolution_clock::time_point	t0, t1, t2;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();
	

	const int	q = subspaceSizeFor(numEigs);

	int iterCount = 0;
	int factCounter = 0;
	int status = 1;

	/* 4: RECURSION */
	if (k == level)			// check if this already in the lowest/coarsest subspace
	{
		XkL = EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), q);
		AkL = EigValsReduced.head(q);

	}
	// RECURSIVE for level k < max_level
	else
	{
		/* 1: RESTRICTION */
		#ifdef VERBOSE
			printf("____<1> Restriction: ");
		#endif

		t1 = chrono::high_resolution_clock::now();
		Eigen::SparseMatrix<double> SK, MK, Id;
		if (k == 0) {
			SK = SL; MK = ML;
		}
		else {
			SK = U[k - 1].transpose()*SL*U[k - 1];
			MK = U[k - 1].transpose()*ML*U[k - 1];
		}
		Id.resize(SK.rows(), SK.cols());
		Id.setIdentity();
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		#ifdef VERBOSE
			cout << "...in " << duration.count() << " seconds." << endl;
		#endif

		#ifdef VERBOSE
			printf("____<2> Going deeper => \n ################");
		#endif
		performMultiLevelSIM_withShiftOnce(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift, shiftType);

		///printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);

		/* [3] SUBSPACE ITERATION METHOD */
		t1 = chrono::high_resolution_clock::now();
		XkL = U[k] * XkL;
		#ifdef VERBOSE
			printf("____<3> Lifting, Init vectors: %dx%d ", XkL.rows(), XkL.cols());
		#endif

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		#ifdef VERBOSE
			cout << "...in " << duration.count() << " seconds." << endl;
		#endif

		/* Adding a small number on the diagonal, for robustness */
		//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;

		//printf("____<4> Embedding a small-valued diagonal matrix \n");

		/* Temporary variables */
		bool isConverged = false;
		Eigen::MatrixXd XbarK(XkL.rows(), q); XbarK.setZero();
		Eigen::MatrixXd Mproj, Sproj;
		Eigen::MatrixXd Q;
		//Eigen::VectorXd Alpha;
		Eigen::VectorXd conv2, conv3;

		int						firstNZ = 0;
		int						lastConvergedId = 0;
		if (!isClampedBasis) {
			if (F.cols() == 3) {			/* FUNCTION */
				firstNZ = 1;				// first non-zero eigenvalues on the reduced/subspace Function Laplacian
				lastConvergedId = 1;		// first non-zero eigenvalue of the expected eigenvalues
			}
		}

		/* Shifting only once, at the beginning */
		//double zhaoParam = 0.25;
		double zhaoParam = zhaoShiftParam;
		int zhaoShiftId;
		double mu = 0.0;
		double eps = 1e3*std::numeric_limits<double>::epsilon();

		#ifdef VERBOSE
			printf("____<5> Subspace iteration.\n");
		#endif

		do {
			t1 = chrono::high_resolution_clock::now();
			if (status != 0)
			{
				/* [4] Zhao's proposal */
				//zhaoShiftId = lastConvergedId + min(5, (int)floor(zhaoParam*(numEigs - lastConvergedId)));
				zhaoShiftId = lastConvergedId + (int)floor(zhaoParam*(numEigs - lastConvergedId));
				if (factCounter == 0) {
					if (tol < 1e-9 && k == 0) {
						#ifdef VERBOSE	
							std::cout << "__On finest level, no shift on really high accuracy!" << std::endl;
						#endif
						mu = 0.0;								// no spectral shift, only a tiny regularisation
						factorization.compute(SK + eps*Id);
					}
					else {
						mu = AkL(zhaoShiftId);
						#ifdef VERBOSE	
							std::cout << "___Zhao's aggressive shift proposal | shifting on index " << zhaoShiftId; // << std::endl;
						#endif
						factorization.compute(SK - mu*MK);
					}
				}
				else {
					eps *= 10;
					mu = eps;
					factorization.compute(SK + mu*Id);
				}

				if (factorization.info() != Eigen::Success)
					throw std::runtime_error("performSubspaceIterationMethod: sparse LDLT factorization failed on level " + to_string(k));

				factCounter++;
				#ifdef VERBOSE
					std::cout << "__> Factorization " << factCounter << " with shift of " << mu; // << std::endl;
				#endif
			}
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			#ifdef VERBOSE
				cout << "...in " << duration.count() << " seconds." << endl;
			#endif
			/* Stage III: Iterations */
			#ifdef VERBOSE
				printf("Iteration: %d \n", iterCount + 1);
			#endif
			#ifdef VERBOSE
				std::cout << "Solving each RHS ";
			#endif
			t1 = chrono::high_resolution_clock::now();
			for (int j = 0; j < q; ++j) {						// easily parallelizeable
				Eigen::VectorXd v;
				v = MK * (XkL.col(j));
				XbarK.col(j) = factorization.solve(v);
			}

			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			#ifdef VERBOSE
				cout << "...in " << duration.count() << " seconds." << endl;
			#endif
			/* Projection of M and K*/
			t1 = chrono::high_resolution_clock::now();
			#ifdef VERBOSE
				printf("________setting up reduced matrix M and S. ");
			#endif
			Sproj = XbarK.transpose()*SK*XbarK;
			Mproj = XbarK.transpose()*MK*XbarK;
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			#ifdef VERBOSE
				cout << "...in " << duration.count() << " seconds." << endl;
			#endif
			t1 = chrono::high_resolution_clock::now();
			#ifdef VERBOSE
				printf("________solve reduced eigenvalue problem. ");
			#endif
			status = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
			//computeEigenSpectraDense(Sproj, Mproj, q, Q, AkL);

			#ifdef VERBOSE
				cout << "__STATUS: " << status << endl;
			#endif

			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			#ifdef VERBOSE
				cout << "...in " << duration.count() << " seconds." << endl;
			#endif

			t1 = chrono::high_resolution_clock::now();
			#ifdef VERBOSE
				printf("________lifting up/prolongation.");
			#endif
			XkL = XbarK*Q;
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			#ifdef VERBOSE
				cout << "...in " << duration.count() << " seconds." << endl;
			#endif
			/* Convergence Check*
			** Type 2: */
			checkConvergenceCat2(XkL, AkL, MK, SK, conv2);
			//cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;

			/** Type 3: */
			checkConvergenceCat3(Q, AkL, isWithShift, mu, conv3);
			///cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

			#ifdef VERBOSE
				cout << "_____ [" << iterCount + 1 << "] convergence2 check:" << conv2(1) << " " << conv2(numEigs / 2) << " " << conv2(numEigs - 1) << endl;			
				cout << "_____ [" << iterCount + 1 << "] convergence3 check:" << conv3(1) << " " << conv3(numEigs / 2) << " " << conv3(numEigs - 1) << endl;
			#endif

			///cout << AkL.head(numEigs).transpose() << endl;
			#ifdef VERBOSE
				std::cout << "Stacking is ....";
				printf("Eigenvalues %dx%d \n", EigValsSIMStack.rows(), EigValsSIMStack.cols());
				printf("Convergence %dx%d \n", convergence2Stacks.rows(), convergence2Stacks.cols());
				printf("Convergence %dx%d \n", convergence2Stacks.rows(), convergence2Stacks.cols());
			#endif

			if (k == 0) {
				EigValsSIMStack.col(iterCount) = AkL.head(numEigs);
				convergence2Stacks.col(iterCount) = conv2.head(numEigs);
				convergence3Stacks.col(iterCount) = conv3.head(numEigs);
			}
			#ifdef VERBOSE
				std::cout << "... successful\n";
			#endif

			iterCount++;

			/* Ignore the first iteration */
			if (iterCount >= maxIter || (conv2(numEigs - 1) < tol)) {
				isConverged = true;
			}

			if (status != 0) {
				std::cout << "Cannot solve reduced eigenproblem \n";
				continue;
			}

			/* Check convergence */
			for (int i = lastConvergedId; i < numEigs; ++i) {
				///cout << "Checking convergence of ID " << i << ": " << conv2(i) << endl;
				if (conv2(i) < tol) {
					lastConvergedId = i;
				}
				else break;
			}

			#ifdef VERBOSE
				std::cout << "Last convergence =" << lastConvergedId << " \n";
			#endif

			if ((iterCount<maxIter) && (lastConvergedId < (numEigs - 1)))
				isConverged = false;

			#ifdef VERBOSE
				cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl;
			#endif
		} while (!isConverged);

		#ifdef VERBOSE
			printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());
		#endif
		/* Print some data when it's on the finest level */
		if (k == 0) {
			#ifdef VERBOSE
				cout << "____<7> Storing the convergence data. \n";
			#endif
			convergence2Stacks.conservativeResize(Eigen::NoChange, iterCount);
			convergence3Stacks.conservativeResize(Eigen::NoChange, iterCount);
			EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
		}
	}

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	
	if(k==level)
		printf("____ level:%d ", k);
	else 
		printf("____ level:%d and below", k);
	cout << "...in " << duration.count() << " seconds." << endl;
	

	if (k < level) {
		timingPerLevel(k) = duration.count();
		itersPerLevel(k) = iterCount;
	}
}

void LBEigenSolver::checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const
{
	convRate.resize(A.rows());
	Eigen::VectorXd Kq, AMq, diff;

	for (int i = 0; i < A.rows(); i++) {
		Kq = SK * Q.col(i);
		AMq = A(i) * MK * Q.col(i);
		diff = Kq - AMq;

		double n1 = diff.transpose() * MK * diff;
		double n2 = Kq.transpose() * MK * Kq;
		convRate(i) = sqrt(n1 / n2);
	}
	//printf("eigVals: %d | S: %dx%d | M: %dx%d \n", A.rows(), SK.rows(), SK.cols(), MK.rows(), MK.cols());
}

void LBEigenSolver::checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, bool isWithShift, double mu, Eigen::VectorXd& convRate) const
{
	convRate.resize(A.rows());

	for (int i = 0; i < A.rows(); ++i)
	{
		double a2 = A(i) * A(i);
		double qTq = Q.col(i).transpose() * Q.col(i);
		if (isWithShift) qTq += mu * (2 * A(i) - mu);
		double rat = a2 / qTq;
		double conv = abs(1.0 - rat);
		convRate(i) = sqrt(conv);
	}
}


/* Applications */
void LBEigenSolver::constructDiffusionTensor(const int numEigs, double t, const Eigen::MatrixXd &EigVects, const Eigen::VectorXd& EigVals, std::vector<Eigen::VectorXd> &DiffusionTensor)
{
	DiffusionTensor.resize(V.rows());
	for (int i = 0; i < V.rows(); ++i) {
		Eigen::VectorXd diff(numEigs);
		for (int j = 0; j < numEigs; ++j) {
			diff(j) = exp(-1.0*EigVals(j)*t/2.0) * EigVects(i, j);
		}
		DiffusionTensor[i] = diff;
	}
}

/* Visualization Tools*/
void LBEigenSolver::visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer& viewer, int k) const
{
	Eigen::MatrixXd VColor;
	igl::jet(geoDist[k], true, VColor);
	viewer.data().set_colors(VColor);
}

void LBEigenSolver::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k,  double scale) const
{
	Eigen::MatrixXd VColor;
	Eigen::VectorXd funcCols;
	if (k < 0 || k >= level) return;
	
	if(k==0) funcCols = Basis.col(id);
	else if (k < level) {
		Eigen::SparseMatrix<double> Id(S.rows(), S.cols());
		Id.setIdentity();
		Eigen::SparseMatrix<double> TempBasis = Id;
		for (int i = 0; i < k; ++i) {
			TempBasis = TempBasis*U[i];
		}
		TempBasis = TempBasis*U[k].col(id);

		funcCols = TempBasis;
	}
	igl::jet(funcCols, true, VColor);
	viewer.data().set_colors(VColor);
}

void LBEigenSolver::visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale) const
{
	cout << "Visualizing the eigenvector of id " << id << endl; 
	Eigen::VectorXd vColor;
	Eigen::MatrixXd VColor;
	//printf("Visual: Pos: %dx%d |vector: %d | ID: %d \n", viewer.data().V.rows(), viewer.data().V.cols(), EigVect.rows(), eigID);

	switch (eigType)
	{
	case EigenVectorType::EIGEN_REDUCED:
		vColor = EigVectsReduced.col(id);		
		break;
	case EigenVectorType::EIGEN_SIM:
		vColor = EigVectsSIM.col(id);
		break;

	default:
		break;
	}

	igl::jet(vColor, id > 0, VColor);
	viewer.data().set_colors(VColor);
}

/* Some getters */
Eigen::MatrixXd LBEigenSolver::getV() const { return V; }
Eigen::MatrixXi LBEigenSolver::getF() const { return F; }
void LBEigenSolver::setV(const Eigen::MatrixXd& Vin) { V = Vin; }
void LBEigenSolver::setF(const Eigen::MatrixXi& Fin) { F = Fin; }
Eigen::SparseMatrix<double> LBEigenSolver::getStiffnessMatrix() const { return S; }
Eigen::SparseMatrix<double> LBEigenSolver::getMassMatrix() const { return M; }
Eigen::SparseMatrix<double> LBEigenSolver::getBasisMatrix() const { return Basis; }
bool LBEigenSolver::getClampBasis() const { return isClampedBasis; }
void LBEigenSolver::setClampBasis(bool isClamped) { isClampedBasis = isClamped; }
int LBEigenSolver::getLevel() const { return level; }
int LBEigenSolver::getSampleSize() const { return sampleSize; }
const Eigen::VectorXi& LBEigenSolver::getSample() const { return Sample; }
const vector<int>& LBEigenSolver::getSubspaceSize() const { return SubspaceSize; }

const Eigen::MatrixXd& LBEigenSolver::getEigenVectorsReduced() const { return EigVectsReducedNonLifted; }
Eigen::VectorXd LBEigenSolver::getEigenVectorsReduced(int id) const
{
	if (id < 0 || id >= EigVectsReduced.cols()) throw std::out_of_range("getEigenVectorsReduced: index " + to_string(id) + " out of range");
	return EigVectsReduced.col(id);
}
const Eigen::VectorXd& LBEigenSolver::getEigenValuesReduced() const { return EigValsReduced; }
double LBEigenSolver::getEigenValueReduced(int id) const
{
	if (id < 0 || id >= EigValsReduced.rows()) throw std::out_of_range("getEigenValueReduced: index " + to_string(id) + " out of range");
	return EigValsReduced(id);
}
const Eigen::MatrixXd& LBEigenSolver::getEigenVectors() const { return EigVectsSIM; }
Eigen::VectorXd LBEigenSolver::getEigenVectors(int id) const
{
	if (id < 0 || id >= EigVectsSIM.cols()) throw std::out_of_range("getEigenVectors: index " + to_string(id) + " out of range");
	return EigVectsSIM.col(id);
}
const Eigen::VectorXd& LBEigenSolver::getEigenValues() const { return EigValsSIM; }
double LBEigenSolver::getEigenValueSIM(int id) const
{
	if (id < 0 || id >= EigValsSIM.rows()) throw std::out_of_range("getEigenValueSIM: index " + to_string(id) + " out of range");
	return EigValsSIM(id);
}
}
