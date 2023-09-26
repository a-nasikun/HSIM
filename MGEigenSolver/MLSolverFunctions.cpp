#include "MLSolverFunctions.h"

#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/invert_diag.h>
#include <igl/avg_edge_length.h>
#include <igl/jet.h>

//#include <Eigen/PardisoSupport>

#include <queue>

/**
*  @brief Reading a (triangle) mesh specified by the input.
*
*  @details
*   Reading a (triangle) mesh specified by the input 'meshFile',
*	currently it supports 'obj' and 'off' file format only
*  @param meshFile  Location of the mesh
*/
void MLSolverFunctions::readMesh(const string& meshFile)
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
void MLSolverFunctions::setInitialSystem(int n)
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
void MLSolverFunctions::initiateAdM()
{
	cout << "> Creating neighborhood structure of the mesh. \n";
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

void MLSolverFunctions::computeArea()
{
	/* Constructing Average length of the edges */
	avgEdgeLength = igl::avg_edge_length(V, F);

	Eigen::VectorXd dArea;
	igl::doublearea(V, F, dArea);	
	surfaceArea = dArea.sum() / 2.0;

	//surfaceArea = 0.0;
	//for (int i = 0; i < F.rows(); i++){
	//	surfaceArea += dArea(i) / 2.0;
	//}
}

void MLSolverFunctions::constructMassMatrix()
{
	cout << "> Creating mass matrix for the surface mesh \n";
	/* Construct Mass matrix M*/
	//igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
	igl::invert_diag(M, Minv);

	Eigen::VectorXd dblArea;
	igl::doublearea(V, F, dblArea);
	surfaceArea = dblArea.sum();
}

void MLSolverFunctions::constructStiffnessMatrix()
{
	cout << "> Creating stiffness matrix (Cotangent Laplacian). \n";
	igl::cotmatrix(V, F, S);
	S = -S;
}


void MLSolverFunctions::setSamplingLevels(Eigen::VectorXd& sampSize, int numEigen, double levelRatio)
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

void MLSolverFunctions::constructSampleHierarchy(const Eigen::VectorXd& samples)
{
	printf("____Sampling on %d levels \n", samples.size());
	for (int i = 0; i < samples.size(); ++i) {
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

void MLSolverFunctions::constructBasis(double support, int axis, double offset)
{
	cout << "> Constructing multi-level basis \n";
	for (int i = 0; i < level; i++) {
		maxNeighDist[i] = sqrt((surfaceArea*support) / (M_PI*SubspaceSize[i]));
		printf("Radius %d is %.4f (avg edge: %.4f)\n", i, maxNeighDist[i], avgEdgeLength);
	}

	for (int i = 0; i < level; i++) {
		constructRadialBasisFunctions(i, axis, isClampedBasis, offset, support);
		printf("basis level %d: done \n", i);
	}

	formPartitionOfUnityMultiLevel();

	Basis.resize(V.rows(), V.rows());
	Basis.setIdentity();
	for (int i = 0; i < level; i++)
		Basis = Basis * U[i];

	printf("Basis sparsity:  %.4f (%d/%d) \n", (double)Basis.nonZeros() / (double)Basis.rows(), Basis.nonZeros(), Basis.rows());
}

void MLSolverFunctions::constructBasisCoarseOnly(double support, int axis, double offset)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "___Basis construction (Coarse only)... ";
	t1 = chrono::high_resolution_clock::now();

	int coarseSample = SubspaceSize[level - 1];
	double				  maxDist;
	maxDist = sqrt((support*surfaceArea) / (M_PI*coarseSample));
	printf("Sample: %d with dist of %.5f (surface area: %.5f) \n", coarseSample, maxDist, surfaceArea);

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef = 2.0 / (pow(maxDist, 3));
	double			bCoef = -3.0 / (pow(maxDist, 2));
	int				i = 0;

	const int		NUM_OF_THREADS = omp_get_num_procs();
	omp_set_num_threads(NUM_OF_THREADS);
	UTriplet.resize(NUM_OF_THREADS);

	int tid, ntids, ipts, istart, iproc;

	vector<int> elementCounter;
	elementCounter.resize(NUM_OF_THREADS);
	int totalElementInsertions = 0;

	/* Manually break the chunck for parallelization */
#pragma omp parallel private(tid,ntids,ipts,istart,i)
	{
		iproc = omp_get_num_procs();
		tid = omp_get_thread_num();
		ntids = omp_get_num_threads();
		ipts = (int)ceil(1.00*(double)coarseSample / (double)ntids);
		istart = tid * ipts;
		if (tid == ntids - 1) ipts = coarseSample - istart;
		if (ipts <= 0) ipts = 0;

		// Instantiate D once, used for all Dijkstra
		Eigen::VectorXd				D(V.rows());
		D.setConstant(numeric_limits<double>::infinity());
		//for (int i = 0; i < V.rows(); i++) {
		//	D(i) = numeric_limits<double>::infinity();
		//}

		UTriplet[tid].reserve(3 * 2 * round((double)ipts / (double)coarseSample) * support * V.rows());

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= coarseSample) break;
			/* Dijkstra basis function */
			//ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist[0], i, SubspaceSize[0], 0, isClamped, axis, offset, UTriplet[tid], elementCounter[tid]);
			constructDijkstraDisk(Sample(i), D, aCoef, bCoef, maxDist, i, 0, axis, offset, UTriplet[tid], elementCounter[tid]);
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

	Basis.resize(V.rows(), coarseSample);
	Basis.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	

	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", Basis.rows(), Basis.cols(), (double)(Basis.nonZeros() / (double)Basis.rows()));

	formPartitionOfUnity();
	
	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}

void MLSolverFunctions::constructBasisNonSmooth(double support, int axis, double offset)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "[1] Basis (non-smooth) construction \n" ;
	t1 = chrono::high_resolution_clock::now();

	Eigen::VectorXd globToLocMap;
	double xBound;

	if (isClampedBasis) {
		double xMin = V.col(axis).minCoeff();
		double xMax = V.col(axis).maxCoeff();
		xBound = xMin + offset*(xMax - xMin);		// within this bound, the Hessian will be discarded

		globToLocMap.resize(V.rows());
		globToLocMap.setConstant(-1);

		int vCounter = 0;
		for (int i = 0; i < V.rows(); i++) {
			if (V(i, axis) >= xBound) {							// only select entries corresponding to selected vertices			
				globToLocMap(i) = vCounter;
				vCounter++;
			}
		}
	}

	clusters[0].resize(V.rows());
	for (int i = 1; i < level; ++i) 
		clusters[i].resize(SubspaceSize[i-1]);

	for (int i = 0; i < level; ++i) {
		maxNeighDist[i] = sqrt((surfaceArea*support) / (M_PI*SubspaceSize[i]));
		printf("Radius %d is %.4f (avg edge: %.4f)\n", i, maxNeighDist[i], avgEdgeLength);
	}

	for (int i = 0; i < level; i++) {
		constructRadialBasisFunctions(i, axis, isClampedBasis, offset, support);
		printf("basis level %d: done \n", i);
	}

	/* Construct clusters on each level, then create basis out of it */
	/* 1. For level 0 */
	cout << "__Clustering level 0\n";
	Eigen::SparseMatrix<double, Eigen::RowMajor> UTemp = U[0];
	for (int k = 0; k < UTemp.outerSize(); ++k) {
		double maxVal = std::numeric_limits<double>::lowest();
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(UTemp, k); it; ++it) {
			if (it.value() > maxVal)
			{
				///if (isClampedBasis && globToLocMap(it.row()) < 0) continue;
				maxVal = it.value();
				clusters[0](it.row()) = it.col();
			}
		}
	}

	/* New basis */
	cout << "__Set up new basis \n";
	vector<Eigen::Triplet<double>> UTriplet; UTriplet.reserve(V.rows());
	for (int i = 0; i < V.rows(); i++)
	{
		if (isClampedBasis && globToLocMap(i) < 0) continue;
		UTriplet.push_back(Eigen::Triplet<double>(i, clusters[0](i), 1.0));
	}
	U[0].resize(0, 0); U[0].resize(V.rows(), SubspaceSize[0]);
	U[0].setFromTriplets(UTriplet.begin(), UTriplet.end());
	UTriplet.clear(); UTriplet.shrink_to_fit();

	/* 1. For level k */
	for (int l = 1; l < level; l++)
	{
		printf("__Clustering level %d \n", l);
		UTemp = U[l];
		for (int k = 0; k < UTemp.outerSize(); ++k) {
			double maxVal = std::numeric_limits<double>::lowest();
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(UTemp, k); it; ++it) {
				if (it.value() > maxVal)
				{
					///if (isClampedBasis && globToLocMap(Sample(it.row()) < 0)) continue;
					maxVal = it.value();
					clusters[l](it.row()) = it.col();
				}
			}
		}

		cout << "__Set up new basis \n";
		UTriplet.reserve(SubspaceSize[l - 1]);
		for (int i = 0; i < SubspaceSize[l - 1]; i++)
		{
			if (isClampedBasis && globToLocMap(Sample(i)) < 0) continue;
			UTriplet.push_back(Eigen::Triplet<double>(i, clusters[l](i), 1.0));
		}
		U[l].resize(0, 0); U[l].resize(SubspaceSize[l - 1], SubspaceSize[l]);
		U[l].setFromTriplets(UTriplet.begin(), UTriplet.end());
		UTriplet.clear(); UTriplet.shrink_to_fit();
	} //  End of clustering

	  //Basis = U[0];
	Basis.resize(V.rows(), V.rows());
	Basis.setIdentity();
	for (int i = 0; i < level; i++) {
		Basis = Basis * U[i];
		printf("__Basis size: %dx%d (nnz: %d) | Loc Basis: %dx%d (nnz: %d). \n", Basis.rows(), Basis.cols(), Basis.nonZeros(), U[i].rows(), U[i].cols(), U[i].nonZeros());
	}

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}

void MLSolverFunctions::constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "___Basis construction (level " << k <<" )... ";
	t1 = chrono::high_resolution_clock::now();

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef = 2.0 / (pow(maxNeighDist[k], 3));
	double			bCoef = -3.0 / (pow(maxNeighDist[k], 2));
	int				i = 0;

	const int		NUM_OF_THREADS = omp_get_num_procs();
	omp_set_num_threads(NUM_OF_THREADS);
	UTriplet.resize(NUM_OF_THREADS);

	int tid, ntids, ipts, istart, iproc;

	vector<int> elementCounter;
	elementCounter.resize(NUM_OF_THREADS);
	int totalElementInsertions = 0;

	/* Manually break the chunck for parallelization */
#pragma omp parallel private(tid,ntids,ipts,istart,i)
	{
		iproc = omp_get_num_procs();
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
	cout << "in " << duration.count() << " seconds" << endl;

	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", U[k].rows(), U[k].cols(), (double)(U[k].nonZeros() / (double)U[k].rows()));
}

void MLSolverFunctions::constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter)
{
	priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistPQueue;

	/* Setting the bound */
	double xBound;
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

void MLSolverFunctions::formPartitionOfUnityMultiLevel()
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
			if (abs(rowSum(k)) < (10.0 * std::numeric_limits<double>::epsilon())) {
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

void MLSolverFunctions::formPartitionOfUnity()
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

void MLSolverFunctions::solveRestrictedProblem()
{
	double eps = 1e-16;
	Eigen::SparseMatrix<double> Id(Basis.cols(), Basis.cols());
	Id.setIdentity();


	cout << "> Creating the reduced system matrices.\n";
	Eigen::SparseMatrix<double> S_, M_;			//reduced stiffness and mass matrix
	S_ = Basis.transpose() * S * Basis;
	M_ = Basis.transpose() * M * Basis;

	printf("__S red %.4f (%d nnz)\n", (double)S_.nonZeros() / (double)S_.rows(), S_.nonZeros());
	//cout << S_.topLeftCorner(10, 10) << endl;
	printf("__M red %.4f (%d nnz)\n", (double)M_.nonZeros() / (double)M_.rows(), M_.nonZeros());
	//cout << M_.topLeftCorner(10, 10) << endl;

	cout << "> Computing the reduced eigenproblem (in GPU). \n";

	int status;
	int iter = 0;

	Eigen::MatrixXd SDense(S_);
	Eigen::MatrixXd MDense(M_);
	Eigen::MatrixXd IdDense; IdDense.setIdentity(S_.rows(), S_.cols());

	do {
		if (iter == 0) {
			status = computeEigenGPU(S_, M_, EigVectsReducedNonLifted, EigValsReduced);
			std::cout << "Show eigenvalues at reduced levels \n" << EigValsReduced.topRows(20).transpose() << std::endl << std::endl;
			//computeEigenGPUDense(SDense, MDense, EigVectsReducedNonLifted, EigValsReduced);
			//computeEigenSpectraDense(SDense, MDense, std::min((int)SDense.rows()/2, 2*), EigVectsReducedNonLifted, EigValsReduced);
			//computeEigenSpectraDense(SDense, MDense, 50, EigVectsReducedNonLifted, EigValsReduced);
			status = 0;
		}
		else {
			eps *= 10;
			//status = computeEigenGPU(S_+eps*Id, M_, EigVectsReducedNonLifted, EigValsReduced);
			Eigen::MatrixXd AA = SDense + eps * IdDense;
			//computeEigenSpectraDense(AA, MDense, std::min((int)SDense.rows()/2, 2*sampleSize), EigVectsReducedNonLifted, EigValsReduced);
			computeEigenSpectraDense(AA, MDense, 50,EigVectsReducedNonLifted, EigValsReduced);
			cout << "Shift: " << eps << endl;
		}
		iter++;
	} while (status != 0 && iter < 10);

	std::cout << "Succesful and lifted to fine resolution \n";
	EigVectsReduced = Basis * EigVectsReducedNonLifted;
}

void MLSolverFunctions::constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis, double offset)
{
	SelMat.resize(M.rows(), M.cols());
	vector<Eigen::Triplet<double>> STriplet;
	STriplet.reserve(M.nonZeros());

	/* Setting the bound */
	double xBound;
	if (isClampedBasis) {
		double xMin = V.col(axis).minCoeff();
		double xMax = V.col(axis).maxCoeff();
		xBound = xMin + offset*(xMax - xMin);		// within this bound, the elements will be discarded
	}

	for (int i = 0; i < V.rows(); i++) {
		if (V(i, axis) >= xBound) {							// only select entries corresponding to selected vertices
			// entries for the selector matrix			
			STriplet.push_back(Eigen::Triplet<double>(i,i, 1.0));
		}
	}

	SelMat.setFromTriplets(STriplet.begin(), STriplet.end());
	printf("__Dim of this selector matrix: %dx%d \n", SelMat.rows(), SelMat.cols());
}

/* Subspace Iteration methods */
void MLSolverFunctions::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset)
{
	cout << "> The Subspace Iteration Method \n";
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	/* Initializations */
	//EigValsSIM.resize(numEigs);
	//EigVectsSIM.resize()
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

	std::cout << "Check modes... ";
	printf("size = %dx%d \n", EigVectsReducedNonLifted.rows(), EigVectsReducedNonLifted.cols());
	std::cout << EigVectsReducedNonLifted.topLeftCorner(10, 3) << std::endl << std::endl;

	std::cout << "Lifting the eigenvectors... ";
	EigVectsReduced = Basis * EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), 2*numEigs);
	printf("size = %dx%d \n", EigVectsReduced.rows(), EigVectsReduced.cols());
	std::cout << "...successful\n";

	std::cout << "Check modes... ";
	std::cout << EigVectsReduced.topLeftCorner(10, 3) << std::endl << std::endl;

	/* Performing the SIM according to user selected methods */
	if (type == SIMType::SIM_MultiLevel) {
		timingPerLevel.resize(level);
		itersPerLevel.resize(level);
		if (isWithShift) {
			performMultiLevelSIM_withShiftOnce(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift, shiftType);
		}
		else {
			performMultiLevelSIM(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift);
		}
	}
	else {
		timingPerLevel.resize(1);
		itersPerLevel.resize(1);
		if (isWithShift) {
			performSIMwithShift_Draft(type, SSel, MSel, numEigs, maxIter, tol, isWithShift, shiftType);
		}
		else {
			performSIMwithWarmStart(type, SSel, MSel, numEigs, maxIter, tol, isWithShift);
		}
	}

	printf("Size of eigenvalues: %d | eigVectors: %dx%d \n", EigValsSIM.rows(), EigVectsSIM.rows(), EigVectsSIM.cols());

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;

	/* Write the convergence information to an external file. */
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

void MLSolverFunctions::performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift)
{
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	/* Initialization */
	cout << "____[1] SIM::Setup initialization of variables\n";
	int					q = max(2 * numEigs, numEigs + 8);
	Eigen::MatrixXd		Xk;
	int					status = 1;
	int					lastConvergedId = 1;
	double eps;
	if (F.cols() == 4) eps = 1e-2 * std::numeric_limits<double>::epsilon();
	else			eps = 1e6 * std::numeric_limits<double>::epsilon();

	/* Selecting the appropriate subspace iteration method */
	bool isNormalized = false;
	switch (type)
	{
	case SIMType::SIM_1level_WarmStart:
		cout << "_______SUBSPACE ITERATION METHOD with a Warm Start.\n";
		Xk = Basis * EigVectsReducedNonLifted.block(0, 0, Basis.cols(), q);
		break;
	case SIMType::SIM_1level_Random:
		cout << "_______SUBSPACE ITERATION METHOD with Random initialization Vectors (normalized).\n";
		generateRandomMatrix(S.rows(), q, Xk);
		if(numEigs>=200) eps = 1e16 * std::numeric_limits<double>::epsilon();

		/* M-normalized per column */
		for (int i = 0; i < q; i++) {
			double mNorm = Xk.col(i).transpose() * ML * Xk.col(i);
			mNorm = (double) 1.0 / sqrt(mNorm);
			Xk.col(i) = mNorm * Xk.col(i);
		}
		break;
	case SIMType::SIM_1level_Bathe:
		cout << "________SUBSPACE ITERATION METHOD with Bathe's proposed initialization Vectors (normalized).\n";
		generateBatheInitVectors(M.rows(), q, ML, Xk);

		/* M-normalized per column */
		if (isNormalized) {
			for (int i = 0; i < q; i++) {
				double mNorm = Xk.col(i).transpose() * ML * Xk.col(i);
				mNorm = (double) 1.0 / sqrt(mNorm);
				Xk.col(i) = mNorm * Xk.col(i);
			}
		}
		break;
	default:
		break;
	}

	/* Setting up the system + factorization */
	cout << "____[2] Factorizing the system matrix. \n";
	Eigen::SparseMatrix<double> SS;
	Eigen::SparseMatrix<double> Id;
	//double eps = 1e-6;

	cout << "Eps: " << eps << endl;
	Id.resize(S.rows(), S.cols());
	Id.setIdentity();
	Id = eps*Id;
	SS = SL + Id;		// Adding epsilon to the diagonal

	//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;
	double mu;
	if (isWithShift) {
		if (EigValsReduced.rows() < 1)			// if no information about shifting is available, use no shift
			factorization.compute(SS);
		else {
			mu = EigValsReduced((int)floor(0.4*q));
			factorization.compute(SL - mu*ML);
		}
	}
	else {
		factorization.compute(SS);
	}

	/* Iteration for eigenproblem */
	cout << "____[3] Iterating steps. \n";
	bool convergenceCheck = false;
	Eigen::MatrixXd XbarK;
	XbarK.resize(S.rows(), q);
	XbarK.setZero();

	int k = 0;
	Eigen::MatrixXd Mproj, Sproj;
	Eigen::MatrixXd Q;
	Eigen::VectorXd Alpha;
	Eigen::VectorXd conv2, conv3;

	do {
		printf("________ (%d) iteration \n", k + 1);
		std::cout << "____Last converged ID: " << lastConvergedId << std::endl;
		printf("________ setting up trial vectors Xk_bar %dx%d \n", XbarK.rows(), XbarK.cols());

		for (int i = 0; i < q; ++i) {						// easily parallelizeable
			Eigen::VectorXd v;
			if (type == SIMType::SIM_1level_Bathe && i == 0) {
				v = Xk.col(i);				// use the input as "MxXi" directly
			}
			else {
				v = ML * (Xk.col(i));
			}
			XbarK.col(i) = factorization.solve(v);
		}

		/* Projection of M and K*/
		printf("________setting up reduced matrix M and S. ");
		Sproj = XbarK.transpose()*SL*XbarK;
		Mproj = XbarK.transpose()*ML*XbarK;
		printf("________Mred: %dx%d | Sred: %dx%d \n", Mproj.rows(), Mproj.cols(), Sproj.rows(), Sproj.cols());

		/* Reduced eigenvalue problem */
		printf("________solve reduced eigenvalue problem. ");
		///status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
		computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);
		if (status != 0) {
			eps *= 10;
			factorization.compute(SL + eps*Id);
			std::cout << "___shifting by \n" << eps << std::endl;
			continue;
		}
		//computeEigenGPUDenseJacobi(Sproj, Mproj, Q, Alpha);
		//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
		//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);

		printf("________lifting up/prolongation.\n");
		Xk = XbarK*Q;

		/* Convergence Check*
		** Type 2: */
		checkConvergenceCat2(Xk, Alpha, ML, S, conv2);
		//cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
		cout << "Conv 2 \n" << conv2(1) << " " << conv2(numEigs/2) << " " << conv2(numEigs-1) << endl;

		/** Type 3: */
		checkConvergenceCat3(Q, Alpha, isWithShift, mu, conv3);
		//cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;
		cout << "Conv 3 \n" << conv3(1) << " " << conv3(numEigs/2) << " " << conv3(numEigs-1) << endl;

		//cout << Alpha.transpose() << endl; 		

		EigValsSIMStack.col(k) = Alpha.block(0, 0, numEigs - 1, 1);
		convergence2Stacks.col(k) = conv2.block(0, 0, numEigs - 1, 1);
		convergence3Stacks.col(k) = conv3.block(0, 0, numEigs - 1, 1);

		k++;
		//convergenceCheck = (k >= maxIter || (conv2(numEigs - 1)<tol));

		if (k >= maxIter || (conv2(numEigs - 1) < tol))
		{
			convergenceCheck = true;
		}
		/* Check convergence */
		for (int i = lastConvergedId; i < numEigs; ++i) {
			if (conv2(i) < tol) {
				///std::cout << "Checking convergence ID " << i << ": " << conv2(i) << std::endl; 
				lastConvergedId = i;
			}
			else {
				break;
			}
		}
		

		if ((k<maxIter) && (lastConvergedId < (numEigs - 1)))
			convergenceCheck = false;

	} while (!convergenceCheck);

	convergence2Stacks.conservativeResize(Eigen::NoChange, k);
	convergence3Stacks.conservativeResize(Eigen::NoChange, k);
	EigValsSIMStack.conservativeResize(Eigen::NoChange, k);

	EigVectsSIM = Xk;
	EigValsSIM = Alpha;

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;

	timingPerLevel(0) = duration.count();
	itersPerLevel(0) = k;

}

void MLSolverFunctions::performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
{	
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	/* Initialization */
	cout << "____[1] SIM::Setup initialization of variables\n";
	int					q = max(2 * numEigs, numEigs + 8);
	Eigen::MatrixXd		Xk;
	const int			Imax = 4;
	int					k = 0;
	int					factCounter = 0;
	double				eps = 1e6*std::numeric_limits<double>::epsilon();
	int					status;
	EigVectsSIM.resize(S.rows(), numEigs);
	EigValsSIM.resize(numEigs);

	/* Selecting the appropriate subspace iteration method */
	bool isNormalized = false;
	switch (type)
	{
	case SIMType::SIM_1level_WarmStart:
		cout << "_______SUBSPACE ITERATION METHOD with a Warm Start.\n";
		Xk = Basis * EigVectsReducedNonLifted.block(0, 0, Basis.cols(), q);
		cout << "Getting approximation \n";
		break;
	case SIMType::SIM_1level_Random:
		cout << "_______SUBSPACE ITERATION METHOD with Random initialization Vectors (normalized).\n";
		generateRandomMatrix(S.rows(), q, Xk);
		

		/* M-normalized per column */
		for (int i = 0; i < q; i++) {
			double mNorm = Xk.col(i).transpose() * ML * Xk.col(i);
			mNorm = (double) 1.0 / sqrt(mNorm);
			Xk.col(i) = mNorm * Xk.col(i);
		}
		break;
	case SIMType::SIM_1level_Bathe:
		cout << "________SUBSPACE ITERATION METHOD with Bathe's proposed initialization Vectors (normalized).\n";
		generateBatheInitVectors(M.rows(), q, ML, Xk);

		/* M-normalized per column */
		if (isNormalized) {
			for (int i = 0; i < q; i++) {
				double mNorm = Xk.col(i).transpose() * ML * Xk.col(i);
				mNorm = (double) 1.0 / sqrt(mNorm);
				Xk.col(i) = mNorm * Xk.col(i);
			}
		}
		break;
	default:
		break;
	}

	bool isConverged = false;
	Eigen::VectorXd Alpha; Alpha = EigValsReduced;
	double mu;
	Eigen::MatrixXd XbarK;
	XbarK.resize(S.rows(), q);
	XbarK.setZero();

	Eigen::MatrixXd Mproj, Sproj;
	Eigen::MatrixXd Q;

	Eigen::VectorXd conv2, conv3;

	int						firstNZ = 0;
	int						lastConvergedId = 0;

	if (!isClampedBasis)
	{
		if (F.cols() == 3) {			/* FUNCTION */
			firstNZ = 1;				// first non-zero eigenvalues on the reduced/subspace Function Laplacian
			lastConvergedId = 1;		// first non-zero eigenvalue of the expected eigenvalues
		}
		else if (F.cols() == 4) {		/* HESSIAN */
			firstNZ = 6;		// first non-zero eigenvalues on the reduced/subspace (Hessian)
			lastConvergedId = 6;		// first non-zero eigenvalue of the expected eigenvalues	
		}
	}
	//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;
	do {
		int dCounter = 0;
		/* Stage II: Shifting and Sturm's check */
		/* At the beginning, no shift */
		if (k == 0) {
			Eigen::SparseMatrix<double> Id;
			//double eps = 1e-8;
			//double eps = 1e10 * std::numeric_limits<double>::epsilon();
			eps *= 10;
			Id.resize(S.rows(), S.cols());
			Id.setIdentity();
			Id = eps*Id;
			factorization.compute(S + Id);
			cout << "__First iter, no shift \n";
			//mu = Alpha(firstNZ/2) ;
			//cout << "Factorizing ";
			//factorization.compute(SL - mu*ML);			
			//cout << " done\n";
		}
		/* for later factorization */
		else
		{
			double zhaoParam = zhaoShiftParam;
			int zhaoShiftId;
			/* [1] Bathe's proposal */
			switch (shiftType)
			{

			case MLSolverFunctions::ShiftType::SHIFT_BATHE:
				cout << "___Bathe's shift proposal | shifting on index " << lastConvergedId << endl;
				mu = (Alpha(lastConvergedId) + Alpha(lastConvergedId - 1)) / 2.0;
				if (mu <= (Alpha(lastConvergedId) + 0.001*Alpha(firstNZ)))			mu = Alpha(lastConvergedId) + 0.001*Alpha(firstNZ) + std::numeric_limits<double>::epsilon();
				else if (mu >= (Alpha(lastConvergedId) - 0.001*Alpha(firstNZ)))		mu = Alpha(lastConvergedId) - 0.001*Alpha(firstNZ) - std::numeric_limits<double>::epsilon();
				factorization.compute(SL - mu*ML);
				break;

			case MLSolverFunctions::ShiftType::SHIFT_ITOH_WILSON:
				/* [2] Itoh and Wilson's proposal */
				cout << "___Itoh and Wilson's shift proposal | shifting on index " << lastConvergedId << endl;
				mu = Alpha(lastConvergedId) + 0.9*(Alpha(lastConvergedId + 1) - Alpha(lastConvergedId));
				if ((Alpha(lastConvergedId + 1) - Alpha(lastConvergedId)) <= 0.001*Alpha(lastConvergedId))
					mu = 0.99*Alpha(lastConvergedId);
				factorization.compute(SL - mu*ML);
				break;

			case MLSolverFunctions::ShiftType::SHIFT_GONG:
				/* [3] Gong's proposal */
				cout << "___Gong's shift proposal | shifting on index " << lastConvergedId << endl;
				mu = (Alpha(lastConvergedId + 1) + Alpha(lastConvergedId + 2)) / 2.0;
				if (mu <= (Alpha(lastConvergedId + 1) + 0.001*Alpha(firstNZ)))			mu = Alpha(lastConvergedId + 1) + 0.001*Alpha(firstNZ) + std::numeric_limits<double>::epsilon();
				else if (mu >= (Alpha(lastConvergedId + 2) - 0.001*Alpha(firstNZ)))		mu = Alpha(lastConvergedId + 2) - 0.001*Alpha(firstNZ) - std::numeric_limits<double>::epsilon();
				factorization.compute(SL - mu*ML);
				break;

			case MLSolverFunctions::ShiftType::SHIFT_ZHAO:
				/* [4] Zhao's proposal */
				//if (k == 0) zhaoParam = 0.1; 
				zhaoShiftId = lastConvergedId + min(5, (int)floor(zhaoParam*(numEigs - lastConvergedId)));
				mu = Alpha(zhaoShiftId);
				cout << "___Zhao's aggressive shift proposal | shifting on index " << zhaoShiftId << endl;
				factorization.compute(SL - mu*ML);
				break;

			default:
				break;
			}
			//factorization.vectorD();		// for sturm's check
		}
		factCounter++;

		///cout << "Head: " << factorization.vectorD().head(10).transpose() << endl;
		///cout << "Tail: " << factorization.vectorD().tail(10).transpose() << endl;

		///cout << "Negative entries: ";
		///vector<int> negDiagStdVect; negDiagStdVect.reserve(100);
		///for (int i = 0; i < S.rows(); ++i) {
		///	if (factorization.vectorD()(i) < 0) {
		///		negDiagStdVect.push_back(i);
		///		cout << i << ": " << factorization.vectorD()(i) << " |";
		///		dCounter++;
		///	}
		///}
		///
		/////Eigen::VectorXd a(negDiagStdVect.data());
		///Eigen::VectorXi negDiags = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(negDiagStdVect.data(), negDiagStdVect.size());
		///
		/////Eigen::VectorXd negDiags(negDiagStdVect.data(), dCounter);
		///cout << "diag with negative ids: \n" << negDiags.transpose() << endl;
		/////cout << "diag with negative ids: \n" << a.transpose() << endl;
		///cout << "Counter: " << dCounter << endl; 

		cout << "__> Factorization " << factCounter << " with shift of " << ((k == 0) ? eps : mu) << endl;

		/* Stage III: Iterations */
		for (int i = 0; i < Imax; ++i) {
			printf("Iteration: %d \n", k + 1);

			for (int j = 0; j < q; ++j) {						// easily parallelizeable
				Eigen::VectorXd v;
				if (type == SIMType::SIM_1level_Bathe && j == 0) {
					v = Xk.col(j);				// use the input as "MxXi" directly
				}
				else {
					v = ML * (Xk.col(j));
				}
				XbarK.col(j) = factorization.solve(v);
			}

			/* Projection of M and K*/
			//printf("________setting up reduced matrix M and S. ");
			Sproj = XbarK.transpose()*SL*XbarK;
			Mproj = XbarK.transpose()*ML*XbarK;
			//printf("________Mred: %dx%d | Sred: %dx%d \n", Mproj.rows(), Mproj.cols(), Sproj.rows(), Sproj.cols());

			/* Reduced eigenvalue problem */
			//printf("________solve reduced eigenvalue problem. ");
			//status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
			computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);
			//computeEigenGPUDenseJacobi(Sproj, Mproj, Q, Alpha);
			//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
			//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);
			if (status != 0) {
				cout << "ERROR! Cannot compute eigenproblem \n";
				break;
			}

			//printf("________lifting up/prolongation.");
			Xk = XbarK*Q;

			/* Convergence Check*
			** Type 2: */
			checkConvergenceCat2(Xk, Alpha, ML, SL, conv2);
			cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
			/** Type 3: */
			checkConvergenceCat3(Q, Alpha, isWithShift, mu, conv3);
			cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

			cout << Alpha.head(numEigs).transpose() << endl;

			EigValsSIMStack.col(k) = Alpha.block(0, 0, numEigs - 1, 1);
			convergence2Stacks.col(k) = conv2.block(0, 0, numEigs - 1, 1);
			convergence3Stacks.col(k) = conv3.block(0, 0, numEigs - 1, 1);

			k++;
			/* kinda ignore the first iteration */
			if (k == 1) break;

			if (k >= maxIter || (conv2(numEigs - 1) < tol)) {
				isConverged = true;
				break;
			}
		}

		if (k == 0) continue;
		if (conv2.size() < 1) continue;

		/* Check convergence */
		for (int i = lastConvergedId; i < numEigs; ++i) {
			cout << "Checking convergence of ID " << i << ": " << conv2(i) << endl;
			if (conv2(i) < tol) {
				lastConvergedId = i;
			}
			else break;
		}

		if ((k<maxIter) && (lastConvergedId < (numEigs - 1)))
			isConverged = false;

		cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl;

	} while (!isConverged);

	convergence2Stacks.conservativeResize(Eigen::NoChange, k);
	convergence3Stacks.conservativeResize(Eigen::NoChange, k);
	EigValsSIMStack.conservativeResize(Eigen::NoChange, k);

	EigVectsSIM = Xk;
	EigValsSIM = Alpha;

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;

	timingPerLevel(0) = duration.count();
	itersPerLevel(0) = k;
}

void MLSolverFunctions::performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift)
{
	
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	printf("____[1] level: %d \n", k);
	int		q = max(2 * numEigs, numEigs + 8);
	double eps = 1e6 * std::numeric_limits<double>::epsilon();
	int status;
	int lastConvergedId = 1;

	/* 4: RECURSION */
	if (k == level)			// check if this already in the lowest/coarsest subspace
	{
		XkL = EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), q);
		AkL = EigValsReduced.head(q);
		//printf("____Size of the initial vectors: %dx%d \n", XkL.rows(), XkL.cols());
	}
	// RECURSIVE for level k < max_level
	else
	{
		/* 1: RESTRICTION */
		printf("____<1> Restriction: \n");
		Eigen::SparseMatrix<double> SK, MK;
		if (k == 0) {
			SK = SL; MK = ML;
		}
		else {
			SK = U[k - 1].transpose()*SL*U[k - 1];
			MK = U[k - 1].transpose()*ML*U[k - 1];
		}

		printf("____<2> Going deeper => \n ################");
		performMultiLevelSIM(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift);
		printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);

		//if (k == (level-1)) {
		///std::cout << "Level (under the hood, this number is 1 lower to the real one, since the finest level is defined as level 0): " << k << std::endl;
		///printf("Basis U: %dx%d \n", U[k].rows(), U[k].cols());
		///std::cout << "MK: \n" << MK.topLeftCorner(10, 10) << std::endl;
		///std::cout << "SK: \n" << SK.topLeftCorner(10, 10) << std::endl;
		///std::cout << "Aprox. EigVals (from coarser): " << AkL.head(numEigs).transpose() << std::endl;
		///std::cout << "Approx. EigVects (from coarser): \n" << XkL.topLeftCorner(10, 10) << std::endl;
		//}

		/* [3] SUBSPACE ITERATION METHOD */
		///std::cout << "Dimensionality check: \n";
		///printf("Basis U:%dx%d | Approx. EigVect: %dx%d \n", U[k].rows(), U[k].cols(), XkL.rows(), XkL.cols());
		XkL = U[k] * XkL;
		printf("____<3> Lifting, Init vectors: %dx%d \n", XkL.rows(), XkL.cols());

		/* Adding a small number on the diagonal, for robustness */
		Eigen::SparseMatrix<double> SS;
		Eigen::SparseMatrix<double> Id; Id.resize(SK.rows(), SK.cols()); Id.setIdentity();
		//double eps = 1e-6;

		SS = SK + eps*Id;		// Adding epsilon to the diagonal
		//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;
		factorization.compute(SS);
		//factorization.compute(SK);
		///cout << "SS: \n" << SS.topLeftCorner(12, 6) << endl;
		///printf("____<4> Embedding a small-valued diagonal matrix ");
		std::cout << eps << std::endl;

		/* Temporary variables */
		bool convergenceCheck = false;
		Eigen::MatrixXd XbarK(XkL.rows(), q); XbarK.setZero();
		Eigen::MatrixXd Mproj, Sproj;
		Eigen::MatrixXd Q;
		//Eigen::VectorXd Alpha;
		Eigen::VectorXd conv2, conv3;


		/* Iterating*/
		printf("____<4> Subspace iteration.\n");
		int iterCount = 0;
		do {
			/* Set up the iteration matrix */
			cout << "___Backward and Forward substitution \n";
			for (int i = 0; i < q; i++) {						// easily parallelizeable
				Eigen::VectorXd v;
				v = MK * (XkL.col(i));
				XbarK.col(i) = factorization.solve(v);
			}

			/* Projected matrix */
			Sproj = XbarK.transpose()*SK*XbarK;
			Mproj = XbarK.transpose()*MK*XbarK;

			/* Computing the eigenproblem in reduced space*/
			//status = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
			computeEigenSpectraDense(Sproj, Mproj, q, Q, AkL);
			if (status != 0) {
				cout << status << "Cannot resolve eigenproblem epsilon " << eps << endl;
				eps *= 10;
				SS = SK + eps*Id;
				factorization.compute(SS);
				cout << "Increase the epsilon to " << -eps << endl;
				continue;
			}

			//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
			//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);

			XkL = XbarK*Q;
			checkConvergenceCat2(XkL, AkL, MK, SK, conv2);
			cout << "_____ [" << iterCount + 1 << "] convergence2 check:" << conv2(1) << " " << conv2(numEigs/2) << " " << conv2(numEigs-1) << endl;

			checkConvergenceCat3(Q, AkL, isWithShift, -eps, conv3);
			cout << "_____ [" << iterCount + 1 << "] convergence3 check:" << conv3(1) << " " << conv3(numEigs/2) << " " << conv3(numEigs-1) << endl;


			/* Store the eigenvalues and convergence rate if it's on the finest level */
			if (k == 0) {
				EigValsSIMStack.col(iterCount) = AkL.block(0, 0, numEigs - 1, 1);
				convergence2Stacks.col(iterCount) = conv2.block(0, 0, numEigs - 1, 1);
				convergence3Stacks.col(iterCount) = conv3.block(0, 0, numEigs - 1, 1);
			}

			iterCount++;
			if (iterCount >= maxIter || (conv2(numEigs - 1) < tol))
			{
				convergenceCheck = true;
			}
			/* Check convergence */
			for (int i = lastConvergedId; i < numEigs; ++i) {
				///cout << "Checking convergence of ID " << i << ": " << conv2(i) << endl;
				if (conv2(i) < tol) {
					lastConvergedId = i;
				}
				else break;
			}

			if ((iterCount<maxIter) && (lastConvergedId < (numEigs - 1)))
				convergenceCheck = false;

		} while (!convergenceCheck);
		//printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());


		/* Print some data when it's on the finest level */
		if (k == 0) {
			cout << "____<5> Storing convergence data. \n";
			convergence2Stacks.conservativeResize(Eigen::NoChange, iterCount);
			convergence3Stacks.conservativeResize(Eigen::NoChange, iterCount);
			EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
		}
		itersPerLevel(k) = iterCount;
	}

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;
	if (k < level) {
		timingPerLevel(k) = duration.count();
	}
}

void MLSolverFunctions::performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
{
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	printf("____[1] level: %d \n", k);
	int		q = max(2 * numEigs, numEigs + 8);
	const int Imax = 4;
	int iterCount = 0;
	int factCounter = 0;
	int status = 0;

	/* 4: RECURSION */
	if (k == level)			// check if this already in the lowest/coarsest subspace
	{
		XkL = EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), q);
		AkL = EigValsReduced.head(q);
		printf("____Size of the initial vectors: %dx%d \n", XkL.rows(), XkL.cols());
	}
	// RECURSIVE for level k < max_level
	else
	{
		/* 1: RESTRICTION */
		printf("____<1> Restriction: ");
		Eigen::SparseMatrix<double> SK, MK;
		if (k == 0) {
			SK = SL; MK = ML;
		}
		else {
			SK = U[k - 1].transpose()*SL*U[k - 1];
			MK = U[k - 1].transpose()*ML*U[k - 1];
		}

		printf("____<2> Going deeper => \n ################");
		if (isWithShift)
			performMultiLevelSIM_withShift_Draft(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift, shiftType);
		else
			performMultiLevelSIM(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift);

		printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);

		/* [3] SUBSPACE ITERATION METHOD */
		XkL = U[k] * XkL;
		printf("____<3> Lifting, Init vectors: %dx%d \n", XkL.rows(), XkL.cols());

		/* Adding a small number on the diagonal, for robustness */
		//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;

		//printf("____<4> Embedding a small-valued diagonal matrix \n");

		/* Temporary variables */
		bool convergenceCheck = false;
		Eigen::MatrixXd XbarK(XkL.rows(), q); XbarK.setZero();
		Eigen::MatrixXd Mproj, Sproj;
		Eigen::MatrixXd Q;
		//Eigen::VectorXd Alpha;
		Eigen::VectorXd conv2, conv3;
		double mu = 0;
		double eps = 1e-5*std::numeric_limits<double>::epsilon();

		int						firstNZ = 0;
		int						lastConvergedId = 0;
		if (!isClampedBasis) {
			if (F.cols() == 3) {			/* FUNCTION */
				firstNZ = 1;				// first non-zero eigenvalues on the reduced/subspace Function Laplacian
				lastConvergedId = 1;		// first non-zero eigenvalue of the expected eigenvalues
			}
			else if (F.cols() == 4) {		/* HESSIAN */
				firstNZ = 6;				// first non-zero eigenvalues on the reduced/subspace (Hessian)
				lastConvergedId = 6;		// first non-zero eigenvalue of the expected eigenvalues	
			}
		}

		/* Iterating*/
		printf("____<5> Subspace iteration.\n");
		do {

			if (iterCount == 0) {
				std::cout << "MK: \n" << MK.topLeftCorner(6, 6) << std::endl;
				std::cout << "SK: \n" << SK.topLeftCorner(6, 6) << std::endl;
				if (factCounter == 0)
				{
					std::cout << "__First iter, shift to the left of approximation \n";
					Eigen::VectorXd headEigVal(6);
					headEigVal << abs(AkL(0)), abs(AkL(1)), abs(AkL(2)), abs(AkL(3)), abs(AkL(4)), abs(AkL(5));
					mu = 0.5 * headEigVal.cwiseAbs().minCoeff();
					mu = min(mu, 1e-8*std::numeric_limits<double>::epsilon());
					factorization.compute(SK - mu*MK);
					std::cout << "__mu=" << mu << std::endl;
				}
				else {
					//mu = AkL(firstNZ/2);
					//factorization.compute(SK - mu*MK);
					Eigen::SparseMatrix<double> Id;
					//double eps = 1e-10;
					eps *= 10;
					cout << "__epsilon added on first iter : " << eps << endl;
					Id.resize(SK.rows(), SK.cols());
					Id.setIdentity();
					factorization.compute(SK + eps*Id);
				}
			}
			else
			{
				/* [1] Bathe's proposal */
				double zhaoParam = 0.4;
				int zhaoShiftId;
				switch (shiftType)
				{
				case MLSolverFunctions::ShiftType::SHIFT_BATHE:
					cout << "___Bathe's shift proposal | shifting on index " << lastConvergedId << endl;
					if (iterCount == 0) {
						mu = (AkL(lastConvergedId));
						if (mu <= (AkL(lastConvergedId) + 0.001*AkL(firstNZ)))			mu = AkL(lastConvergedId) + 0.001*AkL(firstNZ) + std::numeric_limits<double>::epsilon();
						else if (mu >= (AkL(lastConvergedId) - 0.001*AkL(firstNZ)))		mu = AkL(lastConvergedId) - 0.001*AkL(firstNZ) - std::numeric_limits<double>::epsilon();
					}
					else {
						mu = (AkL(lastConvergedId) + AkL(lastConvergedId - 1)) / 2.0;
						if (mu <= (AkL(lastConvergedId) + 0.001*AkL(firstNZ)))			mu = AkL(lastConvergedId) + 0.001*AkL(firstNZ) + std::numeric_limits<double>::epsilon();
						else if (mu >= (AkL(lastConvergedId) - 0.001*AkL(firstNZ)))		mu = AkL(lastConvergedId) - 0.001*AkL(firstNZ) - std::numeric_limits<double>::epsilon();
					}
					//factorization.compute(SK - mu*MK);
					break;

				case MLSolverFunctions::ShiftType::SHIFT_ITOH_WILSON:
					/* [2] Itoh and Wilson's proposal */
					cout << "___Itoh and Wilson's shift proposal | shifting on index " << lastConvergedId << endl;
					mu = AkL(lastConvergedId) + 0.9*(AkL(lastConvergedId + 1) - AkL(lastConvergedId));
					if ((AkL(lastConvergedId + 1) - AkL(lastConvergedId)) <= 0.001*AkL(lastConvergedId))
						mu = 0.99*AkL(lastConvergedId);
					//factorization.compute(SK - mu*MK);
					break;

				case MLSolverFunctions::ShiftType::SHIFT_GONG:
					/* [3] Gong's proposal */
					cout << "___Gong's shift proposal | shifting on index " << lastConvergedId << endl;
					mu = (AkL(lastConvergedId + 1) + AkL(lastConvergedId + 2)) / 2.0;
					if (mu <= (AkL(lastConvergedId + 1) + 0.001*AkL(firstNZ)))			mu = AkL(lastConvergedId + 1) + 0.001*AkL(firstNZ) + std::numeric_limits<double>::epsilon();
					else if (mu >= (AkL(lastConvergedId + 2) - 0.001*AkL(firstNZ)))		mu = AkL(lastConvergedId + 2) - 0.001*AkL(firstNZ) - std::numeric_limits<double>::epsilon();
					//factorization.compute(SK - mu*MK);
					break;

				case MLSolverFunctions::ShiftType::SHIFT_ZHAO:
					/* [4] Zhao's proposal */

					//if (iterCount == 0) zhaoParam = 0.1;
					zhaoShiftId = lastConvergedId + min(5, (int)floor(zhaoParam*(numEigs - lastConvergedId)));
					mu = AkL(zhaoShiftId);
					cout << "___Zhao's aggressive shift proposal | shifting on index " << zhaoShiftId << endl;
					break;

				default:
					break;
				}
				//if (status != 0) mu *= 10;
				factorization.compute(SK - mu*MK);
			}
			factCounter++;
			cout << "__> Factorization " << factCounter << " with shift of " << ((iterCount == 0) ? eps : mu) << endl;

			/* Stage III: Iterations */
			for (int i = 0; i < Imax; ++i) {
				printf("Iteration: %d \n", iterCount + 1);

				for (int j = 0; j < q; ++j) {						// easily parallelizeable
					Eigen::VectorXd v;
					v = MK * (XkL.col(j));
					XbarK.col(j) = factorization.solve(v);
				}

				/* Projection of M and K*/
				printf("________setting up reduced matrix M and S. ");
				Sproj = XbarK.transpose()*SK*XbarK;
				Mproj = XbarK.transpose()*MK*XbarK;

				printf("________solve reduced eigenvalue problem. ");
				//status = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
				computeEigenSpectraDense(Sproj, Mproj, q, Q, AkL);
				cout << "__STATUS: " << status << endl;
				if (status != 0) {
					cout << "___!! COmputation of eigenvalues failed! \n";
					cout << "____Status: " << status << endl;
					i = Imax;
					continue;
					//break;
				}

				printf("________lifting up/prolongation.");
				XkL = XbarK*Q;

				/* Convergence Check*
				** Type 2: */
				checkConvergenceCat2(XkL, AkL, MK, SK, conv2);
				cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
				/** Type 3: */
				checkConvergenceCat3(Q, AkL, isWithShift, mu, conv3);
				cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

				cout << AkL.head(numEigs).transpose() << endl;

				if (k == 0) {
					EigValsSIMStack.col(iterCount) = AkL.block(0, 0, numEigs - 1, 1);
					convergence2Stacks.col(iterCount) = conv2.block(0, 0, numEigs - 1, 1);
					convergence3Stacks.col(iterCount) = conv3.block(0, 0, numEigs - 1, 1);
				}

				iterCount++;

				/* Ignore the first iteration */
				if (iterCount == 1) break;

				if (iterCount >= maxIter || (conv2(numEigs - 1) < tol)) {
					convergenceCheck = true;
					break;
				}
			}

			std::cout << "Convergence: " << convergenceCheck << endl;
			if (conv2.size() < 1) {
				std::cout << "The convergence check has not been done yet \n";
				continue;
			}
			if (status != 0) {
				std::cout << "Cannot solve reduced eigenproblem \n";
				continue;
			}

			/* Check convergence */
			for (int i = lastConvergedId; i < numEigs; ++i) {
				cout << "Checking convergence of ID " << i << ": " << conv2(i) << endl;
				if (conv2(i) < tol) {
					lastConvergedId = i;
				}
				else break;
			}

			if ((iterCount<maxIter) && (lastConvergedId < (numEigs - 1)))
				convergenceCheck = false;

			cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl;

		} while (!convergenceCheck);

		printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());


		/* Print some data when it's on the finest level */
		if (k == 0) {
			cout << "____<7> Storing the convergence data. \n";
			convergence2Stacks.conservativeResize(Eigen::NoChange, iterCount);
			convergence3Stacks.conservativeResize(Eigen::NoChange, iterCount);
			EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
		}
	}

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;
	if (k < level) {
		timingPerLevel(k) = duration.count();
		itersPerLevel(k) = iterCount;
	}
}

void MLSolverFunctions::performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
{
	std::cout << "Functions" << std::endl;

	chrono::high_resolution_clock::time_point	t0, t1, t2;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	printf("____[1] level: %d \n", k);
	int		q = max(2 * numEigs, numEigs + 8);

	int iterCount = 0;
	int factCounter = 0;
	int status = 1;

	/* 4: RECURSION */
	if (k == level)			// check if this already in the lowest/coarsest subspace
	{
		XkL = EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), q);
		AkL = EigValsReduced.head(q);
		///printf("____Size of the initial vectors: %dx%d \n", XkL.rows(), XkL.cols());

		std::cout << "Lowest level [EigenValues] \n" << AkL.topRows(10) << std::endl << std::endl;
		std::cout << "Lowest level [EigenVectors] \n" << XkL.topLeftCorner(10, 5) << std::endl << std::endl;
	}
	// RECURSIVE for level k < max_level
	else
	{
		/* 1: RESTRICTION */
		printf("____<1> Restriction: ");
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
		cout << "...in " << duration.count() << " seconds." << endl;

		std::cout << "Stiffness Matrix: Level "  << k << "\n"  << SK.topLeftCorner(10, 5) << std::endl << std::endl;


		printf("____<2> Going deeper => \n ################");
		performMultiLevelSIM_withShiftOnce(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift, shiftType);

		///printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);

		/* [3] SUBSPACE ITERATION METHOD */
		t1 = chrono::high_resolution_clock::now();
		XkL = U[k] * XkL;
		printf("____<3> Lifting, Init vectors: %dx%d ", XkL.rows(), XkL.cols());
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "...in " << duration.count() << " seconds." << endl;

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
		double mu;
		double eps = 1e3*std::numeric_limits<double>::epsilon();

		printf("____<5> Subspace iteration.\n");
		do {
			t1 = chrono::high_resolution_clock::now();
			if (status != 0)
			{
				/* [4] Zhao's proposal */
				//zhaoShiftId = lastConvergedId + min(5, (int)floor(zhaoParam*(numEigs - lastConvergedId)));
				zhaoShiftId = lastConvergedId + (int)floor(zhaoParam*(numEigs - lastConvergedId));
				if (factCounter == 0) {
					if (tol < 1e-9 && k == 0) {
						std::cout << "__On finest level, no shift on really high accuracy!" << std::endl;						
						factorization.compute(SK + eps*Id);
					}
					else {
						mu = AkL(zhaoShiftId);
						std::cout << "___Zhao's aggressive shift proposal | shifting on index " << zhaoShiftId; // << std::endl;
						factorization.compute(SK - mu*MK);
					}
				}
				else {
					eps *= 10;
					mu = eps;
					factorization.compute(SK + mu*Id);
				}

				factCounter++;
				std::cout << "__> Factorization " << factCounter << " with shift of " << mu; // << std::endl;
			}
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			cout << "...in " << duration.count() << " seconds." << endl;

			/* Stage III: Iterations */
			printf("Iteration: %d \n", iterCount + 1);

			std::cout << "Solving each RHS ";
			t1 = chrono::high_resolution_clock::now();
			for (int j = 0; j < q; ++j) {						// easily parallelizeable
				Eigen::VectorXd v;
				v = MK * (XkL.col(j));
				XbarK.col(j) = factorization.solve(v);
			}

			std::cout << "XBar: Level " << k << "\n" << XbarK.topLeftCorner(10, 5) << std::endl << std::endl;

			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			cout << "...in " << duration.count() << " seconds." << endl;

			/* Projection of M and K*/
			t1 = chrono::high_resolution_clock::now();
			printf("________setting up reduced matrix M and S. ");
			Sproj = XbarK.transpose()*SK*XbarK;
			Mproj = XbarK.transpose()*MK*XbarK;
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			cout << "...in " << duration.count() << " seconds." << endl;

			std::cout << "SProj (very small) Level " << k << "\n" << Sproj.topLeftCorner(10, 5) << std::endl << std::endl;

			t1 = chrono::high_resolution_clock::now();
			printf("________solve reduced eigenvalue problem. ");
			status = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
			//computeEigenSpectraDense(Sproj, Mproj, q, Q, AkL);
			std::cout << "Cur eigvals == " << AkL.topRows(10).transpose() << std::endl;
			std::cout << "Cur eigvects == " << Q.topLeftCorner(10,10) << std::endl;

			cout << "__STATUS: " << status << endl;
			//if (status != 0) {
			//	cout << "___!! COmputation of eigenvalues failed! \n";
			//	cout << "____Status: " << status << endl;
			//	//i = Imax;
			//	continue;
			//	//break;
			//}
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			cout << "...in " << duration.count() << " seconds." << endl;

			t1 = chrono::high_resolution_clock::now();
			printf("________lifting up/prolongation.");
			XkL = XbarK*Q;
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			cout << "...in " << duration.count() << " seconds." << endl;

			/* Convergence Check*
			** Type 2: */
			checkConvergenceCat2(XkL, AkL, MK, SK, conv2);
			cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
			/** Type 3: */
			checkConvergenceCat3(Q, AkL, isWithShift, mu, conv3);
			///cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

			cout << "_____ [" << iterCount + 1 << "] convergence2 check:" << conv2(1) << " " << conv2(numEigs / 2) << " " << conv2(numEigs - 1) << endl;
			cout << "_____ [" << iterCount + 1 << "] convergence3 check:" << conv3(1) << " " << conv3(numEigs / 2) << " " << conv3(numEigs - 1) << endl;

			///cout << AkL.head(numEigs).transpose() << endl;

			if (k == 0) {
				EigValsSIMStack.col(iterCount) = AkL.block(0, 0, numEigs - 1, 1);
				convergence2Stacks.col(iterCount) = conv2.block(0, 0, numEigs - 1, 1);
				convergence3Stacks.col(iterCount) = conv3.block(0, 0, numEigs - 1, 1);
			}

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

			if ((iterCount<maxIter) && (lastConvergedId < (numEigs - 1)))
				isConverged = false;

			cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl;

		} while (!isConverged);

		printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());

		/* Print some data when it's on the finest level */
		if (k == 0) {
			cout << "____<7> Storing the convergence data. \n";
			convergence2Stacks.conservativeResize(Eigen::NoChange, iterCount);
			convergence3Stacks.conservativeResize(Eigen::NoChange, iterCount);
			EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
		}
	}

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;
	if (k < level) {
		timingPerLevel(k) = duration.count();
		itersPerLevel(k) = iterCount;
	}
}

void MLSolverFunctions::checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const
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

void MLSolverFunctions::checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, bool isWithShift, double mu, Eigen::VectorXd& convRate) const
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

/* Reference*/
void MLSolverFunctions::computeReference(int numEigs)
{
	computeEigenSpectraSparse(S, M, numEigs, EigVectsRef, EigValsRef, 0, 1000, 1e-7);
}
/* Applications */
void MLSolverFunctions::constructDiffusionTensor(const int numEigs, double t, const Eigen::MatrixXd &EigVects, const Eigen::VectorXd& EigVals, std::vector<Eigen::VectorXd> &DiffusionTensor)
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
void MLSolverFunctions::visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer& viewer, int k) const
{
	Eigen::MatrixXd VColor;
	igl::jet(geoDist[k], true, VColor);
	viewer.data().set_colors(VColor);
}

void MLSolverFunctions::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, int k,  double scale) const
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

void MLSolverFunctions::visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale) const
{
	cout << "Visualizing the eigenvector of id " << id << endl; 
	Eigen::VectorXd vColor;
	Eigen::MatrixXd VColor;
	//printf("Visual: Pos: %dx%d |vector: %d | ID: %d \n", viewer.data().V.rows(), viewer.data().V.cols(), EigVect.rows(), eigID);

	switch (eigType)
	{
	case EigenVectorType::EIGEN_REFERENCE:
		vColor = EigVectsRef.col(id);
		break;
	case EigenVectorType::EIGEN_REDUCED:
		vColor = EigVectsReduced.col(id);
		std::cout << "eigval=" << EigValsReduced(id);
		break;
	case EigenVectorType::EIGEN_SIM:
		vColor = EigVectsSIM.col(id);
		std::cout << "eigval=" << EigValsSIM(id);
		break;

	default:
		break;
	}

	igl::jet(vColor, id > 0, VColor);
	viewer.data().set_colors(VColor);
}

void MLSolverFunctions::visualizeClusterOfSamplesLevel0(igl::opengl::glfw::Viewer& viewer)
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

void MLSolverFunctions::visualizeClusterOfSamplesLevelK(igl::opengl::glfw::Viewer& viewer, int k)
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
Eigen::MatrixXd MLSolverFunctions::getV() const { return V; }
Eigen::MatrixXi MLSolverFunctions::getF() const { return F; }
void MLSolverFunctions::setV(const Eigen::MatrixXd& Vin) { V = Vin; }
void MLSolverFunctions::setF(const Eigen::MatrixXi& Fin) { F = Fin; }
Eigen::SparseMatrix<double> MLSolverFunctions::getStiffnessMatrix() const { return S; }
Eigen::SparseMatrix<double> MLSolverFunctions::getMassMatrix() const { return M; }
Eigen::SparseMatrix<double> MLSolverFunctions::getBasisMatrix() const { return Basis; }
bool MLSolverFunctions::getClampBasis() const { return isClampedBasis; }
void MLSolverFunctions::setClampBasis(bool isClamped) { isClampedBasis = isClamped; }
int MLSolverFunctions::getLevel() const { return level; }
int MLSolverFunctions::getSampleSize() const { return sampleSize; }
const Eigen::VectorXi& MLSolverFunctions::getSample() const { return Sample; }
const vector<int>& MLSolverFunctions::getSubspaceSize() const { return SubspaceSize; }

/* Getters of the eigen-values + vectors*/
const Eigen::MatrixXd& MLSolverFunctions::getEigenVectorsRef() const { return EigVectsRef; }
const Eigen::VectorXd& MLSolverFunctions::getEigenVectorsRef(int id) const
{
	if (id >= EigVectsRef.cols())	return EigVectsRef.col(EigValsRef.rows() - 1);
	else if (id < 0)						return EigVectsRef.col(0);

	return EigVectsRef.col(id);
}
const Eigen::VectorXd& MLSolverFunctions::getEigenValuesRef() const { return EigValsRef; }
const double MLSolverFunctions::getEigenValueRef(int id) const
{
	if (id >= EigValsRef.rows())	return EigValsRef(EigValsRef.rows() - 1);
	else if (id < 0)						return EigValsRef(0);

	return EigValsRef(id);
}
const Eigen::MatrixXd& MLSolverFunctions::getEigenVectorsReduced() const { return EigVectsReducedNonLifted; }
const Eigen::VectorXd& MLSolverFunctions::getEigenVectorsReduced(int id) const
{
	if (id >= EigVectsReduced.cols())	return EigVectsReduced.col(EigValsReduced.rows() - 1);
	else if (id < 0)						return EigVectsReduced.col(0);

	return EigVectsReduced.col(id);
}
const Eigen::VectorXd& MLSolverFunctions::getEigenValuesReduced() const { return EigValsReduced; }
const double MLSolverFunctions::getEigenValueReduced(int id) const
{
	if (id >= EigValsReduced.rows())	return EigValsReduced(EigValsReduced.rows() - 1);
	else if (id < 0)						return EigValsReduced(0);

	return EigValsReduced(id);
}
const Eigen::MatrixXd& MLSolverFunctions::getEigenVectorsSIM() const { return EigVectsSIM; }
const Eigen::VectorXd& MLSolverFunctions::getEigenVectorsSIM(int id) const
{
	if (id >= EigVectsSIM.cols())		return EigVectsSIM.col(EigValsSIM.rows() - 1);
	else if (id < 0)						return EigVectsSIM.col(0);
	return EigVectsSIM.col(id);
}
const Eigen::VectorXd& MLSolverFunctions::getEigenValuesSIM() const { return EigValsSIM; }
const double MLSolverFunctions::getEigenValueSIM(int id) const
{
	if (id >= EigValsSIM.rows())		return EigValsSIM(EigValsSIM.rows() - 1);
	else if (id < 0)						return EigValsSIM(0);

	return EigValsSIM(id);
}