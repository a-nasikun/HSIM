#include "MLSolverFunctions.h"

#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/invert_diag.h>
#include <igl/avg_edge_length.h>

#include <queue>

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

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;

	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", Basis.rows(), Basis.cols(), (double)(Basis.nonZeros() / (double)Basis.rows()));

	formPartitionOfUnity();
}

void MLSolverFunctions::constructBasisNonSmooth(double support, int axis, double offset)
{
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
	cout << "Clustering level 0\n";
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
	cout << "Set up new basis \n";
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
		printf("Clustering level %d \n", l);
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

		cout << "Set up new basis \n";
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
		printf("Basis size: %dx%d (nnz: %d) | Loc Basis: %dx%d (nnz: %d). \n", Basis.rows(), Basis.cols(), Basis.nonZeros(), U[i].rows(), U[i].cols(), U[i].nonZeros());
	}
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

/* Visualization Tools*/
void MLSolverFunctions::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, double scale) const
{
	Eigen::MatrixXd VColor;
	Eigen::VectorXd funcCols = Basis.col(id);
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

