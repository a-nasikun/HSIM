#include "MLSolverFunctions.h"

#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/invert_diag.h>
#include <igl/avg_edge_length.h>
#include <igl/edges.h>

#include <Eigen/PardisoSupport>

#include <queue>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>


namespace MuLES
{
	/**
	*  @brief	Compuete the area of each triangle constituting the triangle mesh represantion.
	*/
	void MLSolverFunctions::computeArea()
	{
		/* Constructing Average length of the edges */
		avgEdgeLength = igl::avg_edge_length(V, F);

		Eigen::VectorXd dArea;
		igl::doublearea(V, F, dArea);
		surfaceArea = dArea.sum() / 2.0;
	}

	/**
	*  @brief
	*   Set-up the corresponding mass matrix M.
	*  @details
	*	For Laplace--Beltrami eigenproblem,
	*	we generally use the lumped mass matrix (i.e. diagonal mass matrix).
	*  @see	MuLES::MLSolverFunctions::constructMassMatrix()
	*/
	void MLSolverFunctions::constructMassMatrix()
	{
		constructDiagonalMassMatrix();
		//constructConsistentMassMatrix();

		std::string filename = "D:/Nasikun/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/Object" + to_string(V.rows()) +"_Mass.mat";
		//WriteSparseMatrixToMatlab(M, filename);
		
	}

	void MLSolverFunctions::constructDiagonalMassMatrix()
	{
		std::cout << "____> Creating mass matrix for the surface mesh \n";
		/* Construct Mass matrix M*/
		//igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
		igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
		igl::invert_diag(M, Minv);

		Eigen::VectorXd dblArea;
		igl::doublearea(V, F, dblArea);
		surfaceArea = dblArea.sum();
	}

	void MLSolverFunctions::constructConsistentMassMatrix()
	{
		std::cout << "____> Creating mass matrix for the surface mesh \n";
		/* Set up the edges data structure*/
		Eigen::MatrixXi E;
		igl::edges(F, E);
		Eigen::VectorXd dArea;
		igl::doublearea(V, F, dArea);
		surfaceArea = dArea.sum();

		/* Build up vertex-edge neighborhood structure */
		std::vector<std::set<int>> VENeighbors;
		VENeighbors.resize(V.rows());
		for (int i = 0; i < E.rows(); i++) {
			VENeighbors[E(i, 0)].emplace(i);
			VENeighbors[E(i, 1)].emplace(i);			
		}

		/* Build up edge-face neighborhood structure */
		vector<set<int>> EFlist;
		EFlist.resize(E.rows());
		for (int ijk = 0; ijk < F.rows(); ijk++) {
			for (int i = 0; i < F.cols(); i++) {
				int ii = F(ijk, i);
				int in = F(ijk, (i + 1) % (int)F.cols());
				int ip;
				if (i < 1)
					ip = 2;
				else
					ip = (i - 1) % (int)F.cols();

				/* Loop over all edges having element F(i,j) */
				for (set<int>::iterator ij = VENeighbors[ii].begin(); ij != VENeighbors[ii].end(); ++ij) {
					int edge = *ij;
					if ((ii == E(edge, 0) && in == E(edge, 1)) || (ii == E(edge, 1) && in == E(edge, 0)))
					{
						EFlist[edge].emplace(ijk);
					}
				}
			}
		}

		/* Some test*/
		//srand(time(NULL));
		//int v = rand()%V.rows();
		//for (int e : VENeighbors[v]) {
		//	printf("Vertex %d has edge %d (%d) (%d=>%d)as neighbor, that belongs to ", v, e, VENeighbors[v].size(), E(e,0), E(e,1));
		//	for (int f : EFlist[e]) {
		//		std::cout << " face:" << f << "(" << F(f,0) << "|" << F(f, 1) << "|" << F(f, 2) << ")\t";
		//	}
		//	std::cout<<std::endl;
		//}

		/* Set up Mass matrix:: OFF-DIAGONAL */
		M.resize(V.rows(), V.rows());
		std::vector<Eigen::Triplet<double>> MTriplet;
		MTriplet.reserve(10 * V.rows());		// aprox 7 items per row
		for (int vv = 0; vv < V.rows();++vv){
			for (int e : VENeighbors[vv]){
				double triArea = 0.0;
				for (int f : EFlist[e]) {
					triArea += abs(dArea(f)) / 2.0;
				}
				triArea /= (2*12.0);
				MTriplet.push_back(Eigen::Triplet<double>(E(e, 0), E(e, 1), triArea));
				MTriplet.push_back(Eigen::Triplet<double>(E(e, 1), E(e, 0), triArea));
			}
		}
		/* Set up Mass matrix:: DIAGONAL Entries */
		for (int f = 0; f < F.rows(); ++f) {
			for (int vv = 0; vv < F.cols(); ++vv) {
				int tv = F(f, vv);
				MTriplet.push_back(Eigen::Triplet<double>(tv, tv, dArea(f)/(2.0*6.0)));
			}
		}
		M.setFromTriplets(MTriplet.begin(), MTriplet.end());
		std::cout << "Mass matrix" << M.block(0, 0, 20, 20) << std::endl; 		
	}

	/**
	*  @brief	Set-up the corresponding stiffness matrix S.
	*  @details
	*	The widely adopted Laplacian option for discretization of Laplace--Beltrami operator in geometry processing is the cotangent weight.
	*/
	void MLSolverFunctions::constructStiffnessMatrix()
	{
		std::cout << "____> Creating stiffness matrix (Cotangent Laplacian). \n";
		igl::cotmatrix(V, F, S);
		S = -S;
	}

	/**
	*  @brief	Construct a multilevel basis for every level on each of the hierarchy.
	*  @details
	*   The basis is constructed using (1) farthest point sampling, followed by (2) region growing strategy.
	*  @param	support The expected average overlaps of disks on every degree of freedom on each level.
	*  @param	axis	When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset	When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*/
	void MLSolverFunctions::constructBasis(double support, int axis, double offset)
	{
		std::cout << "____Constructing multi-level basis \n";
		for (int i = 0; i < level; i++) {
			maxNeighDist[i] = sqrt((surfaceArea*support) / (M_PI*SubspaceSize[i]));
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

		printf("____> Basis sparsity:  %.4f (%d/%d) \n", (double)Basis.nonZeros() / (double)Basis.rows(), Basis.nonZeros(), Basis.rows());
	}

	/**
	*  @brief	Construct the basis for 2-level hierarchical system
	*  @details In this set-up only 1 basis (restriction and prolongation) matrix is required.
	*	In general, we take the coarsest level DoFs for this construction.
	*  @param	support The expected average overlaps of disks on every degree of freedom on each level.
	*  @param	axis	When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset	When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*/
	void MLSolverFunctions::constructBasisCoarseOnly(double support, int axis, double offset)
	{
		// For Timing
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;
		std::cout << "___Basis construction (Coarse only)... ";
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
				constructDijkstraDisk(Sample(i), D, aCoef, bCoef, maxDist, i, 0, axis, offset, UTriplet[tid]);
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
		std::cout << "in " << duration.count() << " seconds" << endl;
	}

	/**
	*  @details
	*   Our implementation of Xian's piecewise constant weight function for hierarchical problem.
	*	Originally this method is used for fast linear solver used in deformation of elastics body.
	*  @param	support		The expected average overlaps of disks on every degree of freedom on each level.
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*/
	void MLSolverFunctions::constructBasisNonSmooth(double support, int axis, double offset)
	{
		// For Timing
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;
		std::cout << "[1] Basis (non-smooth) construction \n";
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
			clusters[i].resize(SubspaceSize[i - 1]);

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
					//if (isClampedBasis && globToLocMap(it.row()) < 0) continue;
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
						//if (isClampedBasis && globToLocMap(Sample(it.row()) < 0)) continue;
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

	/**
	*  @brief
	*   Build a radial basis functions (i.e. geodesic disks) on every samples for every level on each of the hierarchy.
	*  @param	k			The level that RBFs needs to be constructed.
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	isClamped	Tells whether clamping is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*  @param	support		The expected average overlaps of disks on every degree of freedom on each level.
	*/
	void MLSolverFunctions::constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support)
	{
		// For Timing
		chrono::high_resolution_clock::time_point	t1, t2;
		chrono::duration<double>					duration;
		cout << "____> Basis construction (level " << k << " )... ";
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

			if (k == 0)
				UTriplet[tid].reserve(2 * round((double)ipts / (double)SubspaceSize[0]) * support * V.rows());
			else
				UTriplet[tid].reserve(2 * round((double)ipts / (double)SubspaceSize[0]) * support * SubspaceSize[k - 1]);

			for (i = istart; i < (istart + ipts); i++) {
				if (i >= SubspaceSize[k]) break;
				/* Dijkstra basis function */
				//ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist[0], i, SubspaceSize[0], 0, isClamped, axis, offset, UTriplet[tid], elementCounter[tid]);
				constructDijkstraDisk(Sample(i), D, aCoef, bCoef, maxNeighDist[k], i, k, axis, offset, UTriplet[tid]);
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

		if (k == 0)
			U[0].resize(V.rows(), SubspaceSize[k]);
		else
			U[k].resize(SubspaceSize[k - 1], SubspaceSize[k]);

		U[k].setFromTriplets(AllTriplet.begin(), AllTriplet.end());

		// timing
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "in " << duration.count() << " seconds, ";
		printf("(%dx%d), with %.2f average support.\n", U[k].rows(), U[k].cols(), (double)(U[k].nonZeros() / (double)U[k].rows()));
	}

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
	void MLSolverFunctions::constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet) 
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
					//double val = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;					// face

					/* Linear */
					double val = (maxDist-aVal) / maxDist;
					UTriplet.push_back(Eigen::Triplet<double>(i, col, val));
				}
			}
		}
		else {
			for (int i = 0; i < SubspaceSize[k - 1]; ++i) {
				double aVal = D(Sample(i));
				if (aVal <= maxDist) {
					if (isClampedBasis && V(Sample(i), axis) < xBound) continue;

					/* Polynomial mapping (3rd Order) */
					//double val = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;
					
					/* Linear */
					double val = (maxDist - aVal) / maxDist;
					UTriplet.push_back(Eigen::Triplet<double>(i, col, val));

				}
			}
		}
		D.setConstant(numeric_limits<double>::infinity());
	}

	/**
	*  @brief
	*   Construct a selector matrix for clamped eigen problem.
	*  @param	SelMat		The desired selector matrix
	*  @param	axis		When clamping of basis is used (by default not), it specifies which axis direction is used.
	*  @param	offset		When clamping of basis is used (by default not), it determines the ratio w.r.t selected axis.
	*/
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
				STriplet.push_back(Eigen::Triplet<double>(i, i, 1.0));
			}
		}

		SelMat.setFromTriplets(STriplet.begin(), STriplet.end());
		printf("__Dim of this selector matrix: %dx%d \n", SelMat.rows(), SelMat.cols());
	}


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
		//convergence3Stacks.resize(numEigs, maxIter);
		//convergence3Stacks.setZero();

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

		//std::cout << "Lifting the eigenvectors... ";
		//EigVectsReduced = Basis * EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), 2 * numEigs);
		//std::cout << "...successful\n";

		/* Performing the SIM according to user selected methods */
		if (type == SIMType::SIM_MultiLevel) {
			timingPerLevel.resize(level+1);
			itersPerLevel.resize(level+1);
			if (isWithShift) {
				std::cout << "DISCARD CONVERGED EIGEN \n";
				//performMultiLevelSIM_withShiftOnce(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift, shiftType);
				//performMultiLevelSIM_withShiftOnceConverged(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift, shiftType);
				performMultiLevelSIM_withShiftOnceConverged(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, true, shiftType);
			}
			else {
				std::cout << "USE CONVERGED EIGEN \n";
				performMultiLevelSIM(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift);
				//performMultiLevelSIM_withShiftOnce(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift, shiftType);
				//performMultiLevelSIM_withShiftOnce(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, true, shiftType);
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

		EigVectsSIM.conservativeResize(Eigen::NoChange, numEigs);
		EigValsSIM.conservativeResize(numEigs);
		printf("Size of eigenvalues: %d | eigVectors: %dx%d \n", EigValsSIM.rows(), EigVectsSIM.rows(), EigVectsSIM.cols());

		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t0;
		cout << "...in " << duration.count() << " seconds." << endl;

		// Sturm's check
		sturmCheck(S, M, EigValsSIM);

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
	void MLSolverFunctions::performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift)
	{
		chrono::high_resolution_clock::time_point	t00, t0, t1;
		chrono::duration<double>					duration;
		t00 = chrono::high_resolution_clock::now();
		t0 = chrono::high_resolution_clock::now();

		/* Initialization */
		cout << "____[1] SIM::Setup initialization of variables\n";
		int					q = max(2 * numEigs, numEigs + 8);
		Eigen::MatrixXd		Xk;
		int					status = 1;
		int					lastConvergedId = 1;
		double eps;
		if (F.cols() == 4) eps = 1e-2 * std::numeric_limits<double>::epsilon();
		//else			eps = 1e7 * std::numeric_limits<double>::epsilon();
		else			eps = 1e10 * std::numeric_limits<double>::epsilon();

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
			//if (numEigs >= 200) eps = 1e16 * std::numeric_limits<double>::epsilon();

			/* M-normalized per column */
			for (int i = 0; i < q; i++) {
				double mNorm = Xk.col(i).transpose() * ML * Xk.col(i);
				mNorm = (double) 1.0 / sqrt(mNorm);
				Xk.col(i) = mNorm * Xk.col(i);
			}
			break;
		case SIMType::SIM_1level_Bathe:
			cout << "________SUBSPACE ITERATION METHOD with Bathe's proposed initialization Vectors (normalized).\n";
			generateBatheInitVectors(M.rows(), q, SL, ML, Xk);
			//generateBatheInitVectors(M.rows(), q, ML, Xk);

			/* M-normalized per column */
			//if (isNormalized) {
				for (int i = 0; i < q; i++) {
					double mNorm = Xk.col(i).transpose() * ML * Xk.col(i);
					mNorm = (double) 1.0 / sqrt(mNorm);
					Xk.col(i) = mNorm * Xk.col(i);
				}
			//}
			break;
		default:
			break;
		}
		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t0;
		cout << "...in " << duration.count() << " seconds." << endl;

		/* Setting up the system + factorization */
		cout << "____[2] Factorizing the system matrix. \n";
		t0 = chrono::high_resolution_clock::now();

		Eigen::SparseMatrix<double> SS;
		Eigen::SparseMatrix<double> Id;
		//double eps = 1e-6;


		cout << "Eps: " << eps << endl;
		Id.resize(S.rows(), S.cols());
		Id.setIdentity();
		//Id = eps*Id;
		SS = SL + eps*Id;		// Adding epsilon to the diagonal
		//SS = SL;

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
		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t0;
		cout << "...in " << duration.count() << " seconds." << endl;

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
		Eigen::VectorXd conv2(numEigs), conv3(numEigs);
		conv2.setConstant(std::numeric_limits<double>::max());
		conv3.setConstant(std::numeric_limits<double>::max());

		do {
			printf("________ (%d) iteration \n", k + 1);
			std::cout << "____Last converged ID: " << lastConvergedId << std::endl;
			printf("________ setting up trial vectors Xk_bar %dx%d \n", XbarK.rows(), XbarK.cols());

			t0 = chrono::high_resolution_clock::now();

			if (!isParallelSolverEnabled) {
				std::cout << " NOT PARALLELL! \n";
				int solveCounter = 0;
				Eigen::VectorXd v;
				for (int j = 0; j < q; ++j) {						// easily parallelizeable				
					if (isConvergedVectsNotUsed && (conv2(j) < tol && j > 0)) continue;
					v = ML * (Xk.col(j));
					XbarK.col(j) = factorization.solve(v);
					solveCounter++;
				}
			}
			else
			{
				std::cout << " PARALLELL! \n";
				//const int		NUM_OF_THREADS = omp_get_num_procs();
				const int		NUM_OF_THREADS = std::min(q, omp_get_num_procs());
				cout << "We are using " << NUM_OF_THREADS << " threads for OpenMP\n";
				//const int		NUM_OF_THREADS = 2;
				omp_set_num_threads(NUM_OF_THREADS);
				int tid, ntids, ipts, istart, iproc;
				int iCol = 0;
				#pragma omp parallel private(tid,ntids,ipts,istart,iCol)
				{
					iproc = omp_get_num_procs();
					tid = omp_get_thread_num();
					ntids = omp_get_num_threads();
					//ipts = (int)ceil((double) q / (double)ntids);
					//ipts = q / ntids;
					ipts = (q + ntids - 1) / ntids;
					istart = tid * ipts;
					if (tid == ntids - 1) ipts = q - istart;
					if (ipts <= 0) ipts = 0;

					Eigen::VectorXd v;

					for (iCol = istart; iCol < (istart + ipts); ++iCol) {
						if (iCol >= q) continue;
						if (isConvergedVectsNotUsed && conv2(iCol) < tol && iCol > 0) continue;
						v = ML * (Xk.col(iCol));
						XbarK.col(iCol) = factorization.solve(v);
					}
				}
			}
			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << "...in " << duration.count() << " seconds." << endl;

			//Eigen::MatrixXd XXX;
			//performSIMIteration(SL, ML, XbarK, XXX, Alpha);

			/* Projection of M and K*/
			t0 = chrono::high_resolution_clock::now();
			printf("________setting up reduced matrix M and S. ");
			Sproj = XbarK.transpose()*SL*XbarK;
			Mproj = XbarK.transpose()*ML*XbarK;
			printf("________Mred: %dx%d | Sred: %dx%d \n", Mproj.rows(), Mproj.cols(), Sproj.rows(), Sproj.cols());
			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << "...in " << duration.count() << " seconds." << endl;

			/* Reduced eigenvalue problem */
			printf("________solve reduced eigenvalue problem. ");
			t0 = chrono::high_resolution_clock::now();
			status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
			if (status != 0) {
				eps *= 10;
				factorization.compute(SL + eps*Id);
				std::cout << "___shifting by \n" << eps << std::endl;
				continue;
			}
			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << "...in " << duration.count() << " seconds." << endl;
			//computeEigenGPUDenseJacobi(Sproj, Mproj, Q, Alpha);
			//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
			//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);

			printf("________lifting up/prolongation.\n");
			t0 = chrono::high_resolution_clock::now();
			Xk = XbarK*Q;
			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << "...in " << duration.count() << " seconds." << endl;

			/* Convergence Check*
			** Type 2: */
			std::cout << "__Check convergence \n" ;
			t0 = chrono::high_resolution_clock::now();
			checkConvergenceCat2(Xk, Alpha, numEigs, ML, ML, S, conv2); 	// need to be MK^{-1}
			//cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
			t1 = chrono::high_resolution_clock::now();
			duration = t1 - t0;
			cout << "...in " << duration.count() << " seconds." << endl;
			cout << "Conv 2 \n" << conv2(1) << " " << conv2(numEigs / 2) << " " << conv2(numEigs - 1) << endl;

			/* Type 3: */
			checkConvergenceCat3(Q, Alpha, Mproj, numEigs, true, mu, conv3);
			//cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;
			cout << "Conv 3 \n" << conv3(1) << " " << conv3(numEigs / 2) << " " << conv3(numEigs - 1) << endl;

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
					//std::cout << "Checking convergence ID " << i << ": " << conv2(i) << std::endl; 
					lastConvergedId = i;
				}
				else {
					break;
				}
			}


			if ((k < maxIter) && (lastConvergedId < (numEigs - 1)))
				convergenceCheck = false;

		} while (!convergenceCheck);
		
		convergence2Stacks.conservativeResize(Eigen::NoChange, k);
		convergence3Stacks.conservativeResize(Eigen::NoChange, k);
		EigValsSIMStack.conservativeResize(Eigen::NoChange, k);

		EigVectsSIM = Xk;
		EigValsSIM = Alpha;

		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t00;
		cout << "...in " << duration.count() << " seconds." << endl;

		timingPerLevel(0) = duration.count();
		itersPerLevel(0) = k;
	}

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

		Eigen::VectorXd conv2(numEigs), conv3(numEigs);
		conv2.setConstant(std::numeric_limits<double>::max());
		conv3.setConstant(std::numeric_limits<double>::max());

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
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
		//Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;
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

				case MLEigenSolver::ShiftType::SHIFT_BATHE:
					cout << "___Bathe's shift proposal | shifting on index " << lastConvergedId << endl;
					mu = (Alpha(lastConvergedId) + Alpha(lastConvergedId - 1)) / 2.0;
					if (mu <= (Alpha(lastConvergedId) + 0.001*Alpha(firstNZ)))			mu = Alpha(lastConvergedId) + 0.001*Alpha(firstNZ) + std::numeric_limits<double>::epsilon();
					else if (mu >= (Alpha(lastConvergedId) - 0.001*Alpha(firstNZ)))		mu = Alpha(lastConvergedId) - 0.001*Alpha(firstNZ) - std::numeric_limits<double>::epsilon();
					factorization.compute(SL - mu*ML);
					break;

				case MLEigenSolver::ShiftType::SHIFT_ITOH_WILSON:
					/* [2] Itoh and Wilson's proposal */
					cout << "___Itoh and Wilson's shift proposal | shifting on index " << lastConvergedId << endl;
					mu = Alpha(lastConvergedId) + 0.9*(Alpha(lastConvergedId + 1) - Alpha(lastConvergedId));
					if ((Alpha(lastConvergedId + 1) - Alpha(lastConvergedId)) <= 0.001*Alpha(lastConvergedId))
						mu = 0.99*Alpha(lastConvergedId);
					factorization.compute(SL - mu*ML);
					break;

				case MLEigenSolver::ShiftType::SHIFT_GONG:
					/* [3] Gong's proposal */
					cout << "___Gong's shift proposal | shifting on index " << lastConvergedId << endl;
					mu = (Alpha(lastConvergedId + 1) + Alpha(lastConvergedId + 2)) / 2.0;
					if (mu <= (Alpha(lastConvergedId + 1) + 0.001*Alpha(firstNZ)))			mu = Alpha(lastConvergedId + 1) + 0.001*Alpha(firstNZ) + std::numeric_limits<double>::epsilon();
					else if (mu >= (Alpha(lastConvergedId + 2) - 0.001*Alpha(firstNZ)))		mu = Alpha(lastConvergedId + 2) - 0.001*Alpha(firstNZ) - std::numeric_limits<double>::epsilon();
					factorization.compute(SL - mu*ML);
					break;

				case MLEigenSolver::ShiftType::SHIFT_ZHAO:
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

			//cout << "Head: " << factorization.vectorD().head(10).transpose() << endl;
			//cout << "Tail: " << factorization.vectorD().tail(10).transpose() << endl;

			//cout << "Negative entries: ";
			//vector<int> negDiagStdVect; negDiagStdVect.reserve(100);
			//for (int i = 0; i < S.rows(); ++i) {
			//	if (factorization.vectorD()(i) < 0) {
			//		negDiagStdVect.push_back(i);
			//		cout << i << ": " << factorization.vectorD()(i) << " |";
			//		dCounter++;
			//	}
			//}
			//
			//Eigen::VectorXd a(negDiagStdVect.data());
			//Eigen::VectorXi negDiags = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(negDiagStdVect.data(), negDiagStdVect.size());
			//
			//Eigen::VectorXd negDiags(negDiagStdVect.data(), dCounter);
			//cout << "diag with negative ids: \n" << negDiags.transpose() << endl;
			//cout << "diag with negative ids: \n" << a.transpose() << endl;
			//cout << "Counter: " << dCounter << endl; 

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
				status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
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
				checkConvergenceCat2(Xk, Alpha, numEigs, ML, ML, SL, conv2); 	// need to be ML^{-1}
				cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
				/* Type 3: */
				checkConvergenceCat3(Q, Alpha, Mproj, numEigs, false, mu, conv3);
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

			if ((k < maxIter) && (lastConvergedId < (numEigs - 1)))
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
	*  @param	isWithShift	A boolean value, determining whether a shift is incorporated to the system.
	*/
	void MLSolverFunctions::performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift)
	{
		
		chrono::high_resolution_clock::time_point	t0, t1;
		chrono::duration<double>					duration;
		t0 = chrono::high_resolution_clock::now();
	
		printf("____[1] level: %d \n", k);
		int		q = std::max((int)(qParam * numEigs), numEigs + 8);
		double constexpr eps = 1e6 * std::numeric_limits<double>::epsilon();
		int status = 1;
		int lastConvergedId = 1;

		/* 1: RESTRICTION */
		printf("____<1> Restriction: \n");
		Eigen::SparseMatrix<double> SK, MK;
		if (k == 0) {
			SK = SL; MK = ML;
		}
		else {
			SK = U[k - 1].transpose() * SL * U[k - 1];
			MK = U[k - 1].transpose() * ML * U[k - 1];
		}
	
		/* 4: RECURSION */
		if (k == level)			// check if this already in the lowest/coarsest subspace
		{
			Eigen::MatrixXd EVecTemp;
			Eigen::VectorXd EValTemp;

			computeEigenGPU(SK, MK, EVecTemp, EValTemp);
			XkL = EVecTemp.topLeftCorner(MK.rows(), q);
			AkL = EValTemp.head(q);
		}
		// RECURSIVE for level k < max_level
		else
		{
			printf("____<2> Going deeper => \n ################");
			performMultiLevelSIM(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift);
			printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);
				
			/* [3] SUBSPACE ITERATION METHOD */
			//std::cout << "Dimensionality check: \n";
			//printf("Basis U:%dx%d | Approx. EigVect: %dx%d \n", U[k].rows(), U[k].cols(), XkL.rows(), XkL.cols());
			XkL = U[k] * XkL;
			printf("____<3> Lifting, Init vectors: %dx%d \n", XkL.rows(), XkL.cols());
	
			/* Adding a small number on the diagonal, for robustness */
			Eigen::SparseMatrix<double> SS;
			Eigen::SparseMatrix<double> Id; Id.resize(SK.rows(), SK.cols()); Id.setIdentity();
			//double eps = 1e-6;
	
			//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
			Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;
			//__SS = SK + eps*Id;		// Adding epsilon to the diagonal
			//__Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
			//__factorization.compute(SS);
			//factorization.compute(SK);
			//cout << "SS: \n" << SS.topLeftCorner(12, 6) << endl;
			//printf("____<4> Embedding a small-valued diagonal matrix ");
			std::cout << eps << std::endl;
	
			/* Temporary variables */
			bool convergenceCheck = false;
			Eigen::MatrixXd XbarK(XkL.rows(), q); XbarK.setZero();
			Eigen::MatrixXd Mproj, Sproj;
			Eigen::MatrixXd Q;
			//Eigen::VectorXd Alpha;
			Eigen::VectorXd conv2(numEigs), conv3(numEigs);
			conv2.setConstant(std::numeric_limits<double>::max());
			conv3.setConstant(std::numeric_limits<double>::max());
	
	
			/* Iterating*/
			printf("____<4> Subspace iteration.\n");
			int iterCount = 0;
			subspaceIterationMethodHelper(k, SK, MK, XbarK, XkL, AkL, status, numEigs, q, iterCount, maxIter, tol, factorization, isWithShift, false, lastConvergedId);
	
	
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

		/* 1: RESTRICTION */
		printf("____<1> Restriction: ");
		Eigen::SparseMatrix<double> SK, MK;
		if (k == 0) {
			SK = SL; MK = ML;
		}
		else {
			SK = U[k - 1].transpose() * SL * U[k - 1];
			MK = U[k - 1].transpose() * ML * U[k - 1];
		}
	
		/* 4: RECURSION */
		if (k == level)			// check if this already in the lowest/coarsest subspace
		{
			Eigen::MatrixXd EVecTemp;
			Eigen::VectorXd EValTemp;

			computeEigenGPU(SK, MK, EVecTemp, EValTemp);
			XkL = EVecTemp.topLeftCorner(MK.rows(), q);
			AkL = EValTemp.head(q);
			//XkL = EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), q);
			//AkL = EigValsReduced.head(q);
			printf("____Size of the initial vectors: %dx%d \n", XkL.rows(), XkL.cols());
		}
		// RECURSIVE for level k < max_level
		else
		{	
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
			Eigen::VectorXd conv2(numEigs), conv3(numEigs);
			conv2.setConstant(std::numeric_limits<double>::max());
			conv3.setConstant(std::numeric_limits<double>::max());

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
					case MLEigenSolver::ShiftType::SHIFT_BATHE:
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
	
					case MLEigenSolver::ShiftType::SHIFT_ITOH_WILSON:
						/* [2] Itoh and Wilson's proposal */
						cout << "___Itoh and Wilson's shift proposal | shifting on index " << lastConvergedId << endl;
						mu = AkL(lastConvergedId) + 0.9*(AkL(lastConvergedId + 1) - AkL(lastConvergedId));
						if ((AkL(lastConvergedId + 1) - AkL(lastConvergedId)) <= 0.001*AkL(lastConvergedId))
							mu = 0.99*AkL(lastConvergedId);
						//factorization.compute(SK - mu*MK);
						break;
	
					case MLEigenSolver::ShiftType::SHIFT_GONG:
						/* [3] Gong's proposal */
						cout << "___Gong's shift proposal | shifting on index " << lastConvergedId << endl;
						mu = (AkL(lastConvergedId + 1) + AkL(lastConvergedId + 2)) / 2.0;
						if (mu <= (AkL(lastConvergedId + 1) + 0.001*AkL(firstNZ)))			mu = AkL(lastConvergedId + 1) + 0.001*AkL(firstNZ) + std::numeric_limits<double>::epsilon();
						else if (mu >= (AkL(lastConvergedId + 2) - 0.001*AkL(firstNZ)))		mu = AkL(lastConvergedId + 2) - 0.001*AkL(firstNZ) - std::numeric_limits<double>::epsilon();
						//factorization.compute(SK - mu*MK);
						break;
	
					case MLEigenSolver::ShiftType::SHIFT_ZHAO:
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
				double tolCheck = std::min(1e-3, 0.1 * tol);
				for (int i = 0; i < Imax; ++i) {
					printf("Iteration: %d \n", iterCount + 1);


					//for (int j = 0; j < q; ++j) {						// easily parallelizeable
					//	Eigen::VectorXd v;
					//	v = MK * (XkL.col(j));
					//	XbarK.col(j) = factorization.solve(v);
					//}

					if (!isParallelSolverEnabled) {
						int solveCounter = 0;
						Eigen::VectorXd v;
						for (int j = 0; j < q; ++j) {						// easily parallelizeable				
							if (isConvergedVectsNotUsed && (conv2(j) < tolCheck && j > 0)) continue;
							v = MK * (XkL.col(j));
							XbarK.col(j) = factorization.solve(v);

							if (isMultipleSolve)
							{
								v = MK * XbarK.col(j);
								XbarK.col(j) = factorization.solve(v);
								//v = MK*XbarK.col(j);
								//XbarK.col(j) = factorization.solve(v);
							}
							solveCounter++;
						}
					}
					else
					{

						//const int		NUM_OF_THREADS = omp_get_num_procs();
						const int		NUM_OF_THREADS = std::min(q, omp_get_num_procs());
						cout << "We are using " << NUM_OF_THREADS << " threads for OpenMP\n";
						//const int		NUM_OF_THREADS = 2;
						omp_set_num_threads(NUM_OF_THREADS);
						int tid, ntids, ipts, istart, iproc;
						int iCol = 0;
#pragma omp parallel private(tid,ntids,ipts,istart,iCol)
						{
							iproc = omp_get_num_procs();
							tid = omp_get_thread_num();
							ntids = omp_get_num_threads();
							//ipts = (int)ceil((double) q / (double)ntids);
							//ipts = q / ntids;
							ipts = (q + ntids - 1) / ntids;
							istart = tid * ipts;
							if (tid == ntids - 1) ipts = q - istart;
							if (ipts <= 0) ipts = 0;

							Eigen::VectorXd v;

							for (iCol = istart; iCol < (istart + ipts); ++iCol) {
								if (iCol >= q) continue;
								if (isConvergedVectsNotUsed && conv2(iCol) < tolCheck && iCol > 0) continue;
								v = MK * (XkL.col(iCol));
								XbarK.col(iCol) = factorization.solve(v);

								if (isMultipleSolve)
								{
									v = MK * XbarK.col(iCol);
									XbarK.col(iCol) = factorization.solve(v);
									//v = MK*XbarK.col(iCol);
									//XbarK.col(iCol) = factorization.solve(v);
								}
							}
						}
					}
	
					/* Projection of M and K*/
					printf("________setting up reduced matrix M and S. ");
					Sproj = XbarK.transpose()*SK*XbarK;
					Mproj = XbarK.transpose()*MK*XbarK;
	
					printf("________solve reduced eigenvalue problem. ");
					status = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
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
					checkConvergenceCat2(XkL, AkL, numEigs, MK, MK, SK, conv2);	// need to be MK^{-1}
					cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
					/* Type 3: */
					checkConvergenceCat3(Q, AkL, Mproj, numEigs, false, mu, conv3);
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
	void MLSolverFunctions::performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
	{
		std::cout << "Functions" << std::endl;
	
		chrono::high_resolution_clock::time_point	t0, t1, t2;
		chrono::duration<double>					duration;
		t0 = chrono::high_resolution_clock::now();
	
		printf("____[1] level: %d \n", k);
		//int		q = max(2 * numEigs, numEigs + 8);
		int		q = qParam*numEigs;
		std::cout << "qParam is" << qParam << std::endl;
	
		int iterCount = 0;
		int factCounter = 0;
		int status = 1;
		bool isConvergedNotUsed = true;

		/* 1: RESTRICTION */
		printf("____<1> Restriction: ");
		t1 = chrono::high_resolution_clock::now();
		Eigen::SparseMatrix<double> SK, MK;
		if (k == 0) {
			SK = SL; MK = ML;
		}
		else {
			SK = U[k - 1].transpose() * SL * U[k - 1];
			MK = U[k - 1].transpose() * ML * U[k - 1];
		}
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "...in " << duration.count() << " seconds." << endl;
	
		/* 4: RECURSION */
		if (k == level)			// check if this already in the lowest/coarsest subspace
		{
			Eigen::MatrixXd EVecTemp;
			Eigen::VectorXd EValTemp;

			computeEigenGPU(SK, MK, EVecTemp, EValTemp);
			XkL = EVecTemp.topLeftCorner(MK.rows(), q);
			AkL = EValTemp.head(q);
			//XkL = EigVectsReducedNonLifted.topLeftCorner(EigVectsReducedNonLifted.rows(), q);
			//AkL = EigValsReduced.head(q);
			//printf("____Size of the initial vectors: %dx%d \n", XkL.rows(), XkL.cols());
		}
		// RECURSIVE for level k < max_level
		else
		{
			/* 1: RESTRICTION */			
			Eigen::SparseMatrix<double> Id;
			Id.resize(SK.rows(), SK.cols());
			Id.setIdentity();
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			cout << "...in " << duration.count() << " seconds." << endl;
	
	
			printf("____<2> Going deeper => \n ################");
			performMultiLevelSIM_withShiftOnce(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift, shiftType);
	
			//printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);
	
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
			Eigen::VectorXd conv2(q), conv3(q);
			conv2.setConstant(std::numeric_limits<double>::max());
			conv3.setConstant(std::numeric_limits<double>::max());
	
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
			double constexpr eps = 1e3*std::numeric_limits<double>::epsilon();
	
			printf("____<5> Subspace iteration.\n");
			printf("Size of XkL: %dx%d \n", XkL.rows(), XkL.cols());
			printf("Size of XbarK: %dx%d \n", XbarK.rows(), XbarK.cols());
			
			subspaceIterationMethodHelper(k, SK, MK, XbarK, XkL, AkL, status, numEigs, q, iterCount, maxIter, tol, factorization, isWithShift, false, lastConvergedId);
	
			printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());
	
			/* Print some data when it's on the finest level */
			if (k == 0) {
				cout << "____<7> Storing the convergence data. \n";
				EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
				convergence2Stacks.conservativeResize(Eigen::NoChange, iterCount);
				//convergence3Stacks.conservativeResize(Eigen::NoChange, iterCount);
			}
		}
	
		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t0;
		cout << "...in " << duration.count() << " seconds." << endl;
		if (k <= level) {
			timingPerLevel(k) = duration.count();
			itersPerLevel(k) = iterCount;
		}
	}
	
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
	void MLSolverFunctions::performMultiLevelSIM_withShiftOnceConverged(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
	{
		std::cout << "Functions" << std::endl;

		chrono::high_resolution_clock::time_point	t0, t1, t2;
		chrono::duration<double>					duration;
		t0 = chrono::high_resolution_clock::now();

		printf("____[1] level: %d \n", k);
		//int		q = max(2 * numEigs, numEigs + 8);
		int		q = qParam*numEigs;
		std::cout << "qParam is" << qParam << std::endl;

		int iterCount = 0;
		int factCounter = 0;
		int status = 1;
		bool isConvergedNotUsed = true;

		/* 1: RESTRICTION */
		printf("____<1> Restriction: ");
		t1 = chrono::high_resolution_clock::now();
		Eigen::SparseMatrix<double> SK, MK;
		if (k == 0) {
			SK = SL; MK = ML;
		}
		else {
			SK = U[k - 1].transpose() * SL * U[k - 1];
			MK = U[k - 1].transpose() * ML * U[k - 1];
		}
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "...in " << duration.count() << " seconds." << endl;

		/* 4: RECURSION */
		if (k == level){
			Eigen::MatrixXd EVecTemp;
			Eigen::VectorXd EValTemp;

			computeEigenGPU(SK, MK, EVecTemp, EValTemp);
			XkL = EVecTemp.topLeftCorner(MK.rows(), q);
			AkL = EValTemp.head(q);
		}else {			
			Eigen::SparseMatrix<double> Id;
			Id.resize(SK.rows(), SK.cols());
			Id.setIdentity();

			printf("____<2> Going deeper => \n ################");
			performMultiLevelSIM_withShiftOnceConverged(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift, shiftType);

			/* [3] SUBSPACE ITERATION METHOD */
			t1 = chrono::high_resolution_clock::now();
			printf("Level: %d | XkL: %dx%d | U: %dx%d \n", k, XkL.rows(), XkL.cols(), U[k].rows(), U[k].cols());
			XkL = U[k] * XkL;
			printf("Level: %d | XkL: %dx%d | U: %dx%d \n", k, XkL.rows(), XkL.cols(), U[k].rows(), U[k].cols());
			printf("____<3> Lifting, Init vectors: %dx%d ", XkL.rows(), XkL.cols());
			t2 = chrono::high_resolution_clock::now();
			duration = t2 - t1;
			cout << "...in " << duration.count() << " seconds." << endl;

			/* Adding a small number on the diagonal, for robustness */
			//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
			Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;

			/* Temporary variables */
			bool isConverged = false;
			Eigen::MatrixXd XbarK(XkL.rows(), q); XbarK.setZero();
			Eigen::MatrixXd Mproj, Sproj;
			Eigen::MatrixXd Q;
			Eigen::VectorXd conv2(q), conv3(q);
			conv2.setConstant(std::numeric_limits<double>::max());
			conv3.setConstant(std::numeric_limits<double>::max());			

			int						firstNZ = 0;
			int						lastConvergedId = 0;
			if (!isClampedBasis) {
				if (F.cols() == 3) {			/* FUNCTION */
					firstNZ = 1;				// first non-zero eigenvalues on the reduced/subspace Function Laplacian
					lastConvergedId = 1;		// first non-zero eigenvalue of the expected eigenvalues
				}
			}			

			printf("____<5> Subspace iteration.\n");
			subspaceIterationMethodHelper(k, SK, MK, XbarK, XkL, AkL, status, numEigs, q, iterCount, maxIter, tol, factorization, isWithShift, true, lastConvergedId);						
			printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());
			
			
			/* Print some data when it's on the finest level */
			if (k == 0) {
				cout << "____<7> Storing the convergence data. \n";
				EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
				convergence2Stacks.conservativeResize(Eigen::NoChange, iterCount);
				//convergence3Stacks.conservativeResize(Eigen::NoChange, iterCount);
				
			}
		}

		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t0;
		cout << "...in " << duration.count() << " seconds." << endl;
		if (k <= level) {
			timingPerLevel(k) = duration.count();
			itersPerLevel(k) = iterCount;
		}
	}
void MLSolverFunctions::subspaceIterationMethodHelper(int k, Eigen::SparseMatrix<double>& SK, Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd XbarK, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int status, int numEigs, int q, int& iterCount, int maxIter, double tol, Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& factorization, bool isWithShift, bool isConvergedNotUsed, int lastConvergedId)
{
	chrono::high_resolution_clock::time_point	t0, t1, t2;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	bool isConverged = false;
	int zhaoShiftId;

	//Eigen::SparseMatrix<double, Eigen::RowMajor> SK = SKK;
	//Eigen::SparseMatrix<double, Eigen::RowMajor> MK = MKK;

	iterCount = 0;
	int factCounter = 0;
	//bool isConvergedNotUsed = true;
	double mu;
	double eps = 1e6*std::numeric_limits<double>::epsilon();
	Eigen::SparseMatrix<double, Eigen::ColMajor> Id(SK.rows(), SK.cols());
	Id.setIdentity();
	Eigen::VectorXd conv2(q), conv3(q);
	conv2.setConstant(std::numeric_limits<double>::max());
	conv3.setConstant(std::numeric_limits<double>::max());
	//Eigen::MatrixXd Mproj, Sproj, Q;

	do {
		t1 = chrono::high_resolution_clock::now();
		if (status != 0)
		{
			/* [4] Zhao's proposal */
			//zhaoShiftId = lastConvergedId + min(5, (int)floor(zhaoParam*(numEigs - lastConvergedId)));
			zhaoShiftId = lastConvergedId + (int)floor(zhaoShiftParam*(numEigs - lastConvergedId));
			if (factCounter == 0) {
				if ((tol < 1e-9 && k == 0) || (!isWithShift)){
					std::cout << "__On finest level, no shift on really high accuracy!"; // << std::endl;
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

		double tolCheck = tol;
		tolCheck = std::min(1e-3, tol / 10.0);
		//if (tol > 0.99e-2) tolCheck = 1e-3;

		//SK.makeCompressed();
		//MK.makeCompressed();
		//cusparseDcsrm2(SK - mu * MK,  MK*XkL, XbarK);
		//cusolverSP(SK - mu * MK, MK* XkL, XbarK);
		//std::cout << "GPU: \n" << XbarK.topLeftCorner(10, 10) << std::endl;
		
		if (!isParallelSolverEnabled) {
			int solveCounter = 0;
			Eigen::VectorXd v;
			for (int j = 0; j < q; ++j) {						// easily parallelizeable				
				if (isConvergedVectsNotUsed && (conv2(j) < tolCheck && j > 0)) continue;
				v = MK * (XkL.col(j));
				XbarK.col(j) = factorization.solve(v);

				if (isMultipleSolve)
				{
					v = MK*XbarK.col(j);
					XbarK.col(j) = factorization.solve(v);
					//v = MK*XbarK.col(j);
					//XbarK.col(j) = factorization.solve(v);
				}
				solveCounter++;
			}
		}
		else
		{
			//const int		NUM_OF_THREADS = omp_get_num_procs();
			const int		NUM_OF_THREADS = std::min(q, omp_get_num_procs());
			cout << "We are using " << NUM_OF_THREADS << " threads for OpenMP\n";
			//const int		NUM_OF_THREADS = 2;
			omp_set_num_threads(NUM_OF_THREADS);
			int tid, ntids, ipts, istart, iproc;
			int iCol = 0;
#pragma omp parallel private(tid,ntids,ipts,istart,iCol)
			{
				iproc = omp_get_num_procs();
				tid = omp_get_thread_num();
				ntids = omp_get_num_threads();
				//ipts = (int)ceil((double) q / (double)ntids);
				//ipts = q / ntids;
				ipts = (q + ntids - 1) / ntids;
				istart = tid * ipts;
				if (tid == ntids - 1) ipts = q - istart;
				if (ipts <= 0) ipts = 0;

				Eigen::VectorXd v;

				for (iCol = istart; iCol < (istart + ipts); ++iCol) {
					if (iCol >= q) continue;
					if (isConvergedVectsNotUsed && conv2(iCol) < tolCheck && iCol > 0) continue;
					v = MK * (XkL.col(iCol));
					XbarK.col(iCol) = factorization.solve(v);

					if (isMultipleSolve)
					{
						v = MK*XbarK.col(iCol);
						XbarK.col(iCol) = factorization.solve(v);
						//v = MK*XbarK.col(iCol);
						//XbarK.col(iCol) = factorization.solve(v);
					}
				}
			}
		}

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "...in " << duration.count() << " seconds." << endl;

		//std::cout << "CPU: \n" << XbarK.topLeftCorner(10, 10) << std::endl;

		

		int iterStatus;
		if (isSIMinGPU) {
			iterStatus = SIMiterationInGPU(XbarK, SK, MK, XkL, AkL, iterCount, status);
		}
		else {
			iterStatus = SIMiterationInCPU(XbarK, SK, MK, XkL, AkL, status);
		}
		
		if (iterStatus < 0) continue;

		/* Convergence Check*
		** Type 2: */
		t1 = chrono::high_resolution_clock::now();
		printf("__Convergence check.");
		checkConvergenceCat2(XkL, AkL, numEigs, MK, MK, SK, conv2);	// need to be MK^{-1}
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		cout << "...in " << duration.count() << " seconds." << endl;

		//cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;

		/* Type 3: */
		//checkConvergenceCat3(Q, AkL, isWithShift, mu, conv3);
		//cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

		cout << "_____ [" << iterCount + 1 << "] convergence2 check:" << conv2(1) << " " << conv2(numEigs / 2) << " " << conv2(numEigs - 1) << endl;
		///cout << "_____ [" << iterCount + 1 << "] convergence3 check:" << conv3(1) << " " << conv3(numEigs / 2) << " " << conv3(numEigs - 1) << endl;

		//cout << AkL.head(numEigs).transpose() << endl;

		if (k == 0) {
			EigValsSIMStack.col(iterCount) = AkL.block(0, 0, numEigs - 1, 1);
			convergence2Stacks.col(iterCount) = conv2.block(0, 0, numEigs - 1, 1);
			//convergence3Stacks.col(iterCount) = conv3.block(0, 0, numEigs - 1, 1);
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
			//cout << "Checking convergence of ID " << i << ": " << conv2(i) << endl;
			if (conv2(i) < tol) {
				lastConvergedId = i;
			}
			else break;
		}

		if ((iterCount<maxIter) && (lastConvergedId < (numEigs - 1)))
			isConverged = false;

		cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl;

	} while (!isConverged);
}

int MLSolverFunctions::SIMiterationInCPU(const Eigen::MatrixXd& XbarK, const Eigen::SparseMatrix<double>& SK, const Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int& GPUeigsolverStatus)
{
	chrono::high_resolution_clock::time_point	t0, t1, t2;
	chrono::duration<double>					duration;

	Eigen::MatrixXd Mproj, Sproj, Q;

	/* Projection of M and K*/
	t1 = chrono::high_resolution_clock::now();
	printf("___setting up reduced matrix M and S. ");
	Sproj = XbarK.transpose() * SK;		// two steps for faster result
	Sproj = Sproj * XbarK;
	Mproj = XbarK.transpose() * MK;
	Mproj = Mproj * XbarK;
	//Sproj = SK * XbarK;
	//Sproj = XbarK.transpose() * Sproj;
	//Mproj = MK * XbarK;
	//Mproj = XbarK.transpose() * Mproj;
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "...in " << duration.count() << " seconds." << endl;

	t1 = chrono::high_resolution_clock::now();
	printf("__solve reduced eigenvalue problem. ");
	GPUeigsolverStatus = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
	cout << "__STATUS: " << GPUeigsolverStatus << endl;
	if (GPUeigsolverStatus != 0) {
		cout << "___!! COmputation of eigenvalues failed! \n";
		cout << "____Status: " << GPUeigsolverStatus << endl;
		//i = Imax;
		return -1;
		//break;
	}
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "...in " << duration.count() << " seconds." << endl;

	t1 = chrono::high_resolution_clock::now();
	printf("__lifting up/prolongation.");
	XkL = XbarK * Q;
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "...in " << duration.count() << " seconds." << endl;

	return 0;
}

int MLSolverFunctions::SIMiterationInGPU(const Eigen::MatrixXd& XbarK, const Eigen::SparseMatrix<double>& SK, const Eigen::SparseMatrix<double>& MK, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int iterCount, int& GPUeigsolverStatus)
{
	///chrono::high_resolution_clock::time_point	t0, t1, t2;
	///chrono::duration<double>					duration;
	///
	///std::cout << "IN GPU! \n";
	///std::cout << "Variable decs \n";
	///t0 = chrono::high_resolution_clock::now();
	///t1 = t0;
	///// Set up CUDA parameters and handles 
	///cudaError_t			cudaStat;										// cudaMalloc status
	///cublasStatus_t		blasStat;										// CUBLAS functions status
	///cublasHandle_t		blasHandle;										// CUBLAS context
	///cusparseHandle_t	cuSpHandle;										// cuSparse context
	///cusparseStatus_t	cuSpStat;										// cuSparse function status
	///cusolverDnHandle_t	cusolverHandle = NULL;								// cuSolver handle (for eigensolver)
	///cusolverStatus_t	cusolver_status = CUSOLVER_STATUS_SUCCESS;		// cusolver function status
	///
	///// Setup the data in CPU + GPU coresponding to matrix sizes
	///int m_SSp	= SK.rows();			// Sparse stiffness/cotangent matrix on that level
	///int n_SSp	= SK.cols();
	///int nnz_SSp = SK.nonZeros();
	///int m_SDn = XbarK.rows();			// Dense stiffness matrix, projected to Ritz-Raleigh space
	///int n_SDn = XbarK.cols();
	///int m_SDnSym = XbarK.cols();		// Sbar = X^T*S*X
	///int n_SDnSym = XbarK.cols();
	///int m_MSp	= MK.rows();			// Sparse Mass matrix on that level
	///int n_MSp	= MK.cols();
	///int nnz_MSp = MK.nonZeros();
	///int m_MDn = XbarK.rows();			// Dense mass matrix, projected to Ritz-Rayleigh space
	///int n_MDn = XbarK.cols();
	///int m_MDnSym = XbarK.cols();		// Mbar = X^T*M*X
	///int n_MDnSym = XbarK.cols();
	///int m_X = XbarK.rows();				// Dense projection matrix (the solution to linear systems)
	///int n_X = XbarK.cols();
	///int m_Q = XbarK.rows();				// Dense eigenvectors prolonged back
	///int n_Q = XbarK.cols();
	///int m_Qr = XbarK.cols();			// Dense eigenvectors in the Ritz-Rayleigh space
	///int n_Qr = m_Qr;
	///int n_A = XbarK.cols();				// Vector storing the eigenvalues
	///int* devInfo = NULL;				// For eigensolver in GPU
	///double* d_work = NULL;
	///int		lwork = 0;
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///// =============[SDn = SSp*X and MDn = MSP*X]=============
	///// Allocating space for matrix multiplication
	///std::cout << "__ Allocating memory in GPU and CPU ";
	///t1 = chrono::high_resolution_clock::now();
	///double* d_X; cudaMalloc(&d_X, m_X * n_X * sizeof(double));
	///double* d_SDn; cudaMalloc(&d_SDn, m_SDn * n_SDn * sizeof(double));
	///double* d_MDn; cudaMalloc(&d_MDn, m_MDn * n_MDn * sizeof(double));
	///double* d_SDnSym; cudaMalloc(&d_SDnSym, m_SDnSym * n_SDnSym * sizeof(double));
	///double* d_MDnSym; cudaMalloc(&d_MDnSym, m_MDnSym * n_MDnSym * sizeof(double));
	///
	/////double* h_SDn = (double*)malloc(m_SDn * n_SDn * sizeof(double));
	/////double* h_MDn = (double*)malloc(m_MDn * n_MDn * sizeof(double));
	/////double* h_SDnSym = (double*)malloc(m_SDnSym * n_SDnSym * sizeof(double));
	/////double* h_MDnSym = (double*)malloc(m_MDnSym * n_MDnSym * sizeof(double));
	///
	///double* d_csrVal_SSp; cudaMalloc(&d_csrVal_SSp, nnz_SSp * sizeof(double));
	///int* d_csrRowPtr_SSp; cudaMalloc(&d_csrRowPtr_SSp, (m_SSp + 1) * sizeof(int));
	///int* d_csrColInd_SSp; cudaMalloc(&d_csrColInd_SSp, nnz_SSp * sizeof(int));
	///
	///double* d_csrVal_MSp; cudaMalloc(&d_csrVal_MSp, nnz_MSp * sizeof(double));
	///int* d_csrRowPtr_MSp; cudaMalloc(&d_csrRowPtr_MSp, (m_MSp + 1) * sizeof(int));
	///int* d_csrColInd_MSp; cudaMalloc(&d_csrColInd_MSp, nnz_MSp * sizeof(int));
	///
	///double* d_A; cudaMalloc((void**)&d_A, sizeof(double) * n_A);
	///double* d_Q; cudaMalloc(&d_Q, m_Q * n_Q * sizeof(double));
	///cudaMalloc((void**)&devInfo, sizeof(int));
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__Copying matrices to GPU \n";
	///t1 = chrono::high_resolution_clock::now();
	///// COpying to GPU
	///cudaStat = cudaMemcpy(d_csrVal_SSp,		SK.valuePtr(),		nnz_SSp * sizeof(double),	cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrRowPtr_SSp,	SK.outerIndexPtr(), (m_SSp + 1) * sizeof(int),	cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrColInd_SSp,	SK.innerIndexPtr(), nnz_SSp * sizeof(int),		cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrVal_MSp,		MK.valuePtr(),		nnz_MSp * sizeof(double),	cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrRowPtr_MSp,	MK.outerIndexPtr(), (m_MSp + 1) * sizeof(int),	cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrColInd_MSp,	MK.innerIndexPtr(),	nnz_MSp * sizeof(int),		cudaMemcpyHostToDevice);
	///
	///cudaStat = cudaMemcpy(d_X, XbarK.data(), m_X * n_X * sizeof(double), cudaMemcpyHostToDevice);
	/////cudaStat = cudaMemcpy(d_C, CMat.data(), m_C * n_C * sizeof(double), cudaMemcpyHostToDevice);
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__Creating matrix in GPU ..";
	///t1 = chrono::high_resolution_clock::now();
	///// Create matrix in GPU
	///cusparseSpMatDescr_t matSSp, matMSp;
	///cusparseDnMatDescr_t matSDn, matMDn;
	///cusparseDnMatDescr_t matX;
	///void* dBufferS = NULL, * dBufferM = NULL;
	///size_t bufferSizeS = 0, bufferSizeM = 0;
	///double alpha = 1.0;
	///double beta = 0.0;
	///
	///cusparseCreate(&cuSpHandle);
	///cuSpStat = cusparseCreateCsr(&matSSp, m_SSp, n_SSp, nnz_SSp, d_csrRowPtr_SSp, d_csrColInd_SSp, d_csrVal_SSp, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	///cuSpStat = cusparseCreateCsr(&matMSp, m_MSp, n_MSp, nnz_MSp, d_csrRowPtr_MSp, d_csrColInd_MSp, d_csrVal_MSp, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	///cuSpStat = cusparseCreateDnMat(&matSDn, m_SDn, n_SDn, m_SDn, d_SDn, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///cuSpStat = cusparseCreateDnMat(&matMDn, m_MDn, n_MDn, m_MDn, d_MDn, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///cuSpStat = cusparseCreateDnMat(&matX, m_X, n_X, m_X, d_X, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///
	///
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__Allocating buffer size for SX = S(sp)*X(Dn) ..";
	///t1 = chrono::high_resolution_clock::now();
	///cusparseOperation_t opsNTrans = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;
	///cusparseOperation_t opsTrans = cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE;
	///
	///cuSpStat = cusparseSpMM_bufferSize(cuSpHandle, opsNTrans, opsNTrans, &alpha, matSSp, matX, &beta, matSDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, &bufferSizeS);
	///cudaMalloc((void**)&dBufferS, bufferSizeS);
	///cuSpStat = cusparseSpMM_bufferSize(cuSpHandle, opsNTrans, opsNTrans, &alpha, matMSp, matX, &beta, matMDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, &bufferSizeM);
	///cudaMalloc((void**)&dBufferM, bufferSizeM);
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__SX = S(sp)*X(Dn) ..";
	///t1 = chrono::high_resolution_clock::now();
	///
	///cuSpStat = cusparseSpMM(cuSpHandle, opsNTrans, opsNTrans, &alpha, matSSp, matX, &beta, matSDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, dBufferS);
	///cuSpStat = cusparseSpMM(cuSpHandle, opsNTrans, opsNTrans, &alpha, matMSp, matX, &beta, matMDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, dBufferM);
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///// =============[SDnSym = XT*SSp*X = XT*SDn and MDnSym = XT*MSP*X = XT*MDn]=============
	///std::cout << "__Sbar = XT * SX and Mbar = XY*SX..";
	///t1 = chrono::high_resolution_clock::now();
	///
	///blasStat = cublasCreate(&blasHandle);	
	///blasStat = cublasDgemm(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n_X, n_SDn, m_X, &alpha, d_X, m_X, d_SDn, m_SDn, &beta, d_SDnSym, m_SDnSym);
	///blasStat = cublasDgemm(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n_X, n_MDn, m_X, &alpha, d_X, m_X, d_MDn, m_MDn, &beta, d_MDnSym, m_MDnSym);
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///
	///// =============[SOLVING THE EIGENPROBLEM]=============
	///std::cout << "__Solving eigenproblem) ..";
	///t1 = chrono::high_resolution_clock::now();
	///cusolver_status = cusolverDnCreate(&cusolverHandle);
	///cusolverEigType_t	itype = cusolverEigType_t::CUSOLVER_EIG_TYPE_1;		// A*x = (lambda)*B*x 
	///cusolverEigMode_t	jobz = cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;	// compute eigenvalues and eigenvectors. 
	///cublasFillMode_t	uplo = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
	///cusolver_status = cusolverDnDsygvd_bufferSize(cusolverHandle, itype, jobz, uplo, n_MDnSym, d_SDnSym, m_SDnSym, d_MDnSym, m_MDnSym, d_A, &lwork);
	///cudaStat = cudaMalloc((void**)&d_work, sizeof(double) * lwork);
	///if (cudaStat != 0) {
	///	cout << "Error! Cannot request space to solve the eigenproblem." << endl;
	///}
	///
	///cusolver_status = cusolverDnDsygvd(cusolverHandle, itype, jobz, uplo, n_MDnSym, d_SDnSym, m_SDnSym, d_MDnSym, m_MDnSym, d_A, d_work, lwork, devInfo);
	///if (cusolver_status != 0){
	///	std::cout << "Cannot solve the eigenproblem \n";
	///	return -1;
	///}
	///cudaStat = cudaDeviceSynchronize();
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__Project Q = X * Q ..";
	///t1 = chrono::high_resolution_clock::now();
	///
	///// MAP TO REduced space on that level
	///blasStat = cublasDgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m_X, n_SDnSym, n_X, &alpha, d_X, m_X, d_SDnSym, m_SDnSym, &beta, d_Q, m_Q);
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__Copy to CPU ..";
	///t1 = chrono::high_resolution_clock::now();
	///
	///double* h_Q = (double*)malloc(m_Q * n_Q * sizeof(double));
	///double* h_A = (double*)malloc(n_Q * sizeof(double));
	///cudaStat = cudaMemcpy(h_Q, d_Q, sizeof(double) * m_Q*n_Q, cudaMemcpyDeviceToHost);
	///cudaStat = cudaMemcpy(h_A, d_A, sizeof(double) * m_MDnSym, cudaMemcpyDeviceToHost);
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__Mapping to EigenMatrix ..";
	///t1 = chrono::high_resolution_clock::now();
	///
	///XkL.resize(m_Q, n_Q);
	///AkL.resize(n_Q);
	///XkL = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_Q, m_Q, n_Q, Eigen::OuterStride<>(m_Q));
	///AkL = Eigen::Map<Eigen::VectorXd>(h_A, n_Q);
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///std::cout << "__Cleaning up GPU ..";
	///t1 = chrono::high_resolution_clock::now();
	///if (h_A)				free(h_A);
	///if (h_Q)				free(h_Q);
	///if (d_X)				cudaFree(d_X);
	///if (d_SDn)				cudaFree(d_SDn);
	///if (d_MDn)				cudaFree(d_MDn);
	///if (d_SDnSym)			cudaFree(d_SDnSym);
	///if (d_MDnSym)			cudaFree(d_MDnSym);
	///if (d_csrVal_SSp)		cudaFree(d_csrVal_SSp);
	///if (d_csrRowPtr_SSp)	cudaFree(d_csrRowPtr_SSp);
	///if (d_csrColInd_SSp)	cudaFree(d_csrColInd_SSp);
	///if (d_csrVal_MSp)		cudaFree(d_csrVal_MSp);
	///if (d_csrRowPtr_MSp)	cudaFree(d_csrRowPtr_MSp);
	///if (d_csrColInd_MSp)	cudaFree(d_csrColInd_MSp);
	///if (d_A)				cudaFree(d_A);
	///if (d_Q)				cudaFree(d_Q);
	///if (devInfo)			cudaFree(devInfo);
	///if (d_work)				cudaFree(d_work);
	///if (blasHandle)			cublasDestroy(blasHandle);
	///if (cuSpHandle)			cusparseDestroy(cuSpHandle);
	///if (cusolverHandle)		cusolverDnDestroy(cusolverHandle);
	///cusparseDestroySpMat(matSSp);
	///cusparseDestroySpMat(matMSp);
	///cusparseDestroyDnMat(matSDn);
	///cusparseDestroyDnMat(matMDn);
	///cusparseDestroyDnMat(matX);
	///cudaDeviceReset();
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	///duration = t2 - t0;
	///cout << "TOTAL IN " << duration.count() << " seconds." << endl;	
	///
	///GPUeigsolverStatus = 0;
	return 0;
}

void MLSolverFunctions::SIMinCPU(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol, int maxIter, int k)
{
	chrono::high_resolution_clock::time_point	t0, t1, t2;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	/* [0] Setup required variable */
	int		q = EigVals.rows();
	int		iterCount = 0;
	int		factCounter = 0;
	int		status = 1;
	int		lastConvergedId = 1;
	bool	isConverged = false;
	double  eps = 1e6 * std::numeric_limits<double>::epsilon();

	Eigen::SparseMatrix<double> Id;
	Id.resize(SL.rows(), SL.cols());
	Id.setIdentity();

	Eigen::MatrixXd XbarK(EigVects.rows(), q); XbarK.setZero();
	Eigen::MatrixXd Mproj, Sproj;
	Eigen::MatrixXd Q;
	Eigen::VectorXd conv2(q);
	Eigen::VectorXd Alpha;
	conv2.setConstant(std::numeric_limits<double>::max());
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;

	do {
		// [1] Factorization
		if (status != 0)
		{
			t1 = chrono::high_resolution_clock::now();
			/* Zhao's proposal on shifting*/
			if (factCounter == 0 && mu>std::numeric_limits<double>::epsilon()) {
					factorization.compute(SL - mu*ML);
			}
			else {
				eps *= 10;
				mu = eps;
				factorization.compute(SL + mu*Id);
			}
			factCounter++;
			t2 = chrono::high_resolution_clock::now();

			std::cout << "__[1] Factorization " << factCounter << " with shift of " << mu; // << std::endl;

			duration = t2 - t1;
			std::cout << " in " << duration.count() << " seconds." << std::endl;
		}

		
		std::cout << "Iteration: " << iterCount + 1 << std::endl;

		
		std::cout << "__[2] Solving each RHS ";
		t1 = chrono::high_resolution_clock::now();
		//std::cout << "MU: " << mu << std::endl;

		//cusolverSP(SL - mu * ML, M * EigVects, XbarK);
		//std::cout << "[GPU] XBarK \n" << XbarK.topLeftCorner(10, 10) << std::endl;

		//if (k == 0) {
		//	Eigen::MatrixXd ME = ML * EigVects;
		//	cuSolverCholTest(SL - mu * ML, ME, XbarK);
		//}
		/* Solve each RHS on CPU */
		double tolCheck = tol;
		tolCheck = std::min(1e-3, tol / 10.0);
		/* Regular/Serial solve on CPU */
		if (!isParallelSolverEnabled) {
			int solveCounter = 0;
			Eigen::VectorXd v;
			for (int j = 0; j < q; ++j) {									
				if (isConvergedVectsNotUsed && (conv2(j) < tolCheck && j > 0)) continue;
				v = ML * (EigVects.col(j));
				XbarK.col(j) = factorization.solve(v);				

				if (isMultipleSolve)
				{
					double n2 = XbarK.col(j).transpose() * ML * XbarK.col(j);
					XbarK.col(j) = XbarK.col(j) / sqrt(n2);
					v = ML * XbarK.col(j);					
					XbarK.col(j) = factorization.solve(v);
				}
				solveCounter++;
			}
		}
		/* Parallel solve on CPU*/
		else
		{
			const int		NUM_OF_THREADS = std::min(q, omp_get_num_procs());
			omp_set_num_threads(NUM_OF_THREADS);
			int tid, ntids, ipts, istart, iproc;
			int iCol = 0;
			#pragma omp parallel private(tid,ntids,ipts,istart,iCol)
			{
				iproc	= omp_get_num_procs();
				tid		= omp_get_thread_num();
				ntids	= omp_get_num_threads();
				ipts	= (q + ntids - 1) / ntids;
				istart	= tid * ipts;
				if (tid == ntids - 1) ipts = q - istart;
				if (ipts <= 0) ipts = 0;

				Eigen::VectorXd v;

				for (iCol = istart; iCol < (istart + ipts); ++iCol) {
					if (iCol >= q) continue;
					if (isConvergedVectsNotUsed && conv2(iCol) < tolCheck && iCol > 0) continue;
					v = ML * (EigVects.col(iCol));
					XbarK.col(iCol) = factorization.solve(v);

					if (isMultipleSolve){
						double n2 = XbarK.col(iCol).transpose() * ML * XbarK.col(iCol);
						XbarK.col(iCol) = XbarK.col(iCol) / sqrt(n2);
						v = ML * XbarK.col(iCol);
						XbarK.col(iCol) = factorization.solve(v);
					}
				}
			}
		}

		//std::cout << "[CPU] XBarK \n" << XbarK.topLeftCorner(10, 10) << std::endl;

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "...in " << duration.count() << " seconds." << std::endl;

		/* Projection of M and K*/
		t1 = chrono::high_resolution_clock::now();
		printf("__[3] Projection of matrix M and S. ");
		Sproj = XbarK.transpose() * SL;		// two steps for faster result
		Sproj = Sproj * XbarK;
		Mproj = XbarK.transpose() * ML;
		Mproj = Mproj * XbarK;
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "...in " << duration.count() << " seconds." << std::endl;

		t1 = chrono::high_resolution_clock::now();
		printf("__[4] Solve reduced eigenvalue problem. ");
		status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
		if (status != 0) {
			cout << "___Error!! Computation of eigenvalues failed! \n";
			continue;
		}
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "...in " << duration.count() << " seconds." << std::endl;

		t1 = chrono::high_resolution_clock::now();
		printf("__[5] Prolongation.");
		EigVects = XbarK * Q;
		EigVals = Alpha;
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "...in " << duration.count() << " seconds." << std::endl;

		/* Convergence Check*/
		t1 = chrono::high_resolution_clock::now();
		printf("__[6] Convergence check.");	
		if (convergedCriteria == 3) {
			checkConvergenceCat3(Q, EigVals, Mproj, numEigs, true, mu, conv2);
			//cout << "Conv 3 \n" << conv2.head(numEigs).transpose() << endl;
		}
		else {
			checkConvergenceCat2(EigVects, Alpha, numEigs, ML, MLinv, SL, conv2);
			//cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
		}
		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "...in " << duration.count() << " seconds." << std::endl;
		printf("mu is %.16f \n", mu);

		//cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
		std::cout << "_____ [" << iterCount + 1 << "] convergence2 check:" << conv2(1) << " " << conv2(numEigs / 2) << " " << conv2(numEigs - 1) << endl;
		//std::cout <<"Eigenvalues"<< Alpha.head(numEigs).transpose() << std::endl;

		if (k == 0) {
			EigValsSIMStack.col(iterCount) = EigVals.head(numEigs);
			convergence2Stacks.col(iterCount) = conv2.block(0, 0, numEigs - 1, 1);
		}

		iterCount++;
		if(numEigs>750)
			if (iterCount >= 2) 
				isMultipleSolve = false;

		/* Ignore the first iteration */
		if (iterCount >= maxIter || (conv2(numEigs - 1) < tol)) {
			isConverged = true;
		}

		/* Check convergence */
		for (int i = lastConvergedId; i < numEigs; ++i) {
			if (conv2(i) < tol) {
				lastConvergedId = i;
			}
			else break;
		}

		if ((iterCount < maxIter) && (lastConvergedId < (numEigs - 1)))
			isConverged = false;
		std::cout << "_____ [" << iterCount << "] Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId);
		std::cout << " (next=" << conv2(lastConvergedId+1)<<")" << std::endl;
				
		itersPerLevel(k) = iterCount;

	} while (!isConverged);

	if (k == 0) {
		EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
		convergence2Stacks.conservativeResize(Eigen::NoChange, iterCount);
	}
}

void MLSolverFunctions::SIMinGPU(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol, int maxIter, int k)
{
	///chrono::high_resolution_clock::time_point	t0, t1, t2;
	///chrono::duration<double>					duration;
	///
	///std::cout << "IN GPU! \n";
	///std::cout << "Variable decs \n";
	///t0 = chrono::high_resolution_clock::now();
	///t1 = t0;
	///// Set up CUDA parameters and handles 
	///cudaError_t					cudaStat;															// cudaMalloc status
	///cublasStatus_t				blasStat;															// CUBLAS functions status
	///cublasHandle_t				blasHandle;															// CUBLAS context
	///cusparseHandle_t			cuSpHandle;															// cuSparse context
	///cusparseStatus_t			cuSpStat;															// cuSparse function status
	///cusolverDnHandle_t			cusolverHandle = NULL;												// cuSolver handle (for eigensolver)
	///cusolverStatus_t			cusolver_status = CUSOLVER_STATUS_SUCCESS;							// cusolver function status
	///cusparseOperation_t			opNoTrans = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;
	///cusparseOperation_t			opTrans = cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE;
	///csrsm2Info_t				info = NULL;
	///const cusparseSolvePolicy_t policy = cusparseSolvePolicy_t::CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	///size_t						lworkInBytes = 0;
	///char*						d_workLS = NULL;
	///int*						devInfo = NULL;						// For eigensolver in GPU
	///double*						d_workE = NULL;
	///int							lworkE = 0;
	///int							algo = 0; /* non-block version */
	///cusparseCreate(&cuSpHandle);
	///cublasCreate(&blasHandle);
	///cusolverDnCreate(&cusolverHandle);
	///
	////* S-mu*M on CPU */
	///Eigen::SparseMatrix<double> SM = SL - mu*ML;
	///
	////* Set-up required matrices */
	////* Setup the data in CPU + GPU coresponding to matrix sizes */
	///int m_SM		= SM.rows();
	///int n_SM		= SM.cols();
	///int nnz_SM		= SM.nonZeros();
	///int m_SSp		= SL.rows();				// Sparse stiffness/cotangent matrix on that level
	///int n_SSp		= SL.cols();
	///int nnz_SSp		= SL.nonZeros();
	///int m_SDn		= EigVects.rows();			// Dense stiffness matrix, projected to Ritz-Raleigh space
	///int n_SDn		= EigVects.cols();
	///int m_SDnSym	= EigVects.cols();			// Sbar = X^T*S*X
	///int n_SDnSym	= EigVects.cols();
	///int m_MSp		= ML.rows();				// Sparse Mass matrix on that level
	///int n_MSp		= ML.cols();
	///int nnz_MSp		= ML.nonZeros();
	///int m_MDn		= EigVects.rows();			// Dense mass matrix, projected to Ritz-Rayleigh space
	///int n_MDn		= EigVects.cols();
	///int m_MDnSym	= EigVects.cols();			// Mbar = X^T*M*X
	///int n_MDnSym	= EigVects.cols();
	///int m_X			= EigVects.rows();			// Dense projection matrix (the solution to linear systems)
	///int n_X			= EigVects.cols();
	///int m_Q			= EigVects.rows();			// Dense eigenvectors prolonged back
	///int n_Q			= EigVects.cols();
	///int m_Qr		= EigVects.cols();			// Dense eigenvectors in the Ritz-Rayleigh space
	///int n_Qr		= m_Qr;
	///int n_A			= EigVects.cols();			// Vector storing the eigenvalues
	///int q			= EigVals.rows();
	///
	////* SIM Variable */
	///bool isConverged = false;
	///Eigen::VectorXd conv2(q);
	///conv2.setConstant(std::numeric_limits<double>::max());
	///int iterCount = 0;
	///int lastConvergedId = 1;
	///
	///
	///t2 = chrono::high_resolution_clock::now();
	///duration = t2 - t1;
	///cout << "...in " << duration.count() << " seconds." << endl;
	///
	////* Multiply (MX=M*X0) prior to factorization 
	///** Phi is the approximated eigenvectors from previous level  */
	///double* d_X0; cudaMalloc(&d_X0, m_X * n_X * sizeof(double));
	///double* d_MX; cudaMalloc(&d_MX, m_X * n_X * sizeof(double));
	///double* d_csrVal_MSp; cudaMalloc(&d_csrVal_MSp, nnz_MSp * sizeof(double));
	///int* d_csrRowPtr_MSp; cudaMalloc(&d_csrRowPtr_MSp, (m_MSp + 1) * sizeof(int));
	///int* d_csrColInd_MSp; cudaMalloc(&d_csrColInd_MSp, nnz_MSp * sizeof(int));
	///
	///cudaStat = cudaMemcpy(d_X0, EigVects.data(), m_X * n_X * sizeof(double), cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrVal_MSp, ML.valuePtr(), nnz_MSp * sizeof(double), cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrRowPtr_MSp, ML.outerIndexPtr(), (m_MSp + 1) * sizeof(int), cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrColInd_MSp, ML.innerIndexPtr(), nnz_MSp * sizeof(int), cudaMemcpyHostToDevice);
	///
	///cusparseSpMatDescr_t matMSp;
	///cusparseDnMatDescr_t matMX;
	///cusparseDnMatDescr_t matX0;
	///void* dBufferS = NULL, * dBufferM = NULL;
	///size_t bufferSizeS = 0, bufferSizeM = 0;
	///double alpha = 1.0;
	///double beta = 0.0;
	///
	///cuSpStat = cusparseCreateCsr(&matMSp, m_MSp, n_MSp, nnz_MSp, d_csrRowPtr_MSp, d_csrColInd_MSp, d_csrVal_MSp, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);	
	///cuSpStat = cusparseCreateDnMat(&matX0, m_X, n_X, m_X, d_X0, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///cuSpStat = cusparseCreateDnMat(&matMX, m_X, n_X, m_X, d_MX, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///	
	///cuSpStat = cusparseSpMM_bufferSize(cuSpHandle, opNoTrans, opNoTrans, &alpha, matMSp, matX0, &beta, matMX, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, &bufferSizeM);
	///cudaMalloc((void**)&dBufferM, bufferSizeM);
	///cuSpStat = cusparseSpMM(cuSpHandle, opNoTrans, opNoTrans, &alpha, matMSp, matX0, &beta, matMX, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, dBufferM);		
	///
	/////==========================
	///// UP TO HERE, THINGS ARE CORRECT!
	///// Overriding the d_MX
	/////========================	
	///
	////* Analysis and factorization in GPU of (S-mu*M) */
	///// Set-up SM on GPU (Sparse)
	///double* d_csrVal_SM; cudaMalloc(&d_csrVal_SM, nnz_SM * sizeof(double));
	///int* d_csrRowPtr_SM; cudaMalloc(&d_csrRowPtr_SM, (m_SM + 1) * sizeof(int));
	///int* d_csrColInd_SM; cudaMalloc(&d_csrColInd_SM, nnz_SM * sizeof(int));
	///// Copy to GPU
	///cudaStat = cudaMemcpy(d_csrVal_SM, SM.valuePtr(), nnz_SM * sizeof(double), cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrRowPtr_SM, SM.outerIndexPtr(), (m_SM + 1) * sizeof(int), cudaMemcpyHostToDevice);
	///cudaStat = cudaMemcpy(d_csrColInd_SM, SM.innerIndexPtr(), nnz_SM * sizeof(int), cudaMemcpyHostToDevice);
	///
	///cusparseMatDescr_t matSM;
	///cuSpStat = cusparseCreateMatDescr(&matSM);
	///cusparseSetMatIndexBase(matSM, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
	///cusparseSetMatType(matSM, cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_SYMMETRIC);
	///cusparseSetMatFillMode(matSM, cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER);
	///cusparseSetMatDiagType(matSM, cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT);
	///cusparseCreate(&cuSpHandle);
	///cusparseCreateCsrsm2Info(&info);
	///double alphaLS = 1.0;
	///double betaLS = 0.0;
	///cuSpStat = cusparseDcsrsm2_bufferSizeExt(cuSpHandle, algo, opNoTrans, opNoTrans, m_SM, n_X, nnz_SM, &alphaLS, matSM, d_csrVal_SM, d_csrRowPtr_SM, d_csrColInd_SM, d_MX, m_X, info, policy, &lworkInBytes);
	///std::cout << "Allocate buffer: " << cuSpStat << std::endl;
	///
	///// Analyze
	///cudaMalloc((void**)&d_workLS, lworkInBytes);
	///cuSpStat = cusparseDcsrsm2_analysis(cuSpHandle, algo, opNoTrans, opNoTrans, m_SM, n_X, nnz_SM, &alphaLS, matSM, d_csrVal_SM, d_csrRowPtr_SM, d_csrColInd_SM, d_MX, m_X, info, policy, d_workLS);
	///std::cout << "Factorize matrix: " << cuSpStat << std::endl;
	///
	///// ============================ Test
	///double* h_Mat = (double*)malloc(m_X * n_X * sizeof(double));
	///cudaStat = cudaMemcpy(h_Mat, d_MX, sizeof(double) * m_X * n_X, cudaMemcpyDeviceToHost);
	///Eigen::MatrixXd Mat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_Mat, m_X, n_X, Eigen::OuterStride<>(m_X));
	///std::cout << "[GPU] M*X \n" << Mat.topLeftCorner(10, 10) << std::endl;
	///
	///Mat = ML * EigVects;
	///std::cout << "[GPU] M*X \n" << Mat.topLeftCorner(10, 10) << std::endl;
	///// ============================
	///do {
	///	// Solve
	///	cuSpStat = cusparseDcsrsm2_solve(cuSpHandle, algo, opNoTrans, opNoTrans, m_SM, n_X, nnz_SM, &alpha, matSM, d_csrVal_SM, d_csrRowPtr_SM, d_csrColInd_SM, d_MX, m_X, info, policy, d_workLS);
	///	std::cout << "Solve linear system: " << cuSpStat << std::endl;
	///	//---------------------------
	///	cudaStat = cudaMemcpy(h_Mat, d_MX, sizeof(double) * m_X * n_X, cudaMemcpyDeviceToHost);
	///	Mat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_Mat, m_X, n_X, Eigen::OuterStride<>(m_X));
	///	std::cout << "[GPU] (S-mu*M)Xbar=MX \n" << Mat.topLeftCorner(10, 10) << std::endl;
	///
	///	//double* h_Mat = (double*)malloc(m_X * n_X * sizeof(double));
	///	//cudaStat = cudaMemcpy(h_Mat, d_MX, sizeof(double) * m_X * n_X, cudaMemcpyDeviceToHost);
	///	//Eigen::MatrixXd Mat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_Mat, m_X, n_X, Eigen::OuterStride<>(m_X));
	///	//std::cout << "[GPU] XbarK \n" << Mat.topLeftCorner(10, 10) << std::endl;
	///	
	///	/* In CPU */
	///	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> linSolver;
	///	linSolver.compute(SM);
	///	Eigen::MatrixXd MX = ML * EigVects;
	///	Eigen::MatrixXd Xbar(SL.rows(), EigVects.cols());
	///	for (int i = 0; i < MX.cols(); ++i) {
	///		Eigen::VectorXd v = MX.col(i);
	///		Xbar.col(i) = linSolver.solve(v);
	///	}
	///	std::cout << "[CPU] (S-mu*M)Xbar=MX \n" << Xbar.topLeftCorner(10, 10) << std::endl;
	///
	///	//----------------------------
	///
	///	/* SIM Iteration */
	///	/* [1] Projection */
	///	double* d_SDn; cudaMalloc(&d_SDn, m_SDn * n_SDn * sizeof(double));
	///	double* d_MDn; cudaMalloc(&d_MDn, m_MDn * n_MDn * sizeof(double));
	///	double* d_SDnSym; cudaMalloc(&d_SDnSym, m_SDnSym * n_SDnSym * sizeof(double));
	///	double* d_MDnSym; cudaMalloc(&d_MDnSym, m_MDnSym * n_MDnSym * sizeof(double));
	///
	///	double* d_csrVal_SSp; cudaMalloc(&d_csrVal_SSp, nnz_SSp * sizeof(double));
	///	int* d_csrRowPtr_SSp; cudaMalloc(&d_csrRowPtr_SSp, (m_SSp + 1) * sizeof(int));
	///	int* d_csrColInd_SSp; cudaMalloc(&d_csrColInd_SSp, nnz_SSp * sizeof(int));
	///
	///	cudaStat = cudaMemcpy(d_csrVal_SSp, SL.valuePtr(), nnz_SSp * sizeof(double), cudaMemcpyHostToDevice);
	///	cudaStat = cudaMemcpy(d_csrRowPtr_SSp, SL.outerIndexPtr(), (m_SSp + 1) * sizeof(int), cudaMemcpyHostToDevice);
	///	cudaStat = cudaMemcpy(d_csrColInd_SSp, SL.innerIndexPtr(), nnz_SSp * sizeof(int), cudaMemcpyHostToDevice);
	///
	///	cusparseSpMatDescr_t matSSp;
	///	cusparseDnMatDescr_t matSDn, matMDn;
	///	void* dBufferS = NULL, * dBufferM = NULL;
	///	size_t bufferSizeS = 0, bufferSizeM = 0;
	///
	///	cuSpStat = cusparseCreateCsr(&matSSp, m_SSp, n_SSp, nnz_SSp, d_csrRowPtr_SSp, d_csrColInd_SSp, d_csrVal_SSp, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	///	cuSpStat = cusparseCreateDnMat(&matSDn, m_SDn, n_SDn, m_SDn, d_SDn, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///	cuSpStat = cusparseCreateDnMat(&matMDn, m_MDn, n_MDn, m_MDn, d_MDn, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///	//cuSpStat = cusparseCreateDnMat(&matMX, m_X, n_X, m_X, d_MX, CUDA_R_64F, CUSPARSE_ORDER_COL);
	///
	///	// S1 = S*X and M1=M*S
	///	cuSpStat = cusparseSpMM_bufferSize(cuSpHandle, opNoTrans, opNoTrans, &alpha, matSSp, matMX, &beta, matSDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, &bufferSizeS);
	///	cudaMalloc((void**)&dBufferS, bufferSizeS);
	///	cuSpStat = cusparseSpMM_bufferSize(cuSpHandle, opNoTrans, opNoTrans, &alpha, matMSp, matMX, &beta, matMDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, &bufferSizeM);
	///	cudaMalloc((void**)&dBufferM, bufferSizeM);
	///
	///	cuSpStat = cusparseSpMM(cuSpHandle, opNoTrans, opNoTrans, &alpha, matSSp, matMX, &beta, matSDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, dBufferS);
	///	cuSpStat = cusparseSpMM(cuSpHandle, opNoTrans, opNoTrans, &alpha, matMSp, matMX, &beta, matMDn, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, dBufferM);
	///
	///	// SDnSym = XT * S * X
	///	blasStat = cublasDgemm(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n_X, n_SDn, m_X, &alpha, d_MX, m_X, d_SDn, m_SDn, &beta, d_SDnSym, m_SDnSym);
	///	blasStat = cublasDgemm(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n_X, n_MDn, m_X, &alpha, d_MX, m_X, d_MDn, m_MDn, &beta, d_MDnSym, m_MDnSym);		
	///
	///	/* [2] Solving reduced eigenproblem */
	///	double* d_A; cudaMalloc((void**)&d_A, sizeof(double) * n_A);
	///	double* d_Q; cudaMalloc(&d_Q, m_Q * n_Q * sizeof(double));
	///	cudaMalloc((void**)&devInfo, sizeof(int));
	///
	///	cusolverEigType_t	itype = cusolverEigType_t::CUSOLVER_EIG_TYPE_1;		// A*x = (lambda)*B*x 
	///	cusolverEigMode_t	jobz = cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;	// compute eigenvalues and eigenvectors. 
	///	cublasFillMode_t	uplo = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
	///	cusolver_status = cusolverDnDsygvd_bufferSize(cusolverHandle, itype, jobz, uplo, n_MDnSym, d_SDnSym, m_SDnSym, d_MDnSym, m_MDnSym, d_A, &lworkE);
	///	cudaStat = cudaMalloc((void**)&d_workE, sizeof(double) * lworkE);
	///	if (cudaStat != 0) {
	///		cout << "Error! Cannot request space to solve the eigenproblem." << endl;
	///	}
	///
	///	cusolver_status = cusolverDnDsygvd(cusolverHandle, itype, jobz, uplo, n_MDnSym, d_SDnSym, m_SDnSym, d_MDnSym, m_MDnSym, d_A, d_workE, lworkE, devInfo);
	///	if (cusolver_status != cudaError::cudaSuccess) {
	///		std::cout << "Cannot solve reduced eigenproblem \n";
	///		continue;
	///	}
	///
	///	cudaStat = cudaDeviceSynchronize();
	///
	///	/* [3] Prolongation */
	///	blasStat = cublasDgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m_X, n_SDnSym, n_X, &alpha, d_MX, m_X, d_SDnSym, m_SDnSym, &beta, d_Q, m_Q);
	///	
	///	/* [4] Copy to CPU  */
	///	double* h_Q = (double*)malloc(m_Q * n_Q * sizeof(double));
	///	double* h_A = (double*)malloc(n_Q * sizeof(double));
	///	cudaStat = cudaMemcpy(h_Q, d_Q, sizeof(double) * m_Q * n_Q, cudaMemcpyDeviceToHost);
	///	cudaStat = cudaMemcpy(h_A, d_A, sizeof(double) * m_MDnSym, cudaMemcpyDeviceToHost);
	///
	///	EigVects.resize(m_Q, n_Q);
	///	EigVals.resize(n_Q);
	///	EigVects = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(h_Q, m_Q, n_Q, Eigen::OuterStride<>(m_Q));
	///	EigVals = Eigen::Map<Eigen::VectorXd>(h_A, n_Q);
	///
	///	/* [5] Convergence check */
	///	checkConvergenceCat2(EigVects, EigVals, numEigs, ML, MLinv, SL, conv2);
	///	
	///	iterCount++;
	///
	///	/* Ignore the first iteration */
	///	if (iterCount >= maxIter || (conv2(numEigs - 1) < tol)) {
	///		isConverged = true;
	///	}			
	///
	///	/* Check convergence */
	///	for (int i = lastConvergedId; i < numEigs; ++i) {
	///		if (conv2(i) < tol) {
	///			lastConvergedId = i;
	///		}
	///		else break;
	///	}
	///
	///	if ((iterCount < maxIter) && (lastConvergedId < (numEigs - 1)))
	///		isConverged = false;
	///
	///	cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl;
	///} while (!isConverged);

}


void MLSolverFunctions::solve(int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset, double levelRatio, double basisSupport) {
		constructMassMatrix();
		initiateAdM();
		constructStiffnessMatrix();

		std::cout << "Iterating over converged?" << isConvergedVectsNotUsed << std::endl; 

		chrono::high_resolution_clock::time_point	t0, t1;
		chrono::duration<double>					duration;
		cout << "> Constructing hierarchy... " << endl;
		t0 = chrono::high_resolution_clock::now();

		Eigen::VectorXi sampSize;
		//setSamplingLevels(sampSize, numEigs, levelRatio);
		int hierarchyMode = 0;
		switch (hierarchyMode)
		{
		case 0:
			//setSamplingLevelsConstantRate(sampSize, numEigs, levelRatio);
			setSamplingLevelFixedNum(sampSize, numEigs, levelRatio);
			break;
		case 1:
			setSamplingLevelsAuto(sampSize, numEigs, levelRatio);
			break;
		case 2:
			setSamplingLevelsSameRate(sampSize, numEigs, levelRatio);
			break;
		default:
			break;
		}
		//setSamplingLevelsAuto(sampSize, numEigs, levelRatio);
		setInitialSystem(sampSize.rows());
		constructSampleHierarchy(sampSize);
		constructBasis(basisSupport);

		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t0;
		cout << "in " << duration.count() << " seconds." << endl;
		
		// SHIFT is FAKE-ly used to switch between DISCARDING the converged and not!
		performSubspaceIterationMethod(SIMType::SIM_MultiLevel, numEigs, maxIter, tol, isWithShift);

		t1 = chrono::high_resolution_clock::now();
		duration = t1 - t0;
		cout << "in " << duration.count() << " seconds." << endl;
		std::cout << "TOTAL H-SiM time in " << duration.count() << " seconds." << std::endl;
	}

	
void MLSolverFunctions::solve(int numEigs, double tol, int maxIter, double levelRatio, double basisSupport, bool isWithShift) 
{
	chrono::high_resolution_clock::time_point	t0, t1, t2, t01;
	chrono::duration<double>					duration;
	Eigen::Vector2d	HSIMtime; HSIMtime.setZero();

	std::cout << "[1] Construct system matrices... \n";
	t1 = chrono::high_resolution_clock::now();

	/* [1] Set up the System matrices */
	constructMassMatrix();
	initiateAdM();
	constructStiffnessMatrix();

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	std::cout << "    in " << duration.count() << " seconds." << std::endl;


	/* [2] Set up the Hierarchy */
	std::cout << "[2] Constructing hierarchy... " << std::endl;
	t1 = chrono::high_resolution_clock::now();
	t0 = t1;

	Eigen::VectorXi sampSize;
	
	switch (hierarchyMode){
	case 0:
		//setSamplingLevelsConstantRate(sampSize, numEigs, levelRatio);	break;
		setSamplingLevelFixedNum(sampSize, numEigs, levelRatio);		break;
	case 1:
		setSamplingLevelsAuto(sampSize, numEigs, levelRatio);			break;
	case 2:
		setSamplingLevelsSameRate(sampSize, numEigs, levelRatio);		break;
	default:
		break;
	}
	setInitialSystem(sampSize.rows());
	constructSampleHierarchy(sampSize);
	constructBasis(basisSupport);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	std::cout << "    in " << duration.count() << " seconds." << std::endl;

	/* Set-up reduced system matrices */
	std::cout << "[3] Setup reduced system matrices ... ";
	t1 = chrono::high_resolution_clock::now();
	const int nLevel = sampSize.rows() + 1;
	vector<Eigen::SparseMatrix<double>>	Mred(nLevel), Sred(nLevel);			// MInvRed(nLevel);
	Sred[0] = S; Mred[0] = M;												// MInvRed[0] = Minv;
	for (unsigned int i = 1; i < nLevel; ++i) {
		Sred[i] = U[i-1].transpose() * Sred[i-1] * U[i-1];
		Mred[i] = U[i-1].transpose() * Mred[i-1] * U[i-1];
		//MInvRed[i] = U[i - 1].transpose() * MInvRed[i - 1] * U[i - 1];
	}


	/* Set-up matrices for logging purposes */
	EigValsSIMStack.resize(numEigs, maxIter);
	EigValsSIMStack.setZero();
	convergence2Stacks.resize(numEigs, maxIter);
	convergence2Stacks.setZero();
	timingPerLevel.resize(nLevel);
	itersPerLevel.resize(nLevel);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	std::cout << "in " << duration.count() << " seconds." << std::endl;

	std::cout << "[4] Solve eigenproblem in the coarsest level:... ";
	t1 = chrono::high_resolution_clock::now();

	duration = t1 - t0;
	HSIMtime(0) = duration.count();


	int		q = qParam * numEigs;
	computeEigenGPU(Sred[nLevel-1], Mred[nLevel - 1], EigVectsSIM, EigValsSIM);
	EigVectsSIM.conservativeResize(Eigen::NoChange_t::NoChange, q);
	EigValsSIM.conservativeResize(q);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	std::cout << "in " << duration.count() << " seconds." << std::endl;
	timingPerLevel(nLevel-1) = duration.count();
	itersPerLevel(nLevel - 1) = 0;

	for (int i = nLevel - 2; i >= 0; --i) {
		std::cout << "[5] SIM [a] Prolongation...";
		t1 = chrono::high_resolution_clock::now();		
		t01 = t1;

		EigVectsSIM = U[i] * EigVectsSIM;
		int shiftId = std::floor(zhaoShiftParam * numEigs);
		double mu;
		mu = EigValsSIM(shiftId);
		printf("Shift param: %.2f | shift ID: %d | mu:", zhaoShiftParam, shiftId);
		//std::cout << "mu:" << mu << std::cout;
		std::cout << mu << std::endl;
		//if (!isWithShift) mu = 0.0;

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "in " << duration.count() << " seconds." << std::endl;

		std::cout << "[a] SIM on level " << i << std::endl;
		t1 = chrono::high_resolution_clock::now();
		
		//if (numEigs >= 750) zhaoShiftParam = 0.25;

		if (isSIMinGPU) {
			SIMinGPU(Sred[i], Mred[i], Mred[i], EigVectsSIM, EigValsSIM, numEigs, mu, tol, maxIter, i);
		}else {
			SIMinCPU(Sred[i], Mred[i], Mred[i], EigVectsSIM, EigValsSIM, numEigs, mu, tol, maxIter, i);
		}

		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "   in " << duration.count() << " seconds." << std::endl;

		if (i < nLevel-1) {
			duration = t2 - t01;
			timingPerLevel(i) = duration.count();
		}
	}


	EigVectsSIM.conservativeResize(Eigen::NoChange, numEigs);
	EigValsSIM.conservativeResize(numEigs);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t0;
	std::cout << "TOTAL H-SiM time in " << duration.count() << " seconds." << std::endl;
	HSIMtime(1) = duration.count() - HSIMtime(0);

	// Sturm's check
	sturmCheck(S, M, EigValsSIM);

	/* Write the convergence information to an external file. */
	std::cout << "Write the log to files." << std::endl;
	string modelName = "Model_" + to_string(V.rows()) + "_" + to_string(V.cols());
	string fileConv = "../../Data/" + modelName + "_EigValsStack_" + to_string(numEigs) + "_method_" + to_string(0) + "_" + (isWithShift ? "shift_" + to_string(3) + "_" : "") + to_string(tol) + ".txt";
	std::cout << "File path: " << fileConv << std::endl;
	writeDenseMatrixToFile(itersPerLevel, fileConv);
	writeDenseMatrixToFile(HSIMtime, fileConv);
	writeDenseMatrixToFile(timingPerLevel, fileConv);
	writeDenseMatrixToFile(EigValsSIMStack, fileConv);
	string fileEigs = "../../Data/" + modelName + "_Convergence2Stack_" + to_string(numEigs) + "_method_" + to_string(0) + "_" + (isWithShift ? "shift_" + to_string(3) + "_" : "") + to_string(tol) + ".txt";
	writeDenseMatrixToFile(convergence2Stacks, fileEigs);

	
}

void MLSolverFunctions::solveBMG(int numEigs, double tol, int maxIter, double levelRatio, double basisSupport, bool isWithShift) {
	chrono::high_resolution_clock::time_point	t0, t1, t2, t01;
	chrono::duration<double>					duration;
	Eigen::Vector2d	HSIMtime; HSIMtime.setZero();

	Eigen::MatrixXd EigVectsH, EigVectsHbar, EigVectsHtilde;
	Eigen::MatrixXd XH, Xh, XHh;
	Eigen::VectorXd EigValsH, EigValsHbar, EigValsHtilde;

	std::cout << "[1] Construct system matrices... \n";
	t1 = chrono::high_resolution_clock::now();

	/* [1] Set up the System matrices */
	constructMassMatrix();
	initiateAdM();
	constructStiffnessMatrix();

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	std::cout << "    in " << duration.count() << " seconds." << std::endl;


	/* [2] Set up the Hierarchy */
	std::cout << "[2] Constructing hierarchy... " << std::endl;
	t1 = chrono::high_resolution_clock::now();
	t0 = t1;

	Eigen::VectorXi sampSize;

	switch (hierarchyMode) {
	case 0:
		//setSamplingLevelsConstantRate(sampSize, numEigs, levelRatio);	break;
		setSamplingLevelFixedNum(sampSize, numEigs, levelRatio, false);		break;
	case 1:
		setSamplingLevelsAuto(sampSize, numEigs, levelRatio, false);			break;
	case 2:
		setSamplingLevelsSameRate(sampSize, numEigs, levelRatio, false);		break;
	default:
		break;
	}
	setInitialSystem(sampSize.rows());
	constructSampleHierarchy(sampSize);
	constructBasis(basisSupport);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	std::cout << "    in " << duration.count() << " seconds." << std::endl;

	/* Set-up reduced system matrices */
	std::cout << "[3] Setup reduced system matrices ... ";
	t1 = chrono::high_resolution_clock::now();
	const int nLevel = sampSize.rows() + 1;
	vector<Eigen::SparseMatrix<double>>	Mred(nLevel), Sred(nLevel);			
	Sred[0] = S; Mred[0] = M;												
	for (unsigned int i = 1; i < nLevel; ++i) {
		Sred[i] = U[i - 1].transpose() * Sred[i - 1] * U[i - 1];
		Mred[i] = U[i - 1].transpose() * Mred[i - 1] * U[i - 1];
	}

	std::cout << "[4] Solve eigenproblem in the coarsest level:... ";
	t1 = chrono::high_resolution_clock::now();	

	computeEigenGPU(Sred[nLevel - 1], Mred[nLevel - 1], EigVectsHbar, EigValsHbar);
	EigVectsHbar.conservativeResize(Eigen::NoChange_t::NoChange, numEigs);
	EigValsHbar.conservativeResize(numEigs);
	std::cout << "Level " << nLevel - 1 << std::endl;
	std::cout << EigValsHbar.transpose() << std::endl;

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	std::cout << "in " << duration.count() << " seconds." << std::endl;

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;
	double  eps = 1e6 * std::numeric_limits<double>::epsilon();
	Eigen::SparseMatrix<double> Id;
	Eigen::MatrixXd Mproj, Sproj;
	Eigen::MatrixXd Q;
	Eigen::VectorXd Alpha;
	int status;

	for (int i = nLevel - 2; i >= 0; --i) {
		std::cout << "LEVEL: " << i << std::endl;
		std::cout << "[5] SIM [a] Prolongation...";
		t1 = chrono::high_resolution_clock::now();
		t01 = t1;

		// XH is equal to EigVectsHbar
		EigVectsHbar = U[i] * EigVectsHbar;
		EigVectsHtilde.resize(EigVectsHbar.rows(), EigVectsHbar.cols());

		/* factorization */
		std::cout << "Factorization\n";
		Id.resize(Mred[i].rows(), Mred[i].cols());
		Id.setIdentity();
		//factorization.compute(Sred[i] + eps*Id);
		factorization.compute(Sred[i]);
		
		std::cout << "Solving linear system\n";
		/* Solving using BMG*/
		double tolCheck = tol;
		tolCheck = std::min(1e-3, tol / 10.0);
		/* Regular/Serial solve on CPU */
		if (!isParallelSolverEnabled) {
			int solveCounter = 0;
			Eigen::VectorXd v;
			for (int j = 0; j < numEigs; ++j) {
				v = EigValsHbar(j)* Mred[i] * (EigVectsHbar.col(j));
				EigVectsHtilde.col(j) = factorization.solve(v);

				//if (isMultipleSolve)
				//{
				//	double n2 = EigVectsHtilde.col(j).transpose() * Mred[i] * EigVectsHtilde.col(j);
				//	EigVectsHtilde.col(j) = EigVectsHtilde.col(j) / sqrt(n2);
				//	v = EigValsHbar(j)* Mred[i] * EigVectsHtilde.col(j);
				//	EigVectsHtilde.col(j) = factorization.solve(v);
				//}
				solveCounter++;
			}
		}
		/* Parallel solve on CPU*/
		else
		{
			const int		NUM_OF_THREADS = std::min(numEigs, omp_get_num_procs());
			omp_set_num_threads(NUM_OF_THREADS);
			int tid, ntids, ipts, istart, iproc;
			int iCol = 0;
#pragma omp parallel private(tid,ntids,ipts,istart,iCol)
			{
				iproc = omp_get_num_procs();
				tid = omp_get_thread_num();
				ntids = omp_get_num_threads();
				ipts = (numEigs + ntids - 1) / ntids;
				istart = tid * ipts;
				if (tid == ntids - 1) ipts = numEigs - istart;
				if (ipts <= 0) ipts = 0;

				Eigen::VectorXd v;

				for (iCol = istart; iCol < (istart + ipts); ++iCol) {
					if (iCol >= numEigs) continue;
					v = EigValsHbar(iCol)* Mred[i]* (EigVectsHbar.col(iCol));
					EigVectsHtilde.col(iCol) = factorization.solve(v);

					//if (isMultipleSolve) {
					//	double n2 = EigVectsHtilde.col(iCol).transpose() * Mred[i] * EigVectsHtilde.col(iCol);
					//	EigVectsHtilde.col(iCol) = EigVectsHtilde.col(iCol) / sqrt(n2);
					//	v = EigValsHbar(i)* Mred[i] * EigVectsHtilde.col(iCol);
					//	EigVectsHtilde.col(iCol) = factorization.solve(v);
					//}
				}
			}
		}

		/* Orthogonalization via reduced eigenproblem */
		std::cout << "Orthonormalization via eigensolve (for enrichment) \n";
		std::cout << "__[1] projection.\n";
		Sproj = EigVectsHtilde.transpose()*Sred[i] * EigVectsHtilde;
		Mproj = EigVectsHtilde.transpose()*Mred[i] * EigVectsHtilde;
		
		std::cout << "__[2] Solving reduced eigenproblem: ";
		status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
		std::cout << status << std::endl;
		
		std::cout << "__[3] Prolongation \n";
		EigVectsHtilde = EigVectsHtilde*Q;
		EigValsHtilde = Alpha;
		std::cout << "EigVals:\n" << Alpha.transpose() << std::endl;

		/* Orthogonalizeation via Gram-Schmidt*/
		//std::cout << "Gram-Schmidt orthogonalization \n";
		//double eNorm = EigVectsHtilde.col(0).transpose()*Mred[i] * EigVectsHtilde.col(0);
		//EigVectsHtilde.col(0) /= sqrt(eNorm);
		//for (int j = 1; j < EigVectsHtilde.cols(); ++j) {
		//	//std::cout << "col:" << j << "|" ;
		//	EigVectsHtilde.col(j) -= EigVectsHtilde.leftCols(j) * (EigVectsHtilde.leftCols(j).transpose()*Mred[i]*EigVectsHtilde.col(j));
		//	
		//	eNorm = EigVectsHtilde.col(j).transpose()*Mred[i] * EigVectsHtilde.col(j);
		//	EigVectsHtilde.col(j) /= sqrt(eNorm);
		//}
		// Xh is equal to EigVectsHtilde at this stage

		// [4] Algorithm 2 step 3: coarse grid eigensolve in the enriched space
		///std::cout << "Combining the reduced and enrichment \n";		
		///XHh.resize(EigVectsHtilde.rows(), EigVectsHbar.cols() + EigVectsHtilde.cols());
		///XHh.leftCols(EigVectsHbar.cols()) =  EigVectsHbar;
		///XHh.rightCols(EigVectsHtilde.cols()) = EigVectsHtilde;
		///std::cout << "left:" << EigVectsHbar.col(1).transpose()*Mred[i] * EigVectsHbar.col(1) << std::endl;
		///std::cout << "right:" << EigVectsHtilde.col(1).transpose()*Mred[i] * EigVectsHtilde.col(1) << std::endl;
		///printf("Red: %dx%d | enrichment: %dx%d | combine: %dx%d \n", EigVectsHbar.rows(), EigVectsHbar.cols(), EigVectsHtilde.rows(), EigVectsHtilde.cols(), XHh.rows(), XHh.cols() );
		///
		///std::cout << "Solving eigen problem in enriched space \n";
		///std::cout << "__[1] projection.\n";
		///Sproj = XHh.transpose()*Sred[i] * XHh;
		///Mproj = XHh.transpose()*Mred[i] * XHh;
		///
		///std::cout << "__[2] Solving reduced eigenproblem: ";
		///status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
		///std::cout << status << std::endl;
		///std::cout << "EigVals:\n" << Alpha.transpose() << std::endl;
		///
		///std::cout << "__[3] Prolongation \n";
		///EigVectsHbar = XHh * Q.topLeftCorner(Q.rows(), numEigs);

		/* Step 4: COARSE LEVEL WITH ENRICHED SPACE */
		Eigen::SparseMatrix<double> M00;
		Eigen::MatrixXd M01, M10, M11;
		Eigen::SparseMatrix<double> S00;
		Eigen::MatrixXd S01, S10, S11;

		M00 = Mred[i+1];
		M01 = U[i].transpose()*Mred[i]*EigVectsHtilde;
		M10 = EigVectsHtilde.transpose()*Mred[i]*U[i];
		M11 = EigVectsHtilde.transpose()*Mred[i]*EigVectsHtilde;
		printf("MH: %dx%d | Uh: %dx%d | Ph:%dx%d \n", Mred[i+1].rows(), Mred[i+1].cols(), EigVectsHtilde.rows(), EigVectsHtilde.cols(), U[i].rows(), U[i].cols());

		S00 = Sred[i+1];
		S01 = U[i].transpose()*Sred[i] * EigVectsHtilde;
		S10 = EigVectsHtilde.transpose()*Sred[i] * U[i];
		S11 = EigVectsHtilde.transpose()*Sred[i] * EigVectsHtilde;

		Eigen::MatrixXd SHh(S00.rows()+S10.rows(), S00.cols()+S01.cols()), 
						MHh(M00.rows()+M10.rows(), M00.cols()+M01.cols());
		MHh.topLeftCorner(M00.rows(), M00.cols())		= M00.toDense();
		MHh.bottomLeftCorner(M10.rows(), M10.cols())	= M10;
		MHh.topRightCorner(M01.rows(), M01.cols())		= M01;
		MHh.bottomRightCorner(M11.rows(), M11.cols())	= M11;
		SHh.topLeftCorner(S00.rows(), S00.cols())		= S00.toDense();
		SHh.bottomLeftCorner(S10.rows(), S10.cols())	= S10;
		SHh.topRightCorner(S01.rows(), S01.cols())		= S01;
		SHh.bottomRightCorner(S11.rows(), S11.cols())	= S11;

		printf("SHh:%dx%d (S00:%dx%d | S01:%dx%d | S11:%dx%d) \n", SHh.rows(), SHh.cols(), S00.rows(), S00.cols(), S01.rows(), S01.cols(), S11.rows(), S11.cols());
		std::cout << SHh.topLeftCorner(10, 10) << std::endl;
		std::cout << S00.topLeftCorner(10, 10) << std::endl;
		Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(SHh);
		int rank = lu_decomp.rank();
		printf("Matrix rank: %d | size: %dx%d \n", rank, SHh.rows(), SHh.cols());			

		//computeEigenGPUDense(SHh, MHh, EigVectsHbar, EigValsHbar);
		computeEigenMatlabDense(SHh, MHh, numEigs, EigVectsHbar, EigValsHbar, "");
		std::cout  << "Matlab solver \n" << EigValsHbar.transpose() << std::endl;
		EigVectsHbar.conservativeResize(Eigen::NoChange, numEigs);
		EigVectsHbar = EigVectsHtilde * EigVectsHbar;
		EigValsHbar.conservativeResize(numEigs);


		t2 = chrono::high_resolution_clock::now();
		duration = t2 - t1;
		std::cout << "   in " << duration.count() << " seconds." << std::endl;
	}

	EigValsRef = EigValsHbar.head(numEigs);
	EigVectsRef = EigVectsHtilde;

	Eigen::VectorXd conv2;
	checkConvergenceCat2(EigVectsRef, EigValsRef, numEigs, M, M, S, conv2);


	std::cout << "Write the log to files." << std::endl;
	string modelName = "Model_" + to_string(V.rows()) + "_" + to_string(V.cols());
	string fileConv = "../../Data/" + modelName + "_BoostrapMG_" + to_string(numEigs) + "_method" + ".txt";
	std::cout << "File path: " << fileConv << std::endl;
	writeDenseMatrixToFile(EigValsRef, fileConv);
	fileConv = "../../Data/" + modelName + "_BoostrapMG_Convergence2_" + to_string(numEigs) + "_method" + ".txt";
	writeDenseMatrixToFile(conv2, fileConv);
}

void MLSolverFunctions::BMGiter(const Eigen::SparseMatrix<double>& SL, const Eigen::SparseMatrix<double>& ML, const Eigen::SparseMatrix<double>& MLinv, Eigen::MatrixXd& EigVects, Eigen::VectorXd& EigVals, int numEigs, double mu, double tol, int maxIter, int k)
{
	chrono::high_resolution_clock::time_point	t0, t1, t2;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

}
	/**
	*  @brief
	*   Diffusion distance using the spectrum and eigenvector of the Laplace--Beltrami eigenproblem.
	*  @param	numEigs				Number of eigenpairs (eigenvalues and eigenvectors) used for computing diffusion distance
	*  @param	t					Diffusion parameter
	*  @param	EigVect				Matrix containing the eigenvectors, stored per column.
	*  @param	EigVals				Vector representing the eigenvalues.
	*  @param	DiffusionTensor		The diffusion tensor with the provided data.
	*/
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
	
	/* [VISUALIZATION TOOLS] */
	/**
	*  @brief	Visualize the basis, on varying level.
	*  @param	viewer		GUI for model interaction.
	*  @param	id			ID/index of basis to show (corresponds to column of the basis matrix), mapped to full resolution using the *Prolongation* operator.
	*  @param	k			Level on the hierarchy.
	*  @param	scale		(If applicable) Ratio for scaling of the basis.
	*/
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
	
	/**
	*  @brief	Visualize the selected eigenvector corresponding to a certain eigenvalue.
	*  @param	viewer		GUI for model interaction.
	*  @param	id			ID/index of eigenvector to show (i.e. the ID of the corresponding eigenvalue).
	*  @param	eigType		Type of the eigenvector to show, from MuLES::MLEigenSolver::EigenVectorType enum type.
	*  @param	scale		(If applicable) Ratio for scaling of the basis.
	*/
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
}
