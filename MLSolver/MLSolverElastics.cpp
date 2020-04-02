#include "MLSolverElastics.h"
#include "Hessian.h"

#include <queue>

/* Volumetric stuff */
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/barycenter.h>
#include <igl/volume.h>

/**
*  @brief Creating a tet-mesh from current surface (triangle) mesh.
*
*  @details
*   Creating a tet-mesh from current surface (triangle) mesh,
*	the 'V' and 'F' variables are initially binded to description of the surface mesh,
*   i.e. vertices and face of the surface mesh,
*	but not it is binded to those of tetrahedron mesh.
*	the face ID is now referred as TF 
*	Perhaps not the best naming configuration, but it serves well for OOP purpose of this tool.
*  @param meshFile  Location of the mesh
*/
void MLSolverElastics::createTetrahedralMesh()
{
	cout << "> Tetrahedralization of input triangle mesh: \n";	
	printf("____ Input: V=%dx%d | F=%dx%d \n", V.rows(), V.cols(), F.rows(), F.cols());

	igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.05Y", V, F, TF);
	//igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.114Y", TV, TT, TF);
	igl::barycenter(V, F, VB);
	
	printf("____ Output: TV=%dx%d | TF=%dx%d | TT=%dx%d\n", V.rows(), V.cols(), TF.rows(), TF.cols(), F.rows(), F.cols());
}

/**
*  @brief Computing the volume of each tet in of the model.
*
*  @details
*   Computing the volume of each tet in of the model.
*/
void MLSolverElastics::computeVolume()
{
	cout << "> Computing volume of each tetrahedral mesh\n";
	igl::volume(V, F, volume);
}

void MLSolverElastics::constructMassMatrix()
{
	cout << "Creating mass matrix for the tet-mesh \n";
	vector<Eigen::Triplet<double>> Mriplet, MhNegTriplet;
	Mriplet.reserve(3 * V.rows());
	MhNegTriplet.reserve(3 * V.rows());
	M.resize(3 * V.rows(), 3 * V.rows());
	MhNeg.resize(3 * V.rows(), 3 * V.rows());
	// material density
	double rho = 1.0;							
	double toVol = 0;
	/* Iterating over all tet-mesh element */
	for (int i = 0; i < F.rows(); i++) {
		double vol = volume(i);
		double mass = rho*vol / 4.0;
		toVol += vol;

		/* iterating over each node/vertex on the tetmesh element (4 per element) */
		for (int el = 0; el < 4; el++) {
			/* iterating for 3 coordinates (3 per node/vertex: x,y,z) */
			for (int coor = 0; coor < 3; coor++) {
				Mriplet.push_back(Eigen::Triplet<double>(3 * F(i, el) + coor, 3 * F(i, el) + coor, mass));

			} // end of each coordinate
		}	  // end of vertex list
	}		  // end of mesh elements list

	M.setFromTriplets(Mriplet.begin(), Mriplet.end());

	/* Set the hal-inverse mass matrix [M^(-0.5)] */
	for (int k = 0; k < M.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
			MhNegTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), 1.0 / sqrt(it.value())));
		}
	}
	MhNeg.setFromTriplets(MhNegTriplet.begin(), MhNegTriplet.end());
	printf("Mass matrix %dx%d with %d non-zero entries \n", M.rows(), M.cols(), M.nonZeros());
	cout << "Some part of the mass matrix: \n" << M.topLeftCorner(10, 10) << endl << endl;
	cout << "Some part of the neg matrix: \n" << MhNeg.topLeftCorner(10, 10) << endl << endl;

	/* Checking mass matrix */
	double massTot = 0;
	for (int k = 0; k < M.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
			massTot += it.value();
		}
	}
	printf("The total volume is %.4f and the diag of mass matrix is %.4f \n", toVol, massTot);
}

void MLSolverElastics::setLameParametersConstant(double lame1, double lame2)
{
	Lame.resize(F.rows(), 2);
	Lame.col(0).setConstant(lame1);
	Lame.col(1).setConstant(lame2);
}

void MLSolverElastics::setLameParametersSlices(int axis, int numSlices, double m1l1, double m1l2, double m2l1, double m2l2, Eigen::VectorXd& vColor)
{
	cout << "> Constructing matrix of Lame parameters. \n";
	Lame.resize(F.rows(), 2);
	vColor.resize(V.rows());

	/* Setting the category of slicing */
	Eigen::VectorXd matSliceMap(numSlices);
	for (int i = 0; i < numSlices; i++) {
		matSliceMap(i) = (i % 2);
	}

	/* Divion of material, in user selected-axis */
	double minCoord = V.col(axis).minCoeff();
	double partition = (V.col(axis).maxCoeff() - minCoord) / (double)numSlices;

	/* For coloring the mesh */
	vColor.setZero();
	if (numSlices == 1) {
		vColor.setConstant(0.7);
	}
	else {
		for (int ii = 0; ii < V.rows(); ii++) {
			int whichChunk = floor((V(ii, axis) - minCoord) / (double)partition);
			if (whichChunk == numSlices) whichChunk = numSlices - 1;

			if (matSliceMap(whichChunk) == 0) vColor(ii) = 0.3;
			else vColor(ii) = 0.7;
		}
	}

	/* Computing the Lame parameters*/
	if (numSlices == 1) {
		Lame.col(0).setConstant(m1l1);
		Lame.col(1).setConstant(m1l2);
	}
	/* Division of material for more than 1 slices */
	else {
		for (int i = 0; i < F.rows(); i++) {
			int whichChunk = floor((VB(i, axis) - minCoord) / (double)partition);
			if (whichChunk == numSlices) whichChunk == numSlices - 1;

			if (matSliceMap(whichChunk) == 0) {
				Lame(i, 0) = m1l1;
				Lame(i, 1) = m1l2;				// half is one type of material
			}
			else {
				Lame(i, 0) = m2l1;
				Lame(i, 1) = m2l2;				// the other half is different material
			}
		}
	}

}

void MLSolverElastics::setLameParametersBox(int axis, double clampRatio, double m1l1, double m1l2, double m2l1, double m2l2, Eigen::VectorXd& vColor)
{
	cout << "> Constructing matrix of Lame parameters\n";
	Lame.resize(F.rows(), 2);
	vColor.resize(V.rows());

	/* Divion of material, in user selected-axis */
	double partition = (V.col(axis).maxCoeff() - V.col(axis).minCoeff()) * clampRatio;
	Eigen::Vector3d minBoundVect, maxBoundVect;
	for (int i = 0; i < 3; i++) {
		minBoundVect(i) = V.col(i).minCoeff() + partition;
		maxBoundVect(i) = V.col(i).maxCoeff() - partition;
	}


	/* For coloring the mesh */
	//vColor.setZero();
	vColor.setConstant(0.3);
	for (int i = 0; i < V.rows(); i++) {
		if ((V(i, 0) < minBoundVect(0) || V(i, 0) > maxBoundVect(0)) && (V(i, 1) < minBoundVect(1) || V(i, 1) > maxBoundVect(1) || V(i, 2) < minBoundVect(2) || V(i, 2) > maxBoundVect(2))) {
			vColor(i) = 0.7;
		}
		else if ((V(i, 1) < minBoundVect(1) || V(i, 1) > maxBoundVect(1)) && (V(i, 0) < minBoundVect(0) || V(i, 0) > maxBoundVect(0) || V(i, 2) < minBoundVect(2) || V(i, 2) > maxBoundVect(2))) {
			vColor(i) = 0.7;
		}
		else if ((V(i, 2) < minBoundVect(2) || V(i, 2) > maxBoundVect(2)) && (V(i, 0) < minBoundVect(0) || V(i, 0) > maxBoundVect(0) || V(i, 1) < minBoundVect(1) || V(i, 1) > maxBoundVect(1))) {
			vColor(i) = 0.7;
		}
	}

	/* SeFing Lame parameters */
	Lame.col(0).setConstant(m1l1);
	Lame.col(1).setConstant(m1l2);
	for (int i = 0; i < VB.rows(); i++) {
		if ((VB(i, 0) < minBoundVect(0) || VB(i, 0) > maxBoundVect(0)) && (VB(i, 1) < minBoundVect(1) || VB(i, 1) > maxBoundVect(1) || VB(i, 2) < minBoundVect(2) || VB(i, 2) > maxBoundVect(2))) {
			Lame(i, 0) = m2l1;
			Lame(i, 1) = m2l2;
		}
		else if ((VB(i, 1) < minBoundVect(1) || VB(i, 1) > maxBoundVect(1)) && (VB(i, 0) < minBoundVect(0) || VB(i, 0) > maxBoundVect(0) || VB(i, 2) < minBoundVect(2) || VB(i, 2) > maxBoundVect(2))) {
			Lame(i, 0) = m2l1;
			Lame(i, 1) = m2l2;
		}
		else if ((VB(i, 2) < minBoundVect(2) || VB(i, 2) > maxBoundVect(2)) && (VB(i, 0) < minBoundVect(0) || VB(i, 0) > maxBoundVect(0) || VB(i, 1) < minBoundVect(1) || VB(i, 1) > maxBoundVect(1))) {
			Lame(i, 0) = m2l1;
			Lame(i, 1) = m2l2;
		}
	}
}

void MLSolverElastics::constructStiffnessMatrix()
{
	cout << "Construction the hessian matrix \n";
	S.resize(3*V.rows(), 3*V.rows());

	/* Construct the HESSIAN for every TET-Mesh then DISTRIBUTE to each vertex entry */
	vector<Eigen::Triplet<double>> KTriplet;
	KTriplet.reserve(12 * 12 * F.rows());

	chrono::high_resolution_clock::time_point	t0, t1, t2;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	/* Parallelizing the hessian construction */
	vector<vector<Eigen::Triplet<double>>>		KiTriplet;
	const int									NUM_OF_THREADS = 8; // omp_get_num_procs() / 2;
	omp_set_num_threads(NUM_OF_THREADS);
	KiTriplet.resize(NUM_OF_THREADS);

	int tid, ntids, ipts, istart, iproc, i = 0;

	//for (unsigned int i = 0; i < F.rows(); i++) {
#pragma omp parallel private(i,tid, ntids,ipts,istart) // ADOLC_OPENMP //private(ADOLC_OPENMP) //
	{
		// dividing the stuff per chunk/block
		iproc	= omp_get_num_procs();
		tid		= omp_get_thread_num();
		ntids	= omp_get_num_threads();
		ipts	= (int)ceil(1.00*(double)F.rows() / (double)ntids);
		istart	= tid * ipts;
		if (tid == ntids - 1) ipts = F.rows() - istart;
		if (ipts <= 0) ipts = 0;

		KiTriplet[tid].reserve(12 * 12 * F.rows() / (ntids));

		// Doing iteration per thread of the CPU
		for (i = istart; i < (istart + ipts); i++)
		{
			/* __ Computing the hessian energy + energy value for each tet-mesh */
			Eigen::MatrixXd Hess;
			
			computeHessian_Linear(Lame(i, 0), Lame(i, 1), V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)), V.row(F(i, 3)), Hess);
			if (i == 0)
				cout << "First hessian: \n" << Hess << endl;

			/* Constructing the entries for the GLOBAL Hessian matrix */
			for (unsigned int hi = 0; hi < 4; hi++) {				// for every COL block of the 12x12 matrix
				for (unsigned int hj = 0; hj < 4; hj++) {			// for every ROW block of the 12x12 matrix					
					for (unsigned int ti = 0; ti < 3; ti++) {
						for (unsigned int tj = 0; tj < 3; tj++) {
							KiTriplet[tid].push_back(Eigen::Triplet<double>(3 * F(i, hj) + tj, 3 * F(i, hi) + ti, 0.25*volume(i)*Hess(3 * hj + tj, 3 * hi + ti)));
						}
					}
				}
			}
		}
	}	// End of paralleization

		// Collcting Entries
	int totalElements = 0;
	for (int j = 0; j < NUM_OF_THREADS; j++) {
		totalElements += KiTriplet[j].size();
	}

	/* Combining the Triplet */
	KTriplet.resize(totalElements);
	//#pragma omp parallel 
	{
		for (int j = 0; j < NUM_OF_THREADS; j++) {
			int tripSize = 0;
			for (int k = 0; k < j; k++) {
				tripSize += KiTriplet[k].size();
			}
			std::copy(KiTriplet[j].begin(), KiTriplet[j].end(), KTriplet.begin() + tripSize);
		}
	}

	printf("Triplet size: %d |", KTriplet.size());
	S.setFromTriplets(KTriplet.begin(), KTriplet.end());
	printf("S: %dx%d \n", S.rows(), S.cols());
}

void MLSolverElastics::constructBasis(double support, int axis, double offset)
{
	double			totalVolume = 0.0;
	for (int i = 0; i < F.rows(); i++) totalVolume += (volume(i));	

	maxNeighDist[0] = cbrt((3.0*3.0*support*totalVolume) / (4.0*SubspaceSize[0] * M_PI));
	printf("Radius %d is %.4f\n", 0, maxNeighDist[0]);
	for (int i = 1; i < level; i++) {		
		maxNeighDist[i] = cbrt(SubspaceSize[i-1]/(double)SubspaceSize[i]) * maxNeighDist[i-1];
		printf("Radius %d is %.4f\n", i, maxNeighDist[i]);
	}
		
	for (int i = 0; i < level; i++) {
		constructRadialBasisFunctions(i, axis, isClampedBasis, offset, support);
		printf("basis level %d: done \n", i);
	}

	formPartitionOfUnityMultiLevel();
	Basis.resize(3*V.rows(), 3*V.rows());
	Basis.setIdentity();
	for (int i = 0; i < level; i++)
		Basis = Basis * U[i];

	printf("Basis sparsity:  %.4f (%d/%d) \n", (double)Basis.nonZeros() / (double)Basis.rows(), Basis.nonZeros(), Basis.rows());

}

void MLSolverElastics::constructBasisCoarseOnly(double support, int axis, double offset)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "___Basis construction (Coarse only)... ";
	t1 = chrono::high_resolution_clock::now();

	int coarseSample	= SubspaceSize[level-1];
	double				  maxDist;
	double				  totalVolume = volume.sum();
	maxDist				= cbrt((3.0*3.0*support*totalVolume) / (4.0*coarseSample * M_PI));
	printf("Sample: %d with dist of %.5f total volume: %.4f \n", coarseSample, maxDist, totalVolume);

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef =  2.0 / (pow(maxDist, 3));
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

	Basis.resize(3 * V.rows(), 3 * coarseSample);
	Basis.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;

	///injectRigidModesInBasis();

	formPartitionOfUnity();

	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", Basis.rows(), Basis.cols(), (double)(Basis.nonZeros() / (double)Basis.rows()));

}

void MLSolverElastics::constructBasisNonSmooth(double support, int axis, double offset)
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
		clusters[i].resize(SubspaceSize[i - 1]);

	double			totalVolume = 0.0;
	for (int i = 0; i < F.rows(); i++) totalVolume += (volume(i));

	maxNeighDist[0] = cbrt((3.0*3.0*support*totalVolume) / (4.0*SubspaceSize[0] * M_PI));
	printf("Radius %d is %.4f\n", 0, maxNeighDist[0]);
	for (int i = 1; i < level; i++) {
		maxNeighDist[i] = cbrt(SubspaceSize[i - 1] / (double)SubspaceSize[i]) * maxNeighDist[i - 1];
		printf("Radius %d is %.4f\n", i, maxNeighDist[i]);
	}

	for (int i = 0; i < level; i++) {
		constructRadialBasisFunctions(i, axis, isClampedBasis, offset, support);
		printf("basis level %d: done \n", i);
	}

	/* Construct clusters on each level, then create basis out of it */
	/* 1. For level 0 */
	cout << "Clustering level 0\n";
	Eigen::SparseMatrix<double, Eigen::RowMajor> UTemp = U[0];
	for (int k = 0; k < UTemp.outerSize(); k+=3) {
		double maxVal = std::numeric_limits<double>::lowest();
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(UTemp, k); it; ++it) {
			if (it.value() > maxVal)
			{
				if (it.col() % 3 != 0) continue;
				///if (isClampedBasis && globToLocMap(it.row()) < 0) continue;
				maxVal = it.value();
				clusters[0](it.row()/3) = it.col()/3;
			}
		}
	}

	/* New basis */
	cout << "Set up new basis \n";
	vector<Eigen::Triplet<double>> UTriplet; UTriplet.reserve(3*V.rows());
	for (int i = 0; i < V.rows(); i++)
	{
		if (isClampedBasis && globToLocMap(i) < 0) continue;
		for(int k=0; k<3; k++)
			UTriplet.push_back(Eigen::Triplet<double>(3*i+k, 3*clusters[0](i)+k, 1.0));
	}
	U[0].resize(0, 0); U[0].resize(3*V.rows(), 3*SubspaceSize[0]);
	U[0].setFromTriplets(UTriplet.begin(), UTriplet.end());
	UTriplet.clear(); UTriplet.shrink_to_fit();

	/* 1. For level k */
	for (int l = 1; l < level; l++)
	{
		printf("Clustering level %d \n", l);
		UTemp = U[l];
		for (int k = 0; k < UTemp.outerSize(); k+=3) {
			double maxVal = std::numeric_limits<double>::lowest();
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(UTemp, k); it; ++it) {
				if (it.value() > maxVal)
				{
					if (it.col() % 3 != 0) continue;
					///if (isClampedBasis && globToLocMap(Sample(it.row()) < 0)) continue;
					maxVal = it.value();
					clusters[l](it.row()/3) = it.col()/3;
				}
			}
		}

		cout << "Set up new basis \n";
		UTriplet.reserve(3*SubspaceSize[l-1]);
		for (int i = 0; i < SubspaceSize[l-1]; i++)
		{
			if (isClampedBasis && globToLocMap(Sample(i)) < 0) {
				cout << "This happens | ";  continue;
			}
			for(int j=0; j<3; ++j)
				UTriplet.push_back(Eigen::Triplet<double>(3*i+j, 3*clusters[l](i)+j, 1.0));
		}
		U[l].resize(0, 0); U[l].resize(3*SubspaceSize[l-1], 3*SubspaceSize[l]);
		U[l].setFromTriplets(UTriplet.begin(), UTriplet.end());
		UTriplet.clear(); UTriplet.shrink_to_fit();
	} //  End of clustering

	  //Basis = U[0];
	Basis.resize(3*V.rows(),3*V.rows());
	Basis.setIdentity();
	for (int i = 0; i < level; i++) {
		Basis = Basis * U[i];
		printf("Basis size: %dx%d (nnz: %d) | Loc Basis: %dx%d (nnz: %d). \n", Basis.rows(), Basis.cols(), Basis.nonZeros(), U[i].rows(), U[i].cols(), U[i].nonZeros());
	}
}

void MLSolverElastics::constructRadialBasisFunctions(int k, int axis, bool isClamped, double offset, double support)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "___Basis construction (level " << k << " )... ";
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
		//for (int i = 0; i < V.rows(); i++) {
		//	D(i) = numeric_limits<double>::infinity();
		//}

		if (k == 0)
			UTriplet[tid].reserve(3*2 * round((double)ipts / (double)SubspaceSize[0]) * support * V.rows());
		else
			UTriplet[tid].reserve(3*2 * round((double)ipts / (double)SubspaceSize[0]) * support * SubspaceSize[k - 1]);

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

	if (k == 0)
		U[0].resize(3*V.rows(), 3*SubspaceSize[k]);
	else
		U[k].resize(3*SubspaceSize[k-1], 3*SubspaceSize[k]);

	U[k].setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;

	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", U[k].rows(), U[k].cols(), (double)(U[k].nonZeros() / (double)U[k].rows()));

}

void MLSolverElastics::constructDijkstraDisk(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter)
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
				for (int j = 0; j < 3; j++) {
					UTriplet.push_back(Eigen::Triplet<double>(3*i+j, 3*col+j, val));
				}
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
				for (int j = 0; j < 3; j++) {
					UTriplet.push_back(Eigen::Triplet<double>(3*i+j, 3*col+j, val));
				}

			}
		}
	}
	D.setConstant(numeric_limits<double>::infinity());
}

void MLSolverElastics::injectRigidModesInBasis()
{
	const int NUM_RIGID_MODES = 6;
	const int initSize = Basis.cols();
	vector<Eigen::Triplet<double>> BTriplet;
	BTriplet.reserve(Basis.nonZeros() + 4*Basis.rows());


	/* Getting the values from old basis */
	for (int k = 0; k < Basis.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(Basis, k); it; ++it) {
			BTriplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
		}
	}

	/* Adding 6DoF rigid deformations
	** [1] the first 3 translations */
	for (int id = 0; id < 3; id++)
	{
		for (int i = 0; i < V.rows(); i++) {
			for (int j = 0; j < 3; ++j) {
				BTriplet.push_back(Eigen::Triplet<double>(3 * i + j, initSize+j, 1.0));
			}
		}
	}
	/*[2] the 3 linearized rotations */
	for (int i = 0; i < V.rows(); i++) {
		BTriplet.push_back(Eigen::Triplet<double>(3 * i + 1, initSize + 3,  V(i, 2)));	// x-axis is fixed
		BTriplet.push_back(Eigen::Triplet<double>(3 * i + 2, initSize + 3, -V(i, 1)));
		BTriplet.push_back(Eigen::Triplet<double>(3 * i + 0, initSize + 4,  V(i, 2)));	// y-axis is fixed
		BTriplet.push_back(Eigen::Triplet<double>(3 * i + 2, initSize + 4, -V(i, 0)));
		BTriplet.push_back(Eigen::Triplet<double>(3 * i + 0, initSize + 5,  V(i, 1)));	// z-axis is fixed
		BTriplet.push_back(Eigen::Triplet<double>(3 * i + 1, initSize + 5, -V(i, 0)));
	}

	/* [3] Combining them as a new basis matrix */
	Basis.resize(Basis.rows(), initSize + NUM_RIGID_MODES);
	Basis.data().clear();
	Basis.setFromTriplets(BTriplet.begin(), BTriplet.end());

}

void MLSolverElastics::injectRigidModes()
{
	const int NUM_RIGID_MODES = 3;
	Eigen::MatrixXd RigidModes(M.rows(), NUM_RIGID_MODES); RigidModes.setZero();

	/* Adding 6DoF rigid deformations
	** [1] the first 3 translations */
	//for (int i = 0; i < V.rows(); ++i) {
	//	for (int j = 0; j < V.cols(); ++j) {
	//		RigidModes(3 * i + j, j) = 1.0;
	//	}
	//}

	/* [2] the 3 linearized rotations */
	for (int i = 0; i < V.rows(); ++i) {
		RigidModes(3*i+1, 3-3) =  V(i,2);
		RigidModes(3*i+2, 3-3) = -V(i,1);
		RigidModes(3*i  , 4-3) =  V(i,2);
		RigidModes(3*i+2, 4-3) = -V(i,0);
		RigidModes(3*i  , 5-3) =  V(i,1);
		RigidModes(3*i+1, 5-3) = -V(i,0);
	}

	///for (int i = 0; i < NUM_RIGID_MODES; ++i) {
	///	double mNorm = RigidModes.col(i).transpose() * M * RigidModes.col(i);
	///	mNorm = (double) 1.0 / sqrt(mNorm);
	///	RigidModes.col(i) = mNorm * RigidModes.col(i);
	///}

	Eigen::MatrixXd M1 = RigidModes.transpose()*M*RigidModes;
	Eigen::MatrixXd S1 = RigidModes.transpose()*S*RigidModes;
	Eigen::MatrixXd A1(NUM_RIGID_MODES, NUM_RIGID_MODES);

	Eigen::LDLT<Eigen::MatrixXd> denseSolver;
	denseSolver.compute(M1);
	for (int i = 0; i < NUM_RIGID_MODES; ++i) {
		A1.col(i) = denseSolver.solve(S1.col(i));
	}

	/* eigensolver */
	Eigen::MatrixXd Q1; 
	Eigen::VectorXd A2;
	computeEigenGPUDense(S1, M1, Q1, A2);

	cout << "S: \n" << S1 << endl;
	cout << "M: \n" << M1 << endl;
	cout << "A: \n" << A1 << endl;
	cout << "A1: \n" << A2 << endl;
	cout << "Modes: \n" << RigidModes.topLeftCorner(10, NUM_RIGID_MODES) << endl; 
	cout << "Modes 2: \n" << Q1 << endl; 

	for (int i = 0; i < NUM_RIGID_MODES; ++i) {
		double mNorm = RigidModes.col(i).transpose() * M * RigidModes.col(i);
		cout << "id: " << i << ": " << mNorm << endl;
	}
	//Eigen::MatrixXd Diag(NUM_RIGID_MODES, NUM_RIGID_MODES); Diag.setIdentity();
	//RigidModes = RigidModes*Q1;/// *A2.cwiseSqrt().asDiagonal();
	cout << "Modes: \n" << RigidModes.topLeftCorner(10, NUM_RIGID_MODES) << endl;

	const int initSize = EigVectsReduced.cols();
	//EigVectsReduced.conservativeResize(Eigen::NoChange,  initSize + NUM_RIGID_MODES);
	//EigValsReduced.conservativeResize(initSize + NUM_RIGID_MODES);
	//
	//EigVectsReduced.topRightCorner(EigVectsReduced.rows(), initSize) = EigVectsReduced.topLeftCorner(EigVectsReduced.rows(), initSize);
	//EigVectsReduced.topLeftCorner(EigVectsReduced.rows(), NUM_RIGID_MODES) = RigidModes;
	//
	//EigValsReduced.tail(initSize) = EigValsReduced.head(initSize);
	//EigValsReduced.head(NUM_RIGID_MODES) = A2;

	Eigen::MatrixXd TempVects = EigVectsReduced;
	Eigen::VectorXd TempVals = EigValsReduced;

	EigVectsReduced.resize(0, 0);
	EigValsReduced.resize(0);

	EigVectsReduced.resize(M.rows(), initSize + NUM_RIGID_MODES);
	EigValsReduced.resize(initSize + NUM_RIGID_MODES);

	EigVectsReduced.topLeftCorner(M.rows(), NUM_RIGID_MODES) = TempVects.topLeftCorner(M.rows(), NUM_RIGID_MODES);
	EigVectsReduced.block(0, NUM_RIGID_MODES, M.rows(), NUM_RIGID_MODES) = RigidModes;
	EigVectsReduced.topRightCorner(M.rows(), initSize - NUM_RIGID_MODES) = TempVects.topRightCorner(M.rows(), initSize - NUM_RIGID_MODES);
	
	EigValsReduced << TempVals.head(NUM_RIGID_MODES), A2, TempVals.tail(initSize-NUM_RIGID_MODES);

	cout << "Top old vals: \n" << TempVals.head(6) << endl;
	cout << "Top new vals: \n" << EigValsReduced.head(6) << endl;
	cout << "Top old vects: \n" << TempVects.topLeftCorner(6, 6) << endl;
	cout << "Top new vects: \n" << EigVectsReduced.topLeftCorner(6, 6) << endl;	

	Eigen::VectorXd Sv = S*RigidModes.col(0);
	Eigen::VectorXd Mv = M*RigidModes.col(0);
	cout << "S*v: " << Sv.head(20).transpose() << endl;
	cout << "M*v: " << Mv.head(20).transpose() << endl;
	cout << "Ratio: " << Sv.head(20).cwiseQuotient(Mv.head(20)) << endl;

}

/* Visualization Tools*/
void MLSolverElastics::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, double scale) const
{
	Eigen::MatrixXd TVNew(V.rows(), V.cols());
	TVNew.setZero();

	if (isClampedBasis) {
		Eigen::VectorXd BUp = SelMatrix * Basis.col(id);
		for (unsigned int ti = 0; ti < V.rows(); ti++) {
			TVNew(ti, 0) = BUp(3 * ti + 0);
			TVNew(ti, 1) = BUp(3 * ti + 1);
			TVNew(ti, 2) = BUp(3 * ti + 2);
		}
	}
	else {
		for (unsigned int ti = 0; ti < V.rows(); ti++) {
			TVNew(ti, 0) = Basis.coeff(3 * ti + 0, id);
			TVNew(ti, 1) = Basis.coeff(3 * ti + 1, id);
			TVNew(ti, 2) = Basis.coeff(3 * ti + 2, id);
		}
	}

	//cout << "TV new: " << TVNew.row(0) << " vs TV old: " << simulation.TV.row(0) << endl;

	viewer.data().clear();
	viewer.data().set_mesh(scale*TVNew + V, TF);
}

void MLSolverElastics::visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale) const
{
	Eigen::MatrixXd TVNew(V.rows(), V.cols());
	Eigen::VectorXd EigVectTemp;
	TVNew.setZero();


	switch (eigType)
	{
	case EigenVectorType::EIGEN_REFERENCE:
		EigVectTemp = EigVectsRef.col(id);
		break;
	case EigenVectorType::EIGEN_REDUCED:
		EigVectTemp = EigVectsReduced.col(id);
		break;
	case EigenVectorType::EIGEN_SIM:
		EigVectTemp = EigVectsSIM.col(id);
		break;
	default:
		break;
	}
		
	for (unsigned int ti = 0; ti < V.rows(); ti++) {
		TVNew(ti, 0) = EigVectTemp(3 * ti + 0);
		TVNew(ti, 1) = EigVectTemp(3 * ti + 1);
		TVNew(ti, 2) = EigVectTemp(3 * ti + 2);
	}
	
	viewer.data().set_vertices(V+scale*TVNew);
	viewer.data().compute_normals();
	//viewer.data().clear();
	//viewer.data().set_mesh(scale*TVNew + V, TF);
}

/* Some getters */
Eigen::MatrixXi MLSolverElastics::getTF() const { return TF; }
Eigen::MatrixXd MLSolverElastics::getVB() const { return VB; }