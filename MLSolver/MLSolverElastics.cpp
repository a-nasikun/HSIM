#include "MLSolverElastics.h"
#include "Hessian.h"

#include <queue>
#include <Eigen/PardisoSupport>

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


void MLSolverElastics::constructBasis_LBS(double support, int axis, double offset)
{
	double			totalVolume = 0.0;
	for (int i = 0; i < F.rows(); i++) totalVolume += (volume(i));

	maxNeighDist[0] = cbrt((3.0*3.0*support*totalVolume) / (4.0*SubspaceSize[0] * M_PI));
	printf("Radius %d is %.4f\n", 0, maxNeighDist[0]);
	for (int i = 1; i < level; i++) {
		maxNeighDist[i] = cbrt(SubspaceSize[i - 1] / (double)SubspaceSize[i]) * maxNeighDist[i - 1];
		printf("Radius %d is %.4f\n", i, maxNeighDist[i]);
	}

	for (int i = 0; i < level; i++) {
		constructRadialBasisFunctions_LBS(i, axis, isClampedBasis, offset, support);
		printf("basis level %d: done \n", i);
	}

	formPartitionOfUnityMultiLevel();
	Basis.resize(3 * V.rows(), 3 * V.rows());
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

void MLSolverElastics::constructBasisCoarseOnly_LBS(double support, int axis, double offset)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "___Basis construction (Coarse only)... ";
	t1 = chrono::high_resolution_clock::now();

	int coarseSample = SubspaceSize[level - 1];
	double				  maxDist;
	double				  totalVolume = volume.sum();
	maxDist = cbrt((4*3.0*3.0*support*totalVolume) / (4.0*coarseSample * M_PI));
	printf("Sample: %d with dist of %.5f total volume: %.4f \n", coarseSample, maxDist, totalVolume);

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

		// LBS = requires 4x3 (=12) DoFs on each vertex
		UTriplet[tid].reserve(4 * 3 * 2 * round((double)ipts / (double)coarseSample) * support * V.rows());

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= coarseSample) break;
			/* Dijkstra basis function */
			//ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist[0], i, SubspaceSize[0], 0, isClamped, axis, offset, UTriplet[tid], elementCounter[tid]);
			constructDijkstraDisk_LBS(Sample(i), D, aCoef, bCoef, maxDist, i, 0, axis, offset, UTriplet[tid], elementCounter[tid]);
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

	Basis.resize(3 * V.rows(), 12 * coarseSample);
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
				if (it.col() % 3 != 0) continue;		// if the col is not multiple of 3, skip
				///if (isClampedBasis && globToLocMap(it.row()) < 0) continue;
				maxVal = it.value();
				clusters[0](it.row()/3) = it.col()/3;	// the vertex of id it.row()/3 belongs to cluster it.col()/3
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
void MLSolverElastics::constructBasisNonSmooth_LBS(double support, int axis, double offset)
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
		constructRadialBasisFunctions_LBS(i, axis, isClampedBasis, offset, support);
		printf("basis level %d: done \n", i);
	}
	///formPartitionOfUnityMultiLevel();

	/* Construct clusters on each level, then create basis out of it */
	/* 1. For level 0, use the value on the 9-th index only *
	**    using the transpose of the basis matrix. */
	cout << "Clustering level 0\n";
	Eigen::SparseMatrix<double, Eigen::RowMajor> UTemp = U[0];
	for (int k = 0; k < UTemp.outerSize(); k += 3) {		// index of the row, multiple of 3 (there are 3|V| entries
		double maxVal = std::numeric_limits<double>::lowest();
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(UTemp, k); it; ++it) {
			if (it.value() > maxVal)
			{
				if (it.col() % 12 != 9) continue;
				///if (isClampedBasis && globToLocMap(it.row()) < 0) continue;
				maxVal = it.value();
				clusters[0](it.row() / 3) = it.col() / 12;
			}
		}
	}

	///* New basis */
	cout << "Set up new basis \n";
	vector<Eigen::Triplet<double>> UTriplet;
	UTriplet.reserve((12)*(3*V.rows()));
	for (int i = 0; i < V.rows(); i++)
	{
		if (isClampedBasis && globToLocMap(i) < 0) continue;
		int parentVertex = clusters[0](i);
		///for (int j = 0; j < 4; ++j) {			// for every 'column'
		///	for (int k = 0; k < 3; ++k) {		// for every 'row'
		///		UTriplet.push_back(Eigen::Triplet<double>(3*i+k, 12*parentVertex+j*3+k, U[0].coeff(3*Sample(parentVertex)+k, 12*parentVertex+j*3+k)));
		///	}
		///}
		
		// for every 'column' on the Affine transformation (3x9 matrix)
		for (int j = 0; j < 9; ++j) {			
			// for every 'row'
			for (int k = 0; k < 3; ++k) {		
				//UTriplet.push_back(Eigen::Triplet<double>(3*i+k, 12*parentVertex+j, U[0].coeff(3*Sample(parentVertex)+k, 12*parentVertex+j)));
				UTriplet.push_back(Eigen::Triplet<double>(3*i+k, 12*parentVertex+j, U[0].coeff(3*i+k, 12*parentVertex+j)));
			}
		}
		// translation space
		for (int k = 0; k < 3; ++k) {
			UTriplet.push_back(Eigen::Triplet<double>(3*i+k, 12*parentVertex+9+k, U[0].coeff(3*Sample(parentVertex)+k, 12*parentVertex+9+k)));
		}


		//for (int k = 0; k<3; k++)
		//	UTriplet.push_back(Eigen::Triplet<double>(3 * i + k, 3 * clusters[0](i) + k, 1.0));
	}
	U[0].resize(0, 0); U[0].resize(3 * V.rows(), 12 * SubspaceSize[0]);
	U[0].setFromTriplets(UTriplet.begin(), UTriplet.end());
	UTriplet.clear(); UTriplet.shrink_to_fit();

	/* 1. For level k */
	//vector<Eigen::Triplet<double>> UTriplet;
	for (int l = 1; l < level; l++)
	{
		printf("Clustering level %d \n", l);
		UTemp = U[l];
		for (int k = 0; k < UTemp.outerSize(); k += 12) {
			double maxVal = std::numeric_limits<double>::lowest();
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(UTemp, k); it; ++it) {
				if (it.value() > maxVal)
				{
					if (it.col() % 12 != 0) continue;
					///if (isClampedBasis && globToLocMap(Sample(it.row()) < 0)) continue;
					maxVal = it.value();
					clusters[l](it.row() / 12) = it.col() / 12;
				}
			}
		}

		cout << "Set up new basis \n";
		UTriplet.reserve(12 * SubspaceSize[l - 1]);
		for (int i = 0; i < SubspaceSize[l - 1]; i++)
		{
			if (isClampedBasis && globToLocMap(Sample(i)) < 0) {
				cout << "This happens | ";  continue;
			}
			for (int j = 0; j<12; ++j)
				UTriplet.push_back(Eigen::Triplet<double>(12 * i + j, 12 * clusters[l](i) + j, 1.0));
		}
		U[l].resize(0, 0); U[l].resize(12 * SubspaceSize[l - 1], 12 * SubspaceSize[l]);
		U[l].setFromTriplets(UTriplet.begin(), UTriplet.end());
		UTriplet.clear(); UTriplet.shrink_to_fit();
	} //  End of clustering

	  //Basis = U[0];
	Basis.resize(3 * V.rows(), 3 * V.rows());
	Basis.setIdentity();
	for (int i = 0; i < level; i++) {
		Basis = Basis * U[i];
		printf("Basis size: %dx%d (nnz: %d) | Loc Basis: %dx%d (nnz: %d). \n", Basis.rows(), Basis.cols(), Basis.nonZeros(), U[i].rows(), U[i].cols(), U[i].nonZeros());
	}

	formPartitionOfUnityMultiLevel();
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

void MLSolverElastics::constructRadialBasisFunctions_LBS(int k, int axis, bool isClamped, double offset, double support)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "___Basis construction [LBS] (level " << k << " )... ";
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
			UTriplet[tid].reserve(4*3 * 2 * round((double)ipts / (double)SubspaceSize[0]) * support * V.rows());
		else
			UTriplet[tid].reserve(4*3 * 2 * round((double)ipts / (double)SubspaceSize[0]) * support * SubspaceSize[k - 1]);

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= SubspaceSize[k]) break;
			/* Dijkstra basis function */
			//ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist[0], i, SubspaceSize[0], 0, isClamped, axis, offset, UTriplet[tid], elementCounter[tid]);
			constructDijkstraDisk_LBS(Sample(i), D, aCoef, bCoef, maxNeighDist[k], i, k, axis, offset, UTriplet[tid], elementCounter[tid]);
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
		U[0].resize(3 * V.rows(), 12 * SubspaceSize[k]);
	else
		U[k].resize(12 * SubspaceSize[k - 1], 12 * SubspaceSize[k]);

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

void MLSolverElastics::constructDijkstraDisk_LBS(const int s1, Eigen::VectorXd &D, const double a, const double b, double maxDist, const int col, int k, int axis, double offset, vector<Eigen::Triplet<double>> &UTriplet, int &elementCounter)
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
				double weight = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;					// face
				/* LBS weight: 3x12 matrix for each block *
				** V^T kronecker delta with 3x3 Identity matrix	*/
				// [1] related to vertex coordinates: (col 1-9)
				for (int l = 0; l < 3; ++l) {			// for each vertex' coordinate (i.e. x, y, and z)
					double vertVal = V(i, l);
					for (int j = 0; j < 3; j++) {		// 3 rows on each vertex
						UTriplet.push_back(Eigen::Triplet<double>(3 * i + j, 12*col + 3*l + j, (weight)*vertVal));
					}
				}
				// [1] the additional identity matrix (col: 10-12)
				for (int j = 0; j < 3; j++) {		// 3 rows on each vertex
					UTriplet.push_back(Eigen::Triplet<double>(3 * i + j, 12 * col + 9 + j, (weight)));
				}
			}
		}
	}
	else {
		for (int i = 0; i < SubspaceSize[k - 1]; ++i) {
			double aVal = D(Sample(i));
			if (aVal <= maxDist) {
				if (isClampedBasis && V(Sample(i), axis) < xBound) continue;

				/* Polynomial mapping (3rd Order) */
				double val = a*aVal*aVal*aVal + b*aVal*aVal + 1.0;
				for (int j = 0; j < 12; ++j) {
					UTriplet.push_back(Eigen::Triplet<double>(12 * i + j, 12 * col + j, val));
				}
			}
		}
	}
	D.setConstant(numeric_limits<double>::infinity());
}

void MLSolverElastics::constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis, double offset) 
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
			for (int j = 0; j<3; j++) {
				STriplet.push_back(Eigen::Triplet<double>(3*i+j, 3*i+j, 1.0));
			}
		}
	}

	SelMat.setFromTriplets(STriplet.begin(), STriplet.end());
	printf("__Dim of this selector matrix: %dx%d \n", SelMat.rows(), SelMat.cols());
}

void MLSolverElastics::formPartitionOfUnity()
{

	cout << "> Partition of unity on Elastics materials is called. \n";
	/* Temporary variables */
	vector<Eigen::Triplet<double>>		diagTrip;
	Eigen::SparseMatrix<double>			DiagNormMatrix;
	Eigen::VectorXd						rowSum = Eigen::VectorXd::Zero(Basis.rows());
	DiagNormMatrix.resize(Basis.rows(), Basis.rows());
	diagTrip.reserve(Basis.nonZeros());

	/* Getting the sum of every non-zero elements in a row */
	//std::cout << "Getting the sum of every non-zero elements in a row \n" << std::endl; 
	//std::cout << "Num of elements: " << SubspaceSize[level - 1] << std::endl; 
	for (int i = 0; i < SubspaceSize[level-1]; ++i) {
		//printf("Id: %d of [%d] |", i, SubspaceSize[level - 1]);
		for (int j = 0; j < 3; ++j) {
			int k = 12*i + 9+j;
			for (Eigen::SparseMatrix<double>::InnerIterator it(Basis, k); it; ++it) {
				rowSum(it.row()) += it.value();
			}
		}
	} 
	std::cout << std::endl; 

	//std::cout << "Creating a diagonal matrix. \n" << std::endl;
	/* Get the diagonal matrix, reciprocal of the total sum per row */
	for (int i = 0; i < rowSum.size(); ++i) {
		if(abs(rowSum(i))<10.0*std::numeric_limits<double>::epsilon())
			diagTrip.push_back(Eigen::Triplet<double>(i, i, 1.0));
		else 
			diagTrip.push_back(Eigen::Triplet<double>(i, i, 1.0 / rowSum(i)));
	}
	DiagNormMatrix.setFromTriplets(diagTrip.begin(), diagTrip.end());

	/* Form partition of unity */
	Basis = DiagNormMatrix * Basis;
}

void MLSolverElastics::formPartitionOfUnityMultiLevel()
{
	cout << "Partition of unity (on Elastics model) \n";
	/* Temporary variables */
	vector<Eigen::Triplet<double>>		diagTrip;
	Eigen::SparseMatrix<double>			DiagNormMatrix;
	Eigen::VectorXd						rowSum;

	for (int i = 0; i < level; ++i)
	{
		rowSum.resize(U[i].rows());
		rowSum.setZero();
		DiagNormMatrix.resize(U[i].rows(), U[i].rows());
		diagTrip.reserve(U[i].nonZeros());

		/* Getting the sum of every non-zero elements in a row */
		if (i == 0) {
			// Special treatment for level 0 (only deal with the last 3 columns on each block)
			for (int m = 0; m < SubspaceSize[0]; ++m) {
				for (int j = 0; j < 3; ++j) {
					int k = 12 * m + 9+j;
					for (Eigen::SparseMatrix<double>::InnerIterator it(U[0], k); it; ++it) {
						rowSum(it.row()) += it.value();
					}
				}
			}
		}
		else {
			// regular partition of unity on other levels
			for (int k = 0; k < U[i].outerSize(); ++k) {
				for (Eigen::SparseMatrix<double>::InnerIterator it(U[i], k); it; ++it) {
					rowSum(it.row()) += it.value();
				}
			}
		}

		/* Get the diagonal matrix, reciprocal of the total sum per row */
		for (int k = 0; k < rowSum.size(); k++) {
			// if the sum of the row is very small, then do not do anything on this row
			if (abs(rowSum(k)) < (10.0*std::numeric_limits<double>::epsilon())) {
				diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0));
				cout <<"<"<< i <<  ">Row: " << k << ", value:" << rowSum(k) << endl;
			} 
			else 
				diagTrip.push_back(Eigen::Triplet<double>(k, k, 1.0 / rowSum(k)));
		}
		DiagNormMatrix.setFromTriplets(diagTrip.begin(), diagTrip.end());

		/* Form partition of unity */
		U[i] = DiagNormMatrix * U[i];
		printf("Partition of unity level %d | Diag: %dx%d | U_k=%dx%d (%.5f) \n", i, DiagNormMatrix.rows(), DiagNormMatrix.cols(), U[i].rows(), U[i].cols(), (double) U[i].nonZeros() / U[i].rows());
		cout << "Diag: \n" << DiagNormMatrix.topLeftCorner(6, 6) << endl;

		diagTrip.clear();
		diagTrip.shrink_to_fit();
	}
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

/* Subspace Iteration methods */
void MLSolverElastics::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset)
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

	/* Performing the SIM according to user selected methods */
	if (type == SIMType::SIM_MultiLevel) {
		timingPerLevel.resize(level);
		itersPerLevel.resize(level);
		if (isWithShift) {
			performMultiLevelSIM_withShift_Draft(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift, shiftType);
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
	string modelName = "Bar-Rectangle_" + to_string(V.rows()) + "_" + to_string(F.cols());
	string fileConv = "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + modelName + "_EigValsStack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift ? "shift_" + to_string((int)shiftType) + "_" : "") + to_string(tol) + ".txt";
	writeDenseMatrixToFile(itersPerLevel, fileConv);
	writeDenseMatrixToFile(timingPerLevel, fileConv);
	writeDenseMatrixToFile(EigValsSIMStack, fileConv);
	string fileEigs = "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + modelName + "_Convergence2Stack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift ? "shift_" + to_string((int)shiftType) + "_" : "") + to_string(tol) + ".txt";
	writeDenseMatrixToFile(convergence2Stacks, fileEigs);
	fileEigs = "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + modelName + "_Convergence3Stack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift ? "shift_" + to_string((int)shiftType) + "_" : "") + to_string(tol) + ".txt";
	writeDenseMatrixToFile(convergence3Stacks, fileEigs);
}

void MLSolverElastics::performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift)
{
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	/* Initialization */
	cout << "____[1] SIM::Setup initialization of variables\n";
	int					q = max(2 * numEigs, numEigs + 8);
	Eigen::MatrixXd		Xk;
	int					status = 1;
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
	SS = SL + eps*Id;		// Adding epsilon to the diagonal

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
	double mu;
	if (isWithShift) {
		if (EigValsReduced.rows() < 1)			// if no information about shifting is available, use no shift
			factorization.compute(SS);
		else {
			mu = EigValsReduced((int)floor(0.25*q));
			factorization.compute(SL - mu*ML);
		}
	}
	else {
		factorization.compute(SS);
	}

	/* Iteration for eigenproblem */
	cout << "____[3] Iterating steps. \n";
	bool convergenceCheck;
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
		status = computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
		if (status != 0) {
			eps *= 10;
			factorization.compute(SL + eps*Id);
			std::cout << "___shifting by \n" << eps << std::endl;
			continue;
		}
		//computeEigenGPUDenseJacobi(Sproj, Mproj, Q, Alpha);
		//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
		//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);

		printf("________lifting up/prolongation.");
		Xk = XbarK*Q;

		/* Convergence Check*
		** Type 2: */
		checkConvergenceCat2(Xk, Alpha, ML, S, conv2);
		cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
		/** Type 3: */
		checkConvergenceCat3(Q, Alpha, isWithShift, mu, conv3);
		cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

		//cout << Alpha.transpose() << endl; 		

		EigValsSIMStack.col(k) = Alpha.block(0, 0, numEigs - 1, 1);
		convergence2Stacks.col(k) = conv2.block(0, 0, numEigs - 1, 1);
		convergence3Stacks.col(k) = conv3.block(0, 0, numEigs - 1, 1);

		k++;
		convergenceCheck = (k >= maxIter || (conv2(numEigs - 1)<tol));
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

void MLSolverElastics::performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
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
	double				eps = 1e-2*std::numeric_limits<double>::epsilon();
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
			double zhaoParam = 0.4;
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

void MLSolverElastics::performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift)
{
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	printf("____[1] level: %d \n", k);
	int		q = max(2 * numEigs, numEigs + 8);
	double eps = 1e6 * std::numeric_limits<double>::epsilon();
	int status;

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
		performMultiLevelSIM(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift);
		printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);

		//if (k == (level-1)) {
		std::cout << "Level (under the hood, this number is 1 lower to the real one, since the finest level is defined as level 0): " << k << std::endl;
		printf("Basis U: %dx%d \n", U[k].rows(), U[k].cols());
		std::cout << "MK: \n" << MK.topLeftCorner(10, 10) << std::endl;
		std::cout << "SK: \n" << SK.topLeftCorner(10, 10) << std::endl;
		std::cout << "Aprox. EigVals (from coarser): " << AkL.head(numEigs).transpose() << std::endl;
		std::cout << "Approx. EigVects (from coarser): \n" << XkL.topLeftCorner(10, 10) << std::endl;
		//}

		/* [3] SUBSPACE ITERATION METHOD */
		std::cout << "Dimensionality check: \n";
		printf("Basis U:%dx%d | Approx. EigVect: %dx%d \n", U[k].rows(), U[k].cols(), XkL.rows(), XkL.cols());
		XkL = U[k] * XkL;
		printf("____<3> Lifting, Init vectors: %dx%d \n", XkL.rows(), XkL.cols());

		/* Adding a small number on the diagonal, for robustness */
		Eigen::SparseMatrix<double> SS;
		Eigen::SparseMatrix<double> Id; Id.resize(SK.rows(), SK.cols()); Id.setIdentity();
		//double eps = 1e-6;

		SS = SK + eps*Id;		// Adding epsilon to the diagonal
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
		factorization.compute(SS);
		//factorization.compute(SK);
		cout << "SS: \n" << SS.topLeftCorner(12, 6) << endl;
		printf("____<4> Embedding a small-valued diagonal matrix ");
		std::cout << eps << std::endl;

		/* Temporary variables */
		bool convergenceCheck = false;
		Eigen::MatrixXd XbarK(XkL.rows(), q); XbarK.setZero();
		Eigen::MatrixXd Mproj, Sproj;
		Eigen::MatrixXd Q;
		//Eigen::VectorXd Alpha;
		Eigen::VectorXd conv2, conv3;


		/* Iterating*/
		printf("____<5> Subspace iteration.\n");
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
			status = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
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
			cout << "_____ [" << iterCount + 1 << "] convergence2 check: \n" << conv2.head(numEigs).transpose() << endl;

			checkConvergenceCat3(Q, AkL, isWithShift, -eps, conv3);
			cout << "_____ [" << iterCount + 1 << "] convergence3 check: \n" << conv3.head(numEigs).transpose() << endl;


			/* Store the eigenvalues and convergence rate if it's on the finest level */
			if (k == 0) {
				EigValsSIMStack.col(iterCount) = AkL.block(0, 0, numEigs - 1, 1);
				convergence2Stacks.col(iterCount) = conv2.block(0, 0, numEigs - 1, 1);
				convergence3Stacks.col(iterCount) = conv3.block(0, 0, numEigs - 1, 1);
			}

			iterCount++;
			convergenceCheck = (iterCount >= maxIter || (conv2(numEigs - 1)<tol));

		} while (!convergenceCheck);
		printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());


		/* Print some data when it's on the finest level */
		if (k == 0) {
			cout << "____<7> Storing the convergence data. \n";
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

void MLSolverElastics::performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
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
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;

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

void MLSolverElastics::performMultiLevelSIM_withShiftOnce(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
{
	std::cout << "General" << std::endl;
	std::cout << "Functions" << std::endl;

	chrono::high_resolution_clock::time_point	t0, t1;
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
		printf("____Size of the initial vectors: %dx%d \n", XkL.rows(), XkL.cols());
	}
	// RECURSIVE for level k < max_level
	else
	{
		/* 1: RESTRICTION */
		printf("____<1> Restriction: ");
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

		printf("____<2> Going deeper => \n ################");
		performMultiLevelSIM_withShiftOnce(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift, shiftType);

		printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);

		/* [3] SUBSPACE ITERATION METHOD */
		XkL = U[k] * XkL;
		printf("____<3> Lifting, Init vectors: %dx%d \n", XkL.rows(), XkL.cols());

		/* Adding a small number on the diagonal, for robustness */
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;

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
		double zhaoParam = 0.25;
		int zhaoShiftId;
		double mu;
		double eps = 1e3*std::numeric_limits<double>::epsilon();

		printf("____<5> Subspace iteration.\n");
		do {
			if (status != 0)
			{
				/* [4] Zhao's proposal */
				//zhaoShiftId = lastConvergedId + min(5, (int)floor(zhaoParam*(numEigs - lastConvergedId)));
				zhaoShiftId = lastConvergedId + (int)floor(zhaoParam*(numEigs - lastConvergedId));
				if (factCounter == 0) {
					mu = AkL(zhaoShiftId);
					std::cout << "___Zhao's aggressive shift proposal | shifting on index " << zhaoShiftId << std::endl;
					factorization.compute(SK - mu*MK);
				}
				else {
					eps *= 10;
					mu = eps;
					factorization.compute(SK + mu*Id);
				}

				factCounter++;
				std::cout << "__> Factorization " << factCounter << " with shift of " << mu << std::endl;
			}

			/* Stage III: Iterations */
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
			status = computeEigenGPUDense(Sproj, Mproj, Q, AkL);
			cout << "__STATUS: " << status << endl;
			if (status != 0) {
				cout << "___!! COmputation of eigenvalues failed! \n";
				cout << "____Status: " << status << endl;
				//i = Imax;
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
			if (iterCount >= maxIter || (conv2(numEigs - 1) < tol)) {
				isConverged = true;
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

/* Visualization Tools*/
void MLSolverElastics::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, double scale) const
{
	cout << "Visualizing volumetric basis \n";
	Eigen::MatrixXd TVNew(V.rows(), V.cols());
	TVNew.setZero();

	///if (isClampedBasis) {
	///	Eigen::VectorXd BUp = SelMatrix * Basis.col(id);
	///	for (unsigned int ti = 0; ti < V.rows(); ti++) {
	///		TVNew(ti, 0) = BUp(3 * ti + 0);
	///		TVNew(ti, 1) = BUp(3 * ti + 1);
	///		TVNew(ti, 2) = BUp(3 * ti + 2);
	///	}
	///}
	///else 
	{
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