#include "MLSolverElastics.h"

/* Volumetric stuff */
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/barycenter.h>
#include <igl/volume.h>

#include "Hessian.h"

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

/* Some getters */
Eigen::MatrixXi MLSolverElastics::getTF() const { return TF; }
Eigen::MatrixXd MLSolverElastics::getVB() const { return VB; }