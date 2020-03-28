#include "MLEigenSolver.h"

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

void MLEigenSolver::constructSampleHierarchy(const Eigen::VectorXd& samples)
{
	printf("> Sampling on % levels \n", samples.size());
	for (int i = 0; i < samples.size(); ++i){
		SubspaceSize[i] = samples(i);
		printf("Level %d -> Subspace size: %d\n", i, SubspaceSize[i]);
	}
	sampleSize = SubspaceSize[0];

	SampleMap.resize(V.rows());
	constructFarthestPointSample(V, AdM, Sample, SubspaceSize, geoDist);
	cout << "Assigning level \n";
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

/* Visualization tools */
void MLEigenSolver::visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k) const
{
	Eigen::MatrixXd VColor;
	igl::jet(geoDist[k], true, VColor);
	viewer.data().set_colors(VColor);
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