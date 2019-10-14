#include "FastSpectrum.h"
#include <Eigen/PardisoSupport>
#include <igl/grad.h>

/* [MAIN FUNCTIONS IN FAST APPROXIMATIONG ALGORITHM] */


// Constructor: String mesh name + No predefined sample type + No Basis call + approx eigenbasis
// INPUT: String of mesh-location & number of samples/subpsace, OUTPUT: approximated lifted eigenvectors & eigenvalues)
void FastSpectrum::computeEigenPairs(const string &meshFile, const int &numOfSamples, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals)
{
	computeEigenPairs(meshFile, numOfSamples, Sample_Farthest_Point, Basis, reducedEigVects, reducedEigVals);
	approxEigVects = this->Basis * this->reducedEigVects;
	approxEigVals = this->reducedEigVals;
}

// Constructor: String mesh name + No predefined sample type + Basis call + reduced eigenbasis
// INPUT: String of mesh-location & number of samples/subpsace, OUTPUT: Basis & reduced eigenvectors & eigenvalues )
void FastSpectrum::computeEigenPairs(const string &meshFile, const int &numOfSamples, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals)
{
	computeEigenPairs(meshFile, numOfSamples, Sample_Farthest_Point, Basis, reducedEigVects, reducedEigVals);
}

// Constructor: String mesh name + With predefined sample type + No Basis call + reduced eigenbasis
// INPUT: String of mesh-location & number of samples/subpsace & type of sampling, OUTPUT: approximated lifted eigenvectors & eigenvalues)
void FastSpectrum::computeEigenPairs(const string &meshFile, const int &numOfSamples, SamplingType sampleType, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals)
{	
	computeEigenPairs(meshFile, numOfSamples, sampleType, Basis, reducedEigVects, reducedEigVals);
	approxEigVects = this->Basis * this->reducedEigVects;
	approxEigVals = this->reducedEigVals;
}

/* Sub-main function (the one that called the most important version of this family of overloaded functions */
// Constructor: String mesh name + With predefined sample type + with Basis call + reduced eigenbasis
// INPUT: String of mesh-location & number of samples/subpsace & type of sampling, OUTPUT: Basis & reduced eigenvectors & eigenvalues)
void FastSpectrum::computeEigenPairs(const string &meshFile, const int &numOfSamples, SamplingType sampleType, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals)
{
	readMesh(meshFile, V, F);
	if (V.rows() < 1) return;
	computeEigenPairs(V, F, numOfSamples, sampleType, Basis, reducedEigVects, reducedEigVals);
}

// Constructor: Vertices & Faces + No predefined sample type + No Basis call + reduced eigenbasis
// INPUT: V and F matrices & number of samples/subpsace, OUTPUT: approximated lifted eigenvectors & eigenvalues)
void FastSpectrum::computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals)
{
	computeEigenPairs(V, F, numOfSamples, Sample_Farthest_Point, Basis, reducedEigVects, reducedEigVals);
	approxEigVects	= this->Basis * this->reducedEigVects;
	approxEigVals	= this->reducedEigVals;
}

// Constructor: Vertices & Faces + No predefined sample type + Basis call + reduced eigenbasis
// INPUT: V and F matrices & number of samples/subpsace, OUTPUT: OUTPUT: Basis & reduced eigenvectors & eigenvalues )
void FastSpectrum::computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals)
{
	computeEigenPairs(V, F, numOfSamples, Sample_Farthest_Point, Basis, reducedEigVects, reducedEigVals);
}
 
// Constructor: Vertices & Faces + With predefined sample type + No Basis call + reduced eigenbasis
// INPUT: V and F matrices & number of samples/subpsace & type of sampling, OUTPUT: approximated lifted eigenvectors & eigenvalues)
void FastSpectrum::computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, SamplingType sampleType, Eigen::MatrixXd &approxEigVects, Eigen::VectorXd &approxEigVals)
{
	computeEigenPairs(V, F, numOfSamples, sampleType, Basis, reducedEigVects, reducedEigVals);
	approxEigVects = this->Basis*this->reducedEigVects;
	approxEigVals = this->reducedEigVals;
}

/* Most important version of the overloaded family of this function calls */
// Constructor: Vertices & Faces + With predefined sample type + with Basis call + reduced eigenbasis
// INPUT: V and F matrices & number of samples / subpsace & type of sampling, OUTPUT : Basis & reduced eigenvectors & eigenvalues)
void FastSpectrum::computeEigenPairs(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const int &numOfSamples, SamplingType sampleType, Eigen::SparseMatrix<double> &Basis, Eigen::MatrixXd &reducedEigVects, Eigen::VectorXd &reducedEigVals)
{
	// [1.2] Constructing Laplacian Matrices (Stiffness and Mass-Matrix)
	this->V = V;
	this->F = F;
	constructLaplacianMatrix();

	// [2]	SAMPLING
	this->setSample(numOfSamples, sampleType);
	constructSample();

	// [3]	BASIS CONSTRUCTION
	double	distRatio = sqrt(pow(0.7, 2) + pow(0.7, 2));
	maxNeighDist = distRatio * sqrt((double)V.rows() / (double)Sample.size()) * avgEdgeLength;
	constructBasis();

	// [4]	LOW-DIM PROBLEM
	constructRestrictedProblem();

	// [5]	SOLVING LOW-DIM EIGENPROBLEM
	solveRestrictedProblem();

	// OUTPUT
	Basis = this->Basis;
	reducedEigVects = this->reducedEigVects;
	reducedEigVals = this->reducedEigVals;
}

/* [ENCAPSULATION]*/
/* [Construct the stiffness & mass matrix AND other important tables] */
void FastSpectrum::constructLaplacianMatrix(){
	/* Construct Stiffness matrix S */
	igl::cotmatrix(V, F, S);
	S = -S;

	/* Construct Mass matrix M*/
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

	/* Constructing Average length of the edges */
	avgEdgeLength = igl::avg_edge_length(V, F);

	/* Constructing Adjacency Matrix (neighborhood) */
	initiateAdM(V, F, AdM);

	/* Constructing table of edge-length using sparse Matrix (faster than vector-of-map and vector-of-unordered map) */
	initiateDistTableSpM(V, AdM, DistanceTableSpM);

	/* Construct face-based mass matrix*/
	Eigen::VectorXd doublArea;
	igl::doublearea(V, F, doublArea);
	vector<Eigen::Triplet<double>> M1Triplet; M1Triplet.reserve(3*F.rows());
	vector<Eigen::Triplet<double>> M2Triplet; M2Triplet.reserve(3*F.rows());
	for (int i = 0; i < F.rows(); i++) {
		double area = doublArea(i) / 2.0;
		for (int j = 0; j < 3; j++) {
			M1Triplet.push_back(Eigen::Triplet<double>(3 * i + j, 3 * i + j, area));
			M2Triplet.push_back(Eigen::Triplet<double>(3 * i + j, 3 * i + j, 1.0/area));
		}
	}

	Mf.resize(3 * F.rows(), 3 * F.rows());
	MfInv.resize(3 * F.rows(), 3 * F.rows());
	Mf.setFromTriplets(M1Triplet.begin(), M1Triplet.end());
	MfInv.setFromTriplets(M2Triplet.begin(), M2Triplet.end());

	/* Get the Gradient Matrix + Re-arrange*/
	Eigen::SparseMatrix<double> GTemp;
	igl::grad(V, F, GTemp);
	vector<Eigen::Triplet<double>> GTriplet; GTriplet.reserve(GTemp.nonZeros());
	for (int k = 0; k < GTemp.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(GTemp, k); it; ++it) {
			GTriplet.push_back(Eigen::Triplet<double>(it.row() % F.rows() + (int)(it.row()/F.rows()), it.col(), it.value()));
		}
	}
	G.resize(GTemp.rows(), GTemp.cols());
	G.setFromTriplets(GTriplet.begin(), GTriplet.end());

	printf("A Stiffness matrix S(%dx%d) and Mass matrix M(%dx%d) are constructed.\n", S.rows(), S.cols(), M.rows(), M.cols());
}

/* [Construct samples for the subspace] */
void FastSpectrum::constructSample(){
	int nBox;
	if (sampleSize < 1 || sampleSize>V.rows()) {
		cout << "Error! Sample cannot be 0 or larger than the number of vertices." << endl;
		return;
	}else if (sampleSize <= 300) {
		nBox = 8;
	}
	else if (sampleSize > 300 && sampleSize <= 500) {
		nBox = 10;
	}
	else if (sampleSize > 500 && sampleSize <= 2000) {
		nBox = 13;
	}
	else if (sampleSize > 2000 && sampleSize <= 3000) {
		nBox = 15;
	}
	else if (sampleSize > 3000 && sampleSize <= 6000) {
		nBox = 18;
	}
	else if (sampleSize > 6000 && sampleSize <= 10000) {
		nBox = 23;
	}
	else {
		cout << "Warning! Are you sure to have this huge amount of sampling?." << endl;
		cout << "Our approach won't have much benefit with this large sampling, though." << endl;
	}

	/* Different type of sampling */
	switch (sampleType)
	{
	case Sample_Poisson_Disk:
		constructPoissonDiskSample(V, sampleSize, nBox, avgEdgeLength, Sample);
		break;
	case Sample_Farthest_Point:
		constructFarthestPointSample(V, AdM, sampleSize, Sample);
		break;
	case Sample_Random:
		constructRandomSample(Sample, V, sampleSize);
		break;
	default:
		break;
	}	
	this->sampleSize = Sample.size();

	/* Setting up hierarchy*/
	level = 3;
	SubspaceSize.push_back(Sample.size());
	SubspaceSize.push_back(max(250, (int)Sample.size() / 4));
	SubspaceSize.push_back(max(100, (int)Sample.size() / 16));
	printf("A set of %d samples are constructed.\n", sampleSize);
	for (int i = 0; i < level; i++)
		printf("__level %d: %d samples \n", i, SubspaceSize[i]);
}

void FastSpectrum::constructBasis() {
	double	distRatio = sqrt(pow(1.1, 2) + pow(0.7, 2));			// A heuristic to make the support around 10.00 (i.e. number of non-zeros per row)
	maxNeighDist = distRatio * sqrt((double)V.rows() / (double)Sample.size()) * avgEdgeLength;
	constructBasisFunctions();
	formPartitionOfUnity();
}

void FastSpectrum::constructRestrictedProblem() {
	S_ = Basis.transpose() * S  * Basis;
	M_ = Basis.transpose() * M  * Basis;

	printf("A set of reduced stiffness and mass matrix (each %dx%d) is constructed.\n", S_.rows(), M_.cols());
}

void FastSpectrum::solveRestrictedProblem() {
	computeEigenPair();
	normalizeReducedEigVects();
}

/* [GETTER AMD SETTER] */
void FastSpectrum::getV(Eigen::MatrixXd &V)
{
	V = this->V;
}
void FastSpectrum::getF(Eigen::MatrixXi &F)
{
	F = this->F;
}


void FastSpectrum::getVF(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
	getV(V);
	getF(F);
}

void FastSpectrum::getAvgEdgeLength(double &avgEdgeLength)
{
	avgEdgeLength = this->avgEdgeLength;
}

void FastSpectrum::getMassMatrix(Eigen::SparseMatrix<double> &M) 
{
	M = this->M;
}

void FastSpectrum::getStiffnessMatrix(Eigen::SparseMatrix<double> &S)
{
	S = this->S; 
}

void FastSpectrum::getLaplacian(Eigen::SparseMatrix<double> &S, Eigen::SparseMatrix<double> &M)
{
	getStiffnessMatrix(S);
	getMassMatrix(M);
}

void FastSpectrum::getSamples(Eigen::VectorXi &Sample)
{
	Sample = this->Sample;
}

void FastSpectrum::getSampleSize(int &sampleSize)
{
	sampleSize = this->sampleSize;
}

void FastSpectrum::getFunctionBasis(Eigen::SparseMatrix<double> &Basis)
{
	Basis = this->Basis;
}

void FastSpectrum::getReducedEigVals(Eigen::VectorXd &reducedEigVals)
{
	reducedEigVals = this->reducedEigVals;
}

void FastSpectrum::getReducedEigVects(Eigen::MatrixXd &reducedEigVects)
{
	reducedEigVects = this->reducedEigVects;
}

void FastSpectrum::getApproxEigVects(Eigen::MatrixXd &approxEigVects)
{
	approxEigVects = Basis*reducedEigVects;
}

void FastSpectrum::getReducedStiffnessMatrix(Eigen::SparseMatrix<double> &Sbar)
{
	Sbar = this->S_;
}

void FastSpectrum::getReducedMassMatrix(Eigen::SparseMatrix<double> &Mbar)
{
	Mbar = this->M_;
}

void FastSpectrum::getReducedLaplacian(Eigen::SparseMatrix<double> &SBar, Eigen::SparseMatrix<double> &MBar)
{
	getReducedStiffnessMatrix(SBar);
	getReducedMassMatrix(MBar);
}

void FastSpectrum::setV(const Eigen::MatrixXd &V){
	this->V = V;
}
void FastSpectrum::setF(const Eigen::MatrixXi &F){
	this->F = F;
}

void FastSpectrum::setMesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
	setV(V);
	setF(F);
}

void FastSpectrum::setMesh(const string &meshFile) 
{
	readMesh(meshFile, V, F);
	setMesh(V, F);
}

void FastSpectrum::setSample(const int &sampleSize, SamplingType sampleType)
{
	setSampleNumber(sampleSize);
	setSampleType(sampleType);
}

void FastSpectrum::setSampleNumber(const int &sampleSize)
{
	this->sampleSize = sampleSize;
}

void FastSpectrum::setSampleType(SamplingType sampleType)
{
	this->sampleType = sampleType; 
}

/* Heat Method*/
void FastSpectrum::solveHeatMethod(igl::opengl::glfw::Viewer &viewer)
{
	double tau = avgEdgeLength*avgEdgeLength;
	Eigen::VectorXd inputHeat(V.rows()); inputHeat.setZero(); inputHeat(0) = 1.0; 
	Eigen::VectorXd outputHeat;
	Eigen::VectorXd grad;
	Eigen::VectorXd heatDistance;

	solveHeatEquation(tau, inputHeat, outputHeat);
	grad = G*outputHeat;
	normalizeGradient(grad);
	solvePoissonEquation(grad, heatDistance);
	visualizeHeatGeodesics(viewer, heatDistance);

	//testGaussSeidel();
}

void FastSpectrum::solveHeatEquation(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Solving heat equation (smoothing)... ";
	t1 = chrono::high_resolution_clock::now();
	

	Eigen::SparseMatrix<double> LHS = M + tau*S;
	Eigen::VectorXd rhs = M*input; 

	/// PARDISO (Direct Solver)
	//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> sparseSolver;
	//sparseSolver.analyzePattern(LHS);
	//sparseSolver.factorize(LHS);
	//output = sparseSolver.solve(rhs);

	/// Iterative Solver
	GaussSeidelSparseSolver(LHS, rhs, output, 1000, 1e-5);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}

void FastSpectrum::normalizeGradient(Eigen::VectorXd& grad)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Normalize gradient... ";
	t1 = chrono::high_resolution_clock::now();
	
	Eigen::Vector3d localGrad;
	for (int i = 0; i < F.rows(); i++)
	{
		localGrad = grad.block(3 * i, 0, 3, 1);
		localGrad.normalize();
		grad.block(3 * i, 0, 3, 1) = localGrad;
	}

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}

void FastSpectrum::solvePoissonEquation(const Eigen::VectorXd& input, Eigen::VectorXd& output)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Solve poission equation... ";
	t1 = chrono::high_resolution_clock::now();

	Eigen::VectorXd rhs = G.transpose()*Mf*input; 

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> SparseSolver;
	SparseSolver.analyzePattern(S);
	SparseSolver.factorize(S);
	output = SparseSolver.solve(rhs);

	///GaussSeidelSparseSolver(S, rhs, output, 1000, 0.0003);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}

void FastSpectrum::visualizeHeatGeodesics(igl::opengl::glfw::Viewer &viewer, const Eigen::VectorXd& functions)
{
	Eigen::MatrixXd VColor;
	igl::jet(functions, true, VColor);
	viewer.data().set_colors(VColor);
}

/* Reduced (1-level) Heat Method */
void FastSpectrum::solveHeatMethodRed(igl::opengl::glfw::Viewer &viewer)
{
	double tau = avgEdgeLength*avgEdgeLength;
	Eigen::VectorXd inputHeat(V.rows()); inputHeat.setZero(); inputHeat(0) = 1.0;
	Eigen::VectorXd outputHeat;
	Eigen::VectorXd grad;
	Eigen::VectorXd heatDistance;

	prepareRedMatrix();
	inputHeat = Basis.transpose()*(inputHeat);
	solveHeatEquationRed(tau, inputHeat, outputHeat);
	outputHeat = Basis*outputHeat;
	grad = G*outputHeat;
	normalizeGradientRed(grad);
	solvePoissonEquationRed(grad, heatDistance);
	//heatDistance = Basis*heatDistance;
	visualizeHeatGeodesicsRed(viewer, heatDistance);

}

void FastSpectrum::prepareRedMatrix()
{
	M_ = Basis.transpose()*M*Basis;
	S_ = Basis.transpose()*S*Basis;
	GTMf_ = Basis.transpose()*G.transpose()*Mf;
}

/* Both input and output are in reduced space */
void FastSpectrum::solveHeatEquationRed(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output)
{
	Eigen::SparseMatrix<double> MS_ = M_ + tau*S_;
	Eigen::VectorXd rhs = M_*input;

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> sparseSolver;
	sparseSolver.analyzePattern(MS_);
	sparseSolver.factorize(MS_);
	output = sparseSolver.solve(rhs);	
}
void FastSpectrum::normalizeGradientRed(Eigen::VectorXd& grad)
{
	normalizeGradient(grad);
}

/* Input: full resolution, output: reduced */
void FastSpectrum::solvePoissonEquationRed(const Eigen::VectorXd& input, Eigen::VectorXd& output)
{
	Eigen::VectorXd rhs = GTMf_ * input;

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> sparseSolver;
	sparseSolver.analyzePattern(S_);
	sparseSolver.factorize(S_);
	output = sparseSolver.solve(rhs);
}
void FastSpectrum::visualizeHeatGeodesicsRed(igl::opengl::glfw::Viewer &viewer, const Eigen::VectorXd& functions)
{
	Eigen::VectorXd funcLifted = Basis*functions;

	Eigen::MatrixXd VColor;
	igl::jet(funcLifted, true, VColor);
	viewer.data().set_colors(VColor);
}

void FastSpectrum::testGaussSeidel()
{	
	Eigen::SparseMatrix<double> A(4, 4);
	A.coeffRef(0, 0) = 10;
	A.coeffRef(1, 0) = -1;
	A.coeffRef(2, 0) = 2;
	A.coeffRef(3, 0) = 0;
	A.coeffRef(0, 1) = -1;
	A.coeffRef(1, 1) = 11;
	A.coeffRef(2, 1) = -1;
	A.coeffRef(3, 1) = 3;
	A.coeffRef(0, 2) = 2;
	A.coeffRef(1, 2) = -1;
	A.coeffRef(2, 2) = 10;
	A.coeffRef(3, 2) = -1;
	A.coeffRef(0, 3) = 0;
	A.coeffRef(1, 3) = 3;
	A.coeffRef(2, 3) = -1;
	A.coeffRef(3, 3) = 8;

	Eigen::VectorXd b(4);
	b << 6, 25, -11, 15;

	Eigen::VectorXd x0,x;
	x0.setZero(4);

	//GaussSeidel1Iter(A, b, x0, x);
	GaussSeidelSparseSolver(A, b, x, 100, 0.0001);
}

/* [Read mesh model from file] */
void FastSpectrum::readMesh(const string &meshFile, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
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

/* [Construct Basis Matrix] */
void FastSpectrum::constructBasisFunctions()
{
	Basis.resize(V.rows(), Sample.size());
	Basis.reserve(20 * V.rows());

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef = 2.0 / (maxNeighDist*maxNeighDist*maxNeighDist);
	double			bCoef = -3.0 / (maxNeighDist*maxNeighDist);
	int				i = 0;
	Eigen::VectorXi	Si(1);

	vector<Eigen::Triplet<double>>	emptyTriplet;
	const int						NUM_OF_THREADS = omp_get_num_procs();
	omp_set_num_threads(NUM_OF_THREADS);
	UTriplet.resize(NUM_OF_THREADS);

	int tid, ntids, ipts, istart, iproc;
	chrono::high_resolution_clock::time_point	t4, t5;
	chrono::duration<double>					time_span4;

	vector<int> elementCounter;
	elementCounter.resize(NUM_OF_THREADS);
	int totalElementInsertions = 0;

	/* Manually break the chunck for parallelization */
#pragma omp parallel private(tid,ntids,ipts,istart,i, t4,t5,time_span4)
	{
		t4		= chrono::high_resolution_clock::now();
		iproc	= omp_get_num_procs();
		tid		= omp_get_thread_num();
		ntids	= omp_get_num_threads();
		ipts	= (int)ceil(1.00*(double)sampleSize / (double)ntids);
		istart	= tid * ipts;
		if (tid == ntids - 1) ipts = sampleSize - istart;
		if (ipts <= 0) ipts = 0;

		// Instantiate D once, used for all Dijkstra
		Eigen::VectorXd				D(V.rows());
		for (int i = 0; i < V.rows(); i++) {
			D(i) = numeric_limits<double>::infinity();
		}
		
		UTriplet[tid].reserve(2.0 * ((double)ipts / (double)sampleSize) * 20.0 * V.rows());

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= sampleSize) break;
			/* Dijkstra basis function */
			ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist, i, sampleSize, UTriplet[tid], elementCounter[tid]);
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
	Basis.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	printf("A basis matrix (%dx%d) is constructed.\n", Basis.rows(), Basis.cols());
}

/* [Form partition of unity for the basis matrix Basis] */
void FastSpectrum::formPartitionOfUnity()
{
	/* Temporary variables */
	vector<Eigen::Triplet<double>>		diagTrip;
	Eigen::SparseMatrix<double>			DiagNormMatrix;
	Eigen::VectorXd						rowSum = Eigen::VectorXd::Zero(Basis.rows());
	DiagNormMatrix.resize(Basis.rows(), Basis.rows());

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

/* [Compute the eigenpairs of Low-Dim Problem ] */
void FastSpectrum::computeEigenPair(/*Eigen::SparseMatrix<double> &S_, Eigen::SparseMatrix<double> &M_, Eigen::MatrixXd &LDEigVec, Eigen::VectorXd &LDEigVal*/)
{
	computeEigenGPU(S_, M_, reducedEigVects, reducedEigVals);
}

/* We do this because the result of eigensolver generally orthonormal to each other in low dim,
** but not w.r.t. mass matrix M, */
void FastSpectrum::normalizeReducedEigVects()
{
	Eigen::SparseMatrix<double> UTMU;
	UTMU = Basis.transpose() * M *Basis;
	double mNorm;
	int i = 0;

	for (i = 0; i < reducedEigVects.cols(); i++) {
		mNorm = reducedEigVects.col(i).transpose() * UTMU * reducedEigVects.col(i);
		mNorm = (double) 1.0 / sqrt(mNorm);
		reducedEigVects.col(i) = mNorm * reducedEigVects.col(i);
	}
	printf("A set of %d eigenpairs is computed.\n", reducedEigVects.cols());
}