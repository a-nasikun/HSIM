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
	Eigen::VectorXd sampSize(level); 
	for (int i = 0; i < level; i++) sampSize(i) = (int) (numOfSamples / (double) (pow(2, 2*i)));
	constructSampleHierarchy(sampSize);
	//constructSample();

	// [3]	BASIS CONSTRUCTION
	double	distRatio = sqrt(pow(0.7, 2) + pow(0.7, 2));
	maxNeighDist[0] = distRatio * sqrt((double)V.rows() / (double)Sample.size()) * avgEdgeLength;
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

void FastSpectrum::setInitialSystem(int n)
{
	level = n;
	maxNeighDist.resize(level);
	U.resize(level);
	Sbar.resize(level);
	Mbar.resize(level);
	SubspaceSize.resize(level);
	geoDist.resize(level);
	clusters.resize(level);
	//sampleLevel.resize()
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

	/* Compute Surface area of the triangle meshes */
	surfaceArea = computeSurfaceArea(V, F);

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

void FastSpectrum::constructSampleHierarchy(Eigen::VectorXd& n)
{

	for (int i = 0; i < level; i++)
	{
		SubspaceSize[i] = n(i);
		printf("Level %d -> sample size: %d\n", i, SubspaceSize[i]);
	}

	//constructSample();
	SampleMap.resize(V.rows()); SampleMap.setConstant(-1);
	constructFarthestPointSample(V, AdM, Sample, SubspaceSize, geoDist);

	cout << "Assigning level \n";
	sampleLevel.resize(V.rows());
	sampleLevel.setConstant(-1);
	// for level n-1
	for (int i = 0; i < SubspaceSize[level - 1]; i++)
	{
		sampleLevel(Sample(i)) = level - 1;
		SampleMap(Sample(i)) = i;
	}

	for (int i = level - 2; i >= 0; i--) 
	{
		for (int j = SubspaceSize[i+1]; j < SubspaceSize[i]; j++)
		{
			sampleLevel(Sample(j)) = i; 
			SampleMap(Sample(j)) = j;
		}
	}

	/* TEsting in which level (lowest one) a sample belongs */
	//for (int j = level-2; j >= 0; j--)
	//{
	//	for (int i = 0; i < 10; i++)
	//	{
	//		printf("sample(%d)=%d => level %d \n", SubspaceSize[j+1] + i, Sample(SubspaceSize[j+1] + i), sampleLevel(Sample(SubspaceSize[j+1]+i)));
	//	}
	//}
	//
	//for (int i = 0; i < 10; i++)
	//{
	//	printf("sample(%d)=%d => level %d \n", i, Sample(i), sampleLevel(Sample(i)));
	//}

}

/* [Construct samples for the subspace] */
void FastSpectrum::constructSample()
{
	constructFarthestPointSample(V, AdM, sampleSize, Sample);
	this->sampleSize = Sample.size();
}

void FastSpectrum::constructBasis() {
	//double	distRatio = sqrt(pow(1.1, 2) + pow(0.7, 2));			// A heuristic to make the support around 10.00 (i.e. number of non-zeros per row)
	//maxNeighDist[0] = distRatio * sqrt((double)V.rows() / (double)Sample.size()) * avgEdgeLength;
	const double NUM_SUPPORT = 10.0;
	for (int i = 0; i < level; i++) {
		maxNeighDist[i] = sqrt((surfaceArea*NUM_SUPPORT) / (M_PI*SubspaceSize[i]));
		printf("Radius %d is %.4f \n", i, maxNeighDist[i]);
	}
	
	constructBasisFunctionsLevel0();
	for (int i = 1; i < level; i++)	{
		constructBasisFunctionsLevelK(i);
		printf("Construction of basis level %d is done \n", i);
	}

	formPartitionOfUnityMultiLevel();

	Basis = U[0] * U[1] * U[2];
	//Basis = U[0];
	//formPartitionOfUnity();

	printf("Basis sparsity:  %.4f (%d/%d) \n", (double) Basis.nonZeros() / (double) Basis.rows(), Basis.nonZeros(), Basis.rows());
}

void FastSpectrum::constructBasisCoarseOnly()
{
	constructBasisFunctionsCoarseOnly();

	formPartitionOfUnity();

	printf("Basis sparsity:  %.4f (%d/%d) \n", (double)Basis.nonZeros() / (double)Basis.rows(), Basis.nonZeros(), Basis.rows());
}

void FastSpectrum::constructNonSmoothBasis()
{
	/* Resizing the cluster */
	cout << "Resizing the clusters \n";
	clusters[0].resize(V.rows());
	for (int i = 1; i < level; i++) clusters[i].resize(SubspaceSize[i-1]);

	BuildLocalClusterLevel0();
	///for (int i = 1; i < level; i++)
	///	BuildLocalClusterLevelK(i);
}

void FastSpectrum::constructRestrictedProblem() {
	S_ = Basis.transpose() * S  * Basis;
	M_ = Basis.transpose() * M  * Basis;

	printf("A set of reduced stiffness and mass matrix (each %dx%d) is constructed.\n", S_.rows(), M_.cols());
	printf("__S red %.4f (%d nnz)\n", (double)S_.nonZeros() / (double)S_.rows(), S_.nonZeros());
	printf("__M red %.4f (%d nnz)\n", (double)M_.nonZeros() / (double)M_.rows(), M_.nonZeros());
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
	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> sparseSolver;
	sparseSolver.analyzePattern(LHS);
	sparseSolver.factorize(LHS);
	output = sparseSolver.solve(rhs);

	/// Iterative Solver
	//GaussSeidelSparseSolver(LHS, rhs, output, 1000, 1e-5);

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
	//double tau = 1000.0*avgEdgeLength;
	double tau = avgEdgeLength*avgEdgeLength;
	Eigen::VectorXd inputHeat(V.rows()); inputHeat.setZero(); inputHeat(0) = 1.0;
	Eigen::VectorXd outputHeat;
	Eigen::VectorXd grad;
	Eigen::VectorXd heatDistance;

	prepareRedMatrix();
	inputHeat = Basis.transpose()*(M*inputHeat);
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
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Prepare reduced matrices (M, S, GTMf)... ";
	t1 = chrono::high_resolution_clock::now();

	M_ = Basis.transpose()*M*Basis;
	S_ = Basis.transpose()*S*Basis;
	GTMf_ = Basis.transpose()*G.transpose()*Mf;

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}

/* Both input and output are in reduced space */
void FastSpectrum::solveHeatEquationRed(double tau, const Eigen::VectorXd& input, Eigen::VectorXd& output)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Solve heat equation... ";
	t1 = chrono::high_resolution_clock::now();

	Eigen::SparseMatrix<double> MS_ = M_ + tau*S_;
	//Eigen::VectorXd rhs = M_*input;
	Eigen::VectorXd rhs = input;

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> sparseSolver;
	sparseSolver.analyzePattern(MS_);
	sparseSolver.factorize(MS_);
	output = sparseSolver.solve(rhs);	

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}
void FastSpectrum::normalizeGradientRed(Eigen::VectorXd& grad)
{
	normalizeGradient(grad);
}

/* Input: full resolution, output: reduced */
void FastSpectrum::solvePoissonEquationRed(const Eigen::VectorXd& input, Eigen::VectorXd& output)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Solve poission equation... ";
	t1 = chrono::high_resolution_clock::now();

	Eigen::VectorXd rhs = GTMf_ * input;

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> sparseSolver;
	sparseSolver.analyzePattern(S_);
	sparseSolver.factorize(S_);
	output = sparseSolver.solve(rhs);

	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;
}
void FastSpectrum::visualizeHeatGeodesicsRed(igl::opengl::glfw::Viewer &viewer, const Eigen::VectorXd& functions)
{
	Eigen::VectorXd funcLifted = Basis*functions;

	Eigen::MatrixXd VColor;
	igl::jet(funcLifted, true, VColor);
	viewer.data().set_colors(VColor);
}

/* Supporting functions for non-smooth basis */
void FastSpectrum::BuildLocalClusterLevel0()
{
	printf("Buildign clusters level 0 \n");
	Eigen::VectorXi visited(V.rows());
	visited.setConstant(-1);
	for (int k = 0; k < SubspaceSize[0]; k++)
	{
		
		{	// Dijkstra-like system			
			priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistPQueue;			

			// data on sample point k
			int s1 = Sample(k);
			geoDist[0](s1) = 0.0f;
			VertexPair vp{ s1, geoDist[0](s1) };
			DistPQueue.push(vp);
			visited(s1) = 1;
			clusters[0](s1) = k;

			while (DistPQueue.size()>0)
			{
				VertexPair vp1 = DistPQueue.top();
				double distCtr = geoDist[0](vp1.vId);
				visited(vp1.vId) = 1;
				DistPQueue.pop();


				for (set<int>::iterator it = AdM[vp1.vId].begin(); it != AdM[vp1.vId].end(); ++it) 
				{
					double distCur = geoDist[0](*it);

					if (distCur > distCtr && visited(*it)<1) {
						VertexPair vp2{ *it,distCur};
						DistPQueue.push(vp2);
						clusters[0](*it) = k;
						visited(*it) = 1;
					}
				}
			}
		}
	}
}
void FastSpectrum::BuildLocalClusterLevelK(int l)
{
	printf("Buildign clusters level %d", l);
	Eigen::VectorXi visited(V.rows());
	visited.setConstant(-1);

	for (int k = 0; k < SubspaceSize[l]; k++)
	{
		{
			priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistPQueue;

			// data on sample point k
			int s1 = Sample(k);
			geoDist[l](s1) = 0.0f;
			VertexPair vp{ s1, geoDist[l](s1) };
			DistPQueue.push(vp);
			visited(s1) = 1;
			clusters[l](s1) = k;

			while (DistPQueue.size() > 0)
			{
				VertexPair vp1 = DistPQueue.top();
				double distCtr = geoDist[l](vp1.vId);
				visited(vp1.vId) = 1;
				DistPQueue.pop();

				for (set<int>::iterator it = AdM[vp1.vId].begin(); it != AdM[vp1.vId].end(); ++it)
				{
					double distCur = geoDist[l](*it);

					if (distCur > distCtr && visited(*it)<1) {
						VertexPair vp2{ *it,distCur };
						DistPQueue.push(vp2);
						
						visited(*it) = 1;

						// Only set values for selected sample
						if (sampleLevel(*it) <= l && sampleLevel(*it) >= 0)
						{
							printf("Need to write sample: %d (id=%d) to cluster %d \n", SampleMap(*it), *it, k);
							clusters[l](SampleMap(*it)) = k;
						}
					}
				}
			}	// END of WHILE loop
		}		// END of useless bracket
	}			// END of FOR loop
	cout << " is over \n";
}

void FastSpectrum::visualizeClusterOfSamples(igl::opengl::glfw::Viewer &viewer)
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

void FastSpectrum::visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k)
{
	Eigen::MatrixXd VColor;
	igl::jet(geoDist[k], true, VColor);
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

void FastSpectrum::constructBasisFunctionsCoarseOnly()
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Basis construction (Coarse only)... ";
	t1 = chrono::high_resolution_clock::now();

	//Basis.resize(V.rows(), Sample.size());
	//Basis.reserve(20 * V.rows());
	int coarseSample = SubspaceSize[SubspaceSize.size() - 1];
	
	cout << "__Getting the correct size\n";

	double NUM_SUPPORT = 10.0;
	double maxDist = sqrt((NUM_SUPPORT*surfaceArea) / (M_PI*coarseSample));

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef = 2.0 / (maxDist * maxDist * maxDist);
	double			bCoef = -3.0 / (maxDist * maxDist);
	int				i = 0;

	const int						NUM_OF_THREADS = omp_get_num_procs();
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

		UTriplet[tid].reserve(2.0 * ((double)ipts / (double)coarseSample) * 20.0 * V.rows());

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= coarseSample) break;
			/* Dijkstra basis function */
			ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxDist, i, coarseSample, UTriplet[tid], elementCounter[tid]);
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

	Basis.resize(V.rows(), coarseSample);
	Basis.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	// Clearing up stuff
	AllTriplet.clear(); AllTriplet.shrink_to_fit();
	for (int j = 0; j < NUM_OF_THREADS; j++) { UTriplet[j].clear(); UTriplet[j].shrink_to_fit(); }

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;

	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", Basis.rows(), Basis.cols(), (double) (Basis.nonZeros()/ (double) Basis.rows()));
	//printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", U[0].rows(), U[0].cols(), (double)(U[0].nonZeros() / (double)U[0].rows()));

}

/* [Construct Basis Matrix] */
void FastSpectrum::constructBasisFunctionsLevel0()
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Basis construction (level 0)... ";
	t1 = chrono::high_resolution_clock::now();

	//Basis.resize(V.rows(), Sample.size());
	//Basis.reserve(20 * V.rows());

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef = 2.0 / (maxNeighDist[0]*maxNeighDist[0]*maxNeighDist[0]);
	double			bCoef = -3.0 / (maxNeighDist[0]*maxNeighDist[0]);
	int				i = 0;
	
	const int						NUM_OF_THREADS = omp_get_num_procs();
	omp_set_num_threads(NUM_OF_THREADS);
	UTriplet.resize(NUM_OF_THREADS);

	int tid, ntids, ipts, istart, iproc;

	vector<int> elementCounter;
	elementCounter.resize(NUM_OF_THREADS);
	int totalElementInsertions = 0;

	/* Manually break the chunck for parallelization */
#pragma omp parallel private(tid,ntids,ipts,istart,i)
	{
		iproc	= omp_get_num_procs();
		tid		= omp_get_thread_num();
		ntids	= omp_get_num_threads();
		ipts	= (int)ceil(1.00*(double)sampleSize / (double)ntids);
		istart	= tid * ipts;
		if (tid == ntids - 1) ipts = sampleSize - istart;
		if (ipts <= 0) ipts = 0;

		// Instantiate D once, used for all Dijkstra
		Eigen::VectorXd				D(V.rows());
		D.setConstant(numeric_limits<double>::infinity());
		//for (int i = 0; i < V.rows(); i++) {
		//	D(i) = numeric_limits<double>::infinity();
		//}
		
		UTriplet[tid].reserve(2.0 * ((double)ipts / (double)sampleSize) * 20.0 * V.rows());

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= sampleSize) break;
			/* Dijkstra basis function */
			ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist[0], i, sampleSize, UTriplet[tid], elementCounter[tid]);
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

	U[0].resize(V.rows(), SubspaceSize[0]);
	U[0].setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;

	//printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", Basis.rows(), Basis.cols(), (double) (Basis.nonZeros()/ (double) Basis.rows()));
	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", U[0].rows(), U[0].cols(), (double)(U[0].nonZeros() / (double)U[0].rows()));
}

void FastSpectrum::constructBasisFunctionsLevelK(int k)
{
	// For Timing
	chrono::high_resolution_clock::time_point	t1, t2;
	chrono::duration<double>					duration;
	cout << "> Basis construction (level 0)... ";
	t1 = chrono::high_resolution_clock::now();

	vector<vector<Eigen::Triplet<double>>>	UTriplet;
	vector<Eigen::Triplet<double>>			AllTriplet;
	int				eCounter = 0;
	double			aCoef = 2.0 / (maxNeighDist[k] * maxNeighDist[k] * maxNeighDist[k]);
	double			bCoef = -3.0 / (maxNeighDist[k] * maxNeighDist[k]);
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

		UTriplet[tid].reserve(2.0 * ((double)ipts / (double)SubspaceSize[k]) * 20.0 * SubspaceSize[k-1]);

		for (i = istart; i < (istart + ipts); i++) {
			if (i >= SubspaceSize[k]) break;
			/* Dijkstra basis function */
			ComputeDijkstraCompact(V, Sample(i), AdM, D, aCoef, bCoef, maxNeighDist[k], i, k, Sample, SubspaceSize, UTriplet[tid], elementCounter[tid]);
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
	} /* End of OpenMP parallelization for combining triplet */

	// timing
	t2 = chrono::high_resolution_clock::now();
	duration = t2 - t1;
	cout << "in " << duration.count() << " seconds" << endl;

	U[k].resize(SubspaceSize[k-1], SubspaceSize[k]);
	U[k].setFromTriplets(AllTriplet.begin(), AllTriplet.end());

	printf("A basis matrix (%dx%d) is constructed, with %.2f average support.\n", U[k].rows(), U[k].cols(), (double)(U[k].nonZeros() / (double)U[k].rows()));
}

/* [Form partition of unity for the basis matrix Basis] */
void FastSpectrum::formPartitionOfUnity()
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

void FastSpectrum::formPartitionOfUnityMultiLevel()
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
			//cout << "Reading col " << k << endl; 
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