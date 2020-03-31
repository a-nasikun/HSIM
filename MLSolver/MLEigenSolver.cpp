#include "MLEigenSolver.h"
#include <Eigen/PardisoSupport>

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
	printf("____Sampling on %d levels \n", samples.size());
	for (int i = 0; i < samples.size(); ++i){
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

void MLEigenSolver::solveRestrictedProblem()
{
	cout << "> Creating the reduced system matrices.\n";
	Eigen::SparseMatrix<double> S_, M_;			//reduced stiffness and mass matrix
	S_ = Basis.transpose()*S*Basis;
	M_ = Basis.transpose()*M*Basis;
	printf("__S red %.4f (%d nnz)\n", (double)S_.nonZeros() / (double)S_.rows(), S_.nonZeros());
	printf("__M red %.4f (%d nnz)\n", (double)M_.nonZeros() / (double)M_.rows(), M_.nonZeros());

	cout << "> Computing the reduced eigenproblem (in GPU). \n";
	computeEigenGPU(S_, M_, EigVectsReduced, EigValsReduced);

	EigVectsReduced = Basis * EigVectsReduced;
}

/* Subspace Iteration methods */
void MLEigenSolver::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift)
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
	convergenceStacks.resize(numEigs, maxIter);
	convergenceStacks.setZero();

	/* Performing the SIM according to user selected methods */
	if (type == SIMType::SIM_MultiLevel) {
		performMultiLevelSIM(S, M, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift);
	} else {
		if (isWithShift)
			performSIMwithShift_Draft(type, numEigs, maxIter, tol, isWithShift);
		else
			performSIMwithWarmStart(type, numEigs, maxIter, tol, isWithShift);
	}

	printf("Size of eigenvalues: %d | eigVectors: %dx%d \n", EigValsSIM.rows(), EigVectsSIM.rows(), EigVectsSIM.cols());

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;

	/* Write the convergence information to an external file. */
	string modelName = "Bar-Rectangle_" + to_string(V.rows()) + "_" + to_string(V.cols());
	string fileConv = "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + modelName + "_EigValsStack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + to_string(tol) + ".txt";
	writeDenseMatrixToFile(EigValsSIMStack, fileConv);
	string fileEigs = "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + modelName + "_ConvergenceStack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + to_string(tol) + ".txt";
	writeDenseMatrixToFile(convergenceStacks, fileEigs);
}

void MLEigenSolver::performSIMwithWarmStart(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift)
{
	/* Initialization */
	cout << "____[1] SIM::Setup initialization of variables\n";
	int					q = max(2*numEigs, numEigs+ 8);
	Eigen::MatrixXd		Xk;

	/* Selecting the appropriate subspace iteration method */
	bool isNormalized = false;
	switch (type)
	{
	case SIMType::SIM_1level_WarmStart:
		cout << "_______SUBSPACE ITERATION METHOD with a Warm Start.\n";
		Xk = Basis * EigVectsReduced.block(0, 0, Basis.cols(), q);
		break;
	case SIMType::SIM_1level_Random:
		cout << "_______SUBSPACE ITERATION METHOD with Random initialization Vectors (normalized).\n";
		generateRandomMatrix(S.rows(), q, Xk);

		/* M-normalized per column */
		for (int i = 0; i < q; i++) {
			double mNorm = Xk.col(i).transpose() * M * Xk.col(i);
			mNorm = (double) 1.0 / sqrt(mNorm);
			Xk.col(i) = mNorm * Xk.col(i);
		}
		break;
	case SIMType::SIM_1level_Bathe:
		cout << "________SUBSPACE ITERATION METHOD with Bathe's proposed initialization Vectors (normalized).\n";
		generateBatheInitVectors(M.rows(), q, M, Xk);

		/* M-normalized per column */
		if (isNormalized) {
			for (int i = 0; i < q; i++) {
				double mNorm = Xk.col(i).transpose() * M * Xk.col(i);
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
	double eps = 1e-6;
	Id.resize(S.rows(), S.cols());
	Id.setIdentity();
	Id = eps*Id;
	SS = S + Id;		// Adding epsilon to the diagonal

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
	double mu;
	if (isWithShift) {
		if (EigValsReduced.rows() < 1)			// if no information about shifting is available, use no shift
			factorization.compute(SS);
		else {
			mu = EigValsReduced((int)floor(0.4*q));
			factorization.compute(S - mu*M);
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
				v = M * (Xk.col(i));
			}
			XbarK.col(i) = factorization.solve(v);
		}

		/* Projection of M and K*/
		printf("________setting up reduced matrix M and S. ");		
		Sproj = XbarK.transpose()*S*XbarK;
		Mproj = XbarK.transpose()*M*XbarK;
		printf("________Mred: %dx%d | Sred: %dx%d \n", Mproj.rows(), Mproj.cols(), Sproj.rows(), Sproj.cols());

		/* Reduced eigenvalue problem */
		printf("________solve reduced eigenvalue problem. ");
		computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
		//computeEigenGPUDenseJacobi(Sproj, Mproj, Q, Alpha);
		//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
		//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);
		
		printf("________lifting up/prolongation.");
		Xk = XbarK*Q;

		/* Convergence Check*
		** Type 2: */
		checkConvergenceCat2(Xk, Alpha, M, S, conv2);
		cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
		/** Type 3: */
		checkConvergenceCat3(Q, Alpha, isWithShift, mu, conv3);
		cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

		//cout << Alpha.transpose() << endl; 		

		EigValsSIMStack.col(k) = Alpha.block(0, 0, numEigs-1, 1);
		convergenceStacks.col(k) = conv2.block(0, 0, numEigs-1, 1);

		k++;
		convergenceCheck = (k >= maxIter|| (conv2(numEigs-1)<tol));
	} while (!convergenceCheck);


	EigVectsSIM = Xk;
	EigValsSIM = Alpha;
}

void MLEigenSolver::performSIMwithShift_Draft(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift)
{
	/* Initialization */
	cout << "____[1] SIM::Setup initialization of variables\n";
	int					q = max(2 * numEigs, numEigs + 8);
	Eigen::MatrixXd		Xk;
	const int			Imax = 4;
	int					k = 0;
	int					factCounter = 0;

	/* Selecting the appropriate subspace iteration method */
	bool isNormalized = false;
	switch (type)
	{
	case SIMType::SIM_1level_WarmStart:
		cout << "_______SUBSPACE ITERATION METHOD with a Warm Start.\n";
		Xk = Basis * EigVectsReduced.block(0, 0, Basis.cols(), q);
		break;
	case SIMType::SIM_1level_Random:
		cout << "_______SUBSPACE ITERATION METHOD with Random initialization Vectors (normalized).\n";
		generateRandomMatrix(S.rows(), q, Xk);

		/* M-normalized per column */
		for (int i = 0; i < q; i++) {
			double mNorm = Xk.col(i).transpose() * M * Xk.col(i);
			mNorm = (double) 1.0 / sqrt(mNorm);
			Xk.col(i) = mNorm * Xk.col(i);
		}
		break;
	case SIMType::SIM_1level_Bathe:
		cout << "________SUBSPACE ITERATION METHOD with Bathe's proposed initialization Vectors (normalized).\n";
		generateBatheInitVectors(M.rows(), q, M, Xk);

		/* M-normalized per column */
		if (isNormalized) {
			for (int i = 0; i < q; i++) {
				double mNorm = Xk.col(i).transpose() * M * Xk.col(i);
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
	
	int firstNZ = 0;
	if (F.cols() == 3)		firstNZ = 1;	// Function Laplacian
	else if (F.cols() == 4) firstNZ = 6;	// Hessian
	int						lastConvergedId = firstNZ-1;

	//Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorization;
	do {
		int dCounter = 0;
		/* Stage II: Shifting and Sturm's check */
		/* At the beginning, no shift */
		if (k == 0) {
			//Eigen::SparseMatrix<double> Id;
			//double eps = 1e-6;
			//Id.resize(S.rows(), S.cols());
			//Id.setIdentity();
			//Id = eps*Id;
			//factorization.compute(S + Id);
			mu = Alpha(3) ;
			factorization.compute(S - mu*M);

			
		}
		/* for later factorization */
		else {
			/* [1] Bathe's proposal */
			cout << "___Bathe's shift proposal \n";
			mu = (Alpha(lastConvergedId) + Alpha(lastConvergedId - 1)) / 2.0;
			if		(mu <= (Alpha(lastConvergedId) + 0.001*Alpha(firstNZ)))		mu = Alpha(lastConvergedId) + 0.001*Alpha(firstNZ);
			else if (mu >= (Alpha(lastConvergedId) - 0.001*Alpha(firstNZ)))		mu = Alpha(lastConvergedId) - 0.001*Alpha(firstNZ);
			

			/* [2] Itoh and Wilson => can cause singularity*/

			/* [3] Gong */

			//mu = Alpha((int)floor(0.2*q));		// should be 0.4 (but I wanna check for it's half, meaning 0.4 of the eigenvalues of interests)
			factorization.compute(S - mu*M);
			//factorization.vectorD();		// for sturm's check
		}
		factCounter++;

		cout << "Head: " << factorization.vectorD().head(10).transpose() << endl;
		cout << "Tail: " << factorization.vectorD().tail(10).transpose() << endl;

		for (int i = 0; i < S.rows(); ++i) {
			if (factorization.vectorD()(i) < 0) dCounter++;
		}
		cout << "Counter: " << dCounter << endl; 


		cout << "__> Factorization " << factCounter << " with shift of " << mu << endl; 

		/* Stage III: Iterations */
		for (int i = 0; i < Imax; ++i) {
			printf("Iteration: %d \n", k+1);

			for (int j = 0; j < q; ++j) {						// easily parallelizeable
				Eigen::VectorXd v;
				if (type == SIMType::SIM_1level_Bathe && j == 0) {
					v = Xk.col(j);				// use the input as "MxXi" directly
				} else {
					v = M * (Xk.col(j));
				}
				XbarK.col(j) = factorization.solve(v);
			}

			/* Projection of M and K*/
			printf("________setting up reduced matrix M and S. ");
			Sproj = XbarK.transpose()*S*XbarK;
			Mproj = XbarK.transpose()*M*XbarK;
			printf("________Mred: %dx%d | Sred: %dx%d \n", Mproj.rows(), Mproj.cols(), Sproj.rows(), Sproj.cols());

			/* Reduced eigenvalue problem */
			printf("________solve reduced eigenvalue problem. ");
			computeEigenGPUDense(Sproj, Mproj, Q, Alpha);
			//computeEigenGPUDenseJacobi(Sproj, Mproj, Q, Alpha);
			//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
			//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);

			printf("________lifting up/prolongation.");
			Xk = XbarK*Q;

			/* Convergence Check*
			** Type 2: */
			checkConvergenceCat2(Xk, Alpha, M, S, conv2);
			cout << "Conv 2 \n" << conv2.head(numEigs).transpose() << endl;
			/** Type 3: */
			checkConvergenceCat3(Q, Alpha, isWithShift, mu, conv3);
			cout << "Conv 3 \n" << conv3.head(numEigs).transpose() << endl;

			//cout << Alpha.transpose() << endl; 		

			EigValsSIMStack.col(k) = Alpha.block(0, 0, numEigs - 1, 1);
			convergenceStacks.col(k) = conv2.block(0, 0, numEigs - 1, 1);			

			k++;
			if (k >= maxIter || (conv2(numEigs - 1) < tol)) {
				isConverged = true;
				break;
			}
		}

		/* Check convergence */
		for (int i = lastConvergedId; i < conv2.rows(); ++i) {
			cout << "Checking convergence of ID " << i << ": " << conv2(i) << endl;
			if (conv2(i) > tol)
			{
				lastConvergedId = i;
				if (lastConvergedId < (numEigs - 1)) 
					isConverged = false;
			}
			else break;
		}
		cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl; 

	} while (!isConverged);

	EigVectsSIM = Xk;
	EigValsSIM = Alpha;
}

void MLEigenSolver::performMultiLevelSIM(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift)
{
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();
	
	printf("____[1] level: %d \n", k);
	int		q = max(2*numEigs, numEigs+8);

	/* 4: RECURSION */
	if (k == level)			// check if this already in the lowest/coarsest subspace
	{
		XkL = EigVectsReduced.topLeftCorner(EigVectsReduced.rows(), q);
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
		} else {
			SK = U[k-1].transpose()*SL*U[k-1];
			MK = U[k-1].transpose()*ML*U[k-1];
		}

		printf("____<2> Going deeper => \n ################");
		performMultiLevelSIM(SK, MK, k + 1, XkL, AkL, numEigs, maxIter, tol, isWithShift);
		printf(" level %d | SK: %dx%d | MK: %dx%d | tol: %.10f \n", k, SK.rows(), SK.cols(), MK.rows(), MK.cols(), tol);

		/* [3] SUBSPACE ITERATION METHOD */
		XkL = U[k] * XkL;
		printf("____<3> Lifting, Init vectors: %dx%d \n", XkL.rows(), XkL.cols());

		/* Adding a small number on the diagonal, for robustness */
		Eigen::SparseMatrix<double> SS;
		Eigen::SparseMatrix<double> Id; Id.resize(SK.rows(), SK.cols()); Id.setIdentity();
		double eps = 1e-6;
		Id = eps*Id;
		SS = SK + Id;		// Adding epsilon to the diagonal
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
		factorization.compute(SS);
		printf("____<4> Embedding a small-valued diagonal matrix \n");

		/* Temporary variables */
		bool convergenceCheck;
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
			for (int i = 0; i < q; i++) {						// easily parallelizeable
				Eigen::VectorXd v;
				v = MK * (XkL.col(i));
				XbarK.col(i) = factorization.solve(v);
			}

			/* Projected matrix */
			Sproj = XbarK.transpose()*SK*XbarK;
			Mproj = XbarK.transpose()*MK*XbarK;

			/* Computing the eigenproblem in reduced space*/
			computeEigenGPUDense(Sproj, Mproj, Q, AkL);
			//computeEigenMatlabDense(Sproj, Mproj, Sproj.rows(), Q, Alpha, "");
			//computeEigenSpectraDense(Sproj, Mproj, q, Q, Alpha);

			XkL = XbarK*Q;

			checkConvergenceCat2(XkL, AkL, MK, SS, conv2);
			cout << "_____ [" << iterCount + 1 << "] convergence check: \n" << conv2.head(numEigs).transpose() << endl;

			/* Store the eigenvalues and convergence rate if it's on the finest level */
			if (k == 0) {
				EigValsSIMStack.col(iterCount) = AkL.block(0, 0, numEigs-1, 1);
				convergenceStacks.col(iterCount) = conv2.block(0, 0, numEigs-1, 1);
			}

			iterCount++;
			convergenceCheck = (iterCount >= maxIter || (conv2(numEigs-1)<tol));

		} while (!convergenceCheck);
		printf("____<6> After iteration: XL= %dx%d \n", XkL.rows(), XkL.cols());


		/* Print some data when it's on the finest level */
		if (k == 0) {
			cout << "____<7> Storing the convergence data. \n";
			convergenceStacks.conservativeResize(Eigen::NoChange, iterCount);
			EigValsSIMStack.conservativeResize(Eigen::NoChange, iterCount);
		}
	}
	
	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;
}

void MLEigenSolver::checkConvergenceCat2(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, const Eigen::SparseMatrix<double>& MK, const Eigen::SparseMatrix<double>& SK, Eigen::VectorXd& convRate) const
{
	convRate.resize(A.rows());
	Eigen::VectorXd Kq, AMq, diff;

	for (int i = 0; i < A.rows(); i++) {
		Kq		= SK * Q.col(i);
		AMq		= A(i)*MK*Q.col(i);
		diff	= Kq - AMq;

		double n1 = diff.transpose()*MK*diff;
		double n2 = Kq.transpose()*MK*Kq;
		convRate(i) = sqrt(n1 / n2);
	}
	//printf("eigVals: %d | S: %dx%d | M: %dx%d \n", A.rows(), SK.rows(), SK.cols(), MK.rows(), MK.cols());
}

void MLEigenSolver::checkConvergenceCat3(const Eigen::MatrixXd& Q, const Eigen::VectorXd& A, bool isWithShift, double mu, Eigen::VectorXd& convRate) const
{
	convRate.resize(A.rows());

	for (int i = 0; i < A.rows(); ++i)
	{
		double a2 = A(i)*A(i);
		double qTq = Q.col(i).transpose()*Q.col(i);
		if (isWithShift) qTq += mu * (2 * A(i) - mu);
		double rat = a2 / qTq;
		double conv = abs(1.0 - rat);
		convRate(i) = sqrt(conv);
	}
}

/* Computing the reference eigenproblem */
void MLEigenSolver::computeRefEigenProblem(int numEigs, const EigenSolverType& solverType)
{
	switch (solverType)
	{
	case EigenSolverType::EIG_SOLVER_MATLAB:
		computeEigenMatlab(S, M, numEigs, EigVectsRef, EigValsRef, "");
		break;
	case EigenSolverType::EIG_SOLVER_SPECTRA:
		computeEigenSpectraSparse(S, M, numEigs, EigVectsRef, EigValsRef);
		break;
	case EigenSolverType::EIG_SOLVER_CUDA:
		if ( M.rows()>7500 ) {
			cout << "We think the problem is too big to be solved in GPU! \nConsider other solver for this problem.\n";
			return;
		}
		computeEigenGPU(S, M, EigVectsRef, EigValsRef);
		break;

	default:
		break;
	}
}

void MLEigenSolver::computeRefEigenProblemWithSelector(int numEigs, const EigenSolverType& solverType, int axis, double offset)
{
	cout << "____ Creating selector matrix \n";
	double xMin;
	double xMax;
	double xScale;
	double xBound;		// within this bound, the system will be discarded

	xMin	= V.col(axis).minCoeff();
	xMax	= V.col(axis).maxCoeff();
	xScale	= offset;
	xBound	= xMin + xScale*(xMax - xMin);

	/* Create selector matrix */
	vector<Eigen::Triplet<double>> STriplet;
	int vCounter = 0;

	/* Selector matrix for eigen-Functions */
	if (F.cols()==3 && S.rows()==V.rows()) {
		STriplet.reserve(round((1.0 - xScale)*V.rows()));
		for (int i = 0; i < V.rows(); ++i) {
			if (V(i, axis) >= xBound) {
				STriplet.push_back(Eigen::Triplet<double>(i, vCounter, 1.0));
				vCounter++;
			}
		}
		SelMatrix.resize(V.rows(), vCounter);
	}
	/* Selector Matrix for vibration modes of elestics */
	else if (F.cols()==4 && S.rows()==3*V.rows()) {
		STriplet.reserve(3 * round((1.0 - xScale)*V.rows()));
		for (int i = 0; i < V.rows(); ++i) {
			if (V(i, axis) >= xBound) {							// only select entries corresponding to selected vertices
				for (int j = 0; j < 3; j++) {
					STriplet.push_back(Eigen::Triplet<double>(3 * i + j, 3 * vCounter + j, 1));
				}
				vCounter++;
			}
		}
		SelMatrix.resize(3*V.rows(), 3*vCounter);
	}
	SelMatrix.setFromTriplets(STriplet.begin(), STriplet.end());
	STriplet.clear(); STriplet.shrink_to_fit();

	/* Reduced problem */
	Eigen::SparseMatrix<double>		SLoc;
	Eigen::SparseMatrix<double>		MLoc;
	Eigen::MatrixXd					EigVectsLoc;
	cout << "____Setting up the reduced problem \n";
	SLoc = SelMatrix.transpose()*S*SelMatrix;
	MLoc = SelMatrix.transpose()*M*SelMatrix;

	cout << "____Computing the eigenproblem with selector. \n";
	switch (solverType)
	{
	case EigenSolverType::EIG_SOLVER_MATLAB:
		computeEigenMatlab(SLoc, MLoc, numEigs, EigVectsLoc, EigValsRef, "");
		break;
	case EigenSolverType::EIG_SOLVER_SPECTRA:
		computeEigenSpectraSparse(SLoc, MLoc, numEigs, EigVectsLoc, EigValsRef);
		break;
	case EigenSolverType::EIG_SOLVER_CUDA:
		if (MLoc.rows()>7500) {
			cout << "We think the problem is too big to be solved in GPU! \nConsider other solver for this problem.\n";
			return;
		}
		computeEigenGPU(SLoc, MLoc, EigVectsLoc, EigValsRef);
		break;

	default:
		break;
	}

	/* Lift it up to the full space */
	cout << "____Lifting to the original size\n";
	EigVectsRef = SelMatrix*EigVectsLoc;
	printf("Size of eigenvectors: %dx%d \n", EigVectsRef.rows(), EigVectsRef.cols());

}

/* Visualization tools */
void MLEigenSolver::visualizeGeodesicDistanceMultLevel(igl::opengl::glfw::Viewer &viewer, int k) const
{
	Eigen::MatrixXd VColor;
	igl::jet(geoDist[k], true, VColor);
	viewer.data().set_colors(VColor);
}

void MLEigenSolver::visualizeBasis(igl::opengl::glfw::Viewer &viewer, int id, double scale) const {}
void MLEigenSolver::visualizeEigenvector(igl::opengl::glfw::Viewer &viewer, int id, const EigenVectorType& eigType, double scale) const {}

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
const Eigen::VectorXi& MLEigenSolver::getSample() const { return Sample; }
const vector<int>& MLEigenSolver::getSubspaceSize() const { return SubspaceSize; }

/* Getters of the eigen-values + vectors*/
const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsRef() const			{ return EigVectsRef; }
const Eigen::VectorXd& MLEigenSolver::getEigenVectorsRef(int id) const		
{ 
	if		(id >= EigVectsRef.cols())	return EigVectsRef.col(EigValsRef.rows()-1);
	else if	(id< 0)						return EigVectsRef.col(0);

	return EigVectsRef.col(id); 
}
const Eigen::VectorXd& MLEigenSolver::getEigenValuesRef() const				{ return EigValsRef; }
const double MLEigenSolver::getEigenValueRef(int id) const					
{ 
	if		(id >= EigValsRef.rows())	return EigValsRef(EigValsRef.rows()-1);
	else if	(id< 0)						return EigValsRef(0);

	return EigValsRef(id); 
}
const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsReduced() const		{ return EigVectsReduced; }
const Eigen::VectorXd& MLEigenSolver::getEigenVectorsReduced(int id) const
{
	if (id >= EigVectsReduced.cols())	return EigVectsReduced.col(EigValsReduced.rows()-1);
	else if (id< 0)						return EigVectsReduced.col(0);

	return EigVectsReduced.col(id); 
}
const Eigen::VectorXd& MLEigenSolver::getEigenValuesReduced() const			{ return EigValsReduced; }
const double MLEigenSolver::getEigenValueReduced(int id) const
{
	if (id >= EigValsReduced.rows())	return EigValsReduced(EigValsReduced.rows() - 1);
	else if (id< 0)						return EigValsReduced(0);

	return EigValsReduced(id); 
}
const Eigen::MatrixXd& MLEigenSolver::getEigenVectorsSIM() const			{ return EigVectsSIM; }
const Eigen::VectorXd& MLEigenSolver::getEigenVectorsSIM(int id) const
{
	if (id >= EigVectsSIM.cols())		return EigVectsSIM.col(EigValsSIM.rows()-1);
	else if (id< 0)						return EigVectsSIM.col(0);
	return EigVectsSIM.col(id); 
}
const Eigen::VectorXd& MLEigenSolver::getEigenValuesSIM() const				{ return EigValsSIM; }
const double MLEigenSolver::getEigenValueSIM(int id) const
{
	if (id >= EigValsSIM.rows())		return EigValsSIM(EigValsSIM.rows() - 1);
	else if (id< 0)						return EigValsSIM(0);

	return EigValsSIM(id); 
}