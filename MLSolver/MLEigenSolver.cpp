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
			// if the sum of the row is very small, then do not do anything on this row
			if (abs(rowSum(k)) < (10.0*std::numeric_limits<double>::epsilon())) {
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
	double eps = 1e-16;
	Eigen::SparseMatrix<double> Id(Basis.cols(), Basis.cols());
	Id.setIdentity();
	

	cout << "> Creating the reduced system matrices.\n";
	Eigen::SparseMatrix<double> S_, M_;			//reduced stiffness and mass matrix
	S_ = Basis.transpose()*S*Basis;
	M_ = Basis.transpose()*M*Basis;

	printf("__S red %.4f (%d nnz)\n", (double)S_.nonZeros() / (double)S_.rows(), S_.nonZeros());
	printf("__M red %.4f (%d nnz)\n", (double)M_.nonZeros() / (double)M_.rows(), M_.nonZeros());

	cout << "> Computing the reduced eigenproblem (in GPU). \n";

	int status;

	do {
		eps *= 10;
		status = computeEigenGPU(S_+eps*Id, M_, EigVectsReduced, EigValsReduced);
		cout << "Shift: " << eps << endl;
	} while (status != 0);

	EigVectsReduced = Basis * EigVectsReduced;
}

void MLEigenSolver::constructSelectorMatrixFullSize(Eigen::SparseMatrix<double>& SelMat, int axis, double offset) {}

/* Subspace Iteration methods */
void MLEigenSolver::performSubspaceIterationMethod(SIMType type, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType, int axis, double offset)
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
		if (isWithShift)
			performMultiLevelSIM_withShift_Draft(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift, shiftType);
		else 
			performMultiLevelSIM(SSel, MSel, 0, EigVectsSIM, EigValsSIM, numEigs, maxIter, tol, isWithShift);
	} else {
		if (isWithShift)
			performSIMwithShift_Draft(type, SSel, MSel, numEigs, maxIter, tol, isWithShift, shiftType);
		else
			performSIMwithWarmStart(type, SSel, MSel, numEigs, maxIter, tol, isWithShift);
	}

	printf("Size of eigenvalues: %d | eigVectors: %dx%d \n", EigValsSIM.rows(), EigVectsSIM.rows(), EigVectsSIM.cols());

	t1 = chrono::high_resolution_clock::now();
	duration = t1 - t0;
	cout << "...in " << duration.count() << " seconds." << endl;

	/* Write the convergence information to an external file. */
	string modelName = "Bar-Rectangle_" + to_string(V.rows()) + "_" + to_string(V.cols());
	string fileConv = "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + modelName + "_EigValsStack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift? "shift_"+ to_string((int)shiftType)+"_" : "" ) + to_string(tol) + ".txt";
	writeDenseMatrixToFile(EigValsSIMStack, fileConv);
	string fileEigs = "D:/4_SCHOOL/TU Delft/Research/Projects/MultigridSolver/Data/" + modelName + "_ConvergenceStack_" + to_string(numEigs) + "_method_" + to_string((int)type) + "_" + (isWithShift ? "shift_" + to_string((int)shiftType) + "_" : "") + to_string(tol) + ".txt";
	writeDenseMatrixToFile(convergenceStacks, fileEigs);
}

void MLEigenSolver::performSIMwithWarmStart(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift)
{
	/* Initialization */
	cout << "____[1] SIM::Setup initialization of variables\n";
	int					q = max(2*numEigs, numEigs+ 8);
	Eigen::MatrixXd		Xk;
	double eps = 1e-2 * std::numeric_limits<double>::epsilon();
	int status; 

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

	Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
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

		EigValsSIMStack.col(k) = Alpha.block(0, 0, numEigs-1, 1);
		convergenceStacks.col(k) = conv2.block(0, 0, numEigs-1, 1);

		k++;
		convergenceCheck = (k >= maxIter|| (conv2(numEigs-1)<tol));
	} while (!convergenceCheck);

	convergenceStacks.conservativeResize(Eigen::NoChange, k);
	EigValsSIMStack.conservativeResize(Eigen::NoChange, k);

	EigVectsSIM = Xk;
	EigValsSIM = Alpha;
}

void MLEigenSolver::performSIMwithShift_Draft(SIMType type, Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
{
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
		Xk = Basis * EigVectsReduced.block(0, 0, Basis.cols(), q);
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

		cout << "__> Factorization " << factCounter << " with shift of " << ((k==0)? eps : mu) << endl; 

		/* Stage III: Iterations */
		for (int i = 0; i < Imax; ++i) {
			printf("Iteration: %d \n", k+1);

			for (int j = 0; j < q; ++j) {						// easily parallelizeable
				Eigen::VectorXd v;
				if (type == SIMType::SIM_1level_Bathe && j == 0) {
					v = Xk.col(j);				// use the input as "MxXi" directly
				} else {
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

			EigValsSIMStack.col(k) = Alpha.block(0, 0, numEigs-1, 1);
			convergenceStacks.col(k) = conv2.block(0, 0, numEigs-1, 1);			

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
			if (conv2(i) < tol)	{
				lastConvergedId = i;
			}
			else break;
		}

		if ((k<maxIter) && (lastConvergedId < (numEigs - 1)))
			isConverged = false;

		cout << "____Last convergence on ID: " << lastConvergedId << " with value of " << conv2(lastConvergedId) << endl; 

	} while (!isConverged);

	convergenceStacks.conservativeResize(Eigen::NoChange, k);
	EigValsSIMStack.conservativeResize(Eigen::NoChange, k);

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
	double eps = 1e-1 * std::numeric_limits<double>::epsilon();
	int status; 

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
		//double eps = 1e-6;
		
		Id = eps*Id;
		SS = SK + Id;		// Adding epsilon to the diagonal
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> factorization;
		factorization.compute(SS);
		cout << "SS: \n" << SS.topLeftCorner(12, 6) << endl; 
		printf("____<4> Embedding a small-valued diagonal matrix \n");

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
			cout << "___Factorizing \n";
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
				cout << status << "Cannot resolve eigenproblem with shifting " << eps << endl;
				eps *= 10;
				SS = SK + eps*Id;
				factorization.compute(SS);
				cout << "Increase the shift to " << eps << endl; 
				continue;
			}

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

void MLEigenSolver::performMultiLevelSIM_withShift_Draft(Eigen::SparseMatrix<double>& SL, Eigen::SparseMatrix<double>& ML, int k, Eigen::MatrixXd& XkL, Eigen::VectorXd& AkL, int numEigs, int maxIter, double tol, bool isWithShift, const ShiftType& shiftType)
{
	chrono::high_resolution_clock::time_point	t0, t1;
	chrono::duration<double>					duration;
	t0 = chrono::high_resolution_clock::now();

	printf("____[1] level: %d \n", k);
	int		q = max(2 * numEigs, numEigs + 8);
	const int Imax = 4;
	int iterCount = 0;
	int factCounter = 0;
	int status=0;

	/* 4: RECURSION */
	if (k == level)			// check if this already in the lowest/coarsest subspace
	{
		XkL = EigVectsReduced.topLeftCorner(EigVectsReduced.rows(), q);
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
		if(isWithShift)
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
		bool convergenceCheck;
		Eigen::MatrixXd XbarK(XkL.rows(), q); XbarK.setZero();
		Eigen::MatrixXd Mproj, Sproj;
		Eigen::MatrixXd Q;
		//Eigen::VectorXd Alpha;
		Eigen::VectorXd conv2, conv3;
		double mu=0;
		double eps = 1e-5*std::numeric_limits<double>::epsilon();

		int						firstNZ = 0;
		int						lastConvergedId = 0;
		if (!isClampedBasis) {
			if (F.cols() == 3) {			/* FUNCTION */
				firstNZ = 1;		// first non-zero eigenvalues on the reduced/subspace Function Laplacian
				lastConvergedId = 1;		// first non-zero eigenvalue of the expected eigenvalues
			}
			else if (F.cols() == 4) {		/* HESSIAN */
				firstNZ = 6;		// first non-zero eigenvalues on the reduced/subspace (Hessian)
				lastConvergedId = 6;		// first non-zero eigenvalue of the expected eigenvalues	
			}
		}

		/* Iterating*/
		printf("____<5> Subspace iteration.\n");
		do {
			if (iterCount == 0) {
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
			cout << "__> Factorization " << factCounter << " with shift of " << ((iterCount==0) ? eps : mu) << endl;

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
					break;
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
					convergenceStacks.col(iterCount) = conv2.block(0, 0, numEigs - 1, 1);
				}

				iterCount++;

				/* Ignore the first iteration */
				if (iterCount == 1) break;

				if (iterCount >= maxIter || (conv2(numEigs - 1) < tol)) {
					convergenceCheck = true;
					break;
				}
			}

			if (conv2.size() < 1) continue;

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

void MLEigenSolver::LBSTest_pos(igl::opengl::glfw::Viewer &viewer, int numBasis, int id, double trans, int transAxis, double angle, int rotAxis)
{
	if (id >= V.rows()) id = 0;
	cout << "Set up angles and translation \n";
	double c1;
	double s1;
	Eigen::MatrixXd Q(3, 4); Q.setZero();
	Eigen::VectorXd qBar(12), q(12*numBasis);
	Eigen::SparseMatrix<double> UU(3*V.rows(), 12*numBasis);
	vector<Eigen::Triplet<double>> UTrip;
	UTrip.reserve(12 * numBasis* V.rows());

	cout << "Set up affine transformaiton matrix \n";
	for (int k = 0; k < numBasis; ++k)
	{
		srand(time(NULL));
		c1 = cos(k*angle / 180.0 * M_PI);
		s1 = sin(k*angle / 180.0 * M_PI);

		//switch (rotAxis)
		switch (k%3)
		{
		case 0:
			Q(0, 0) = 1.0;
			Q(1, 1) = c1;
			Q(1, 2) = -s1;
			Q(2, 1) = s1;
			Q(2, 2) = c1;
			break;
		case 1:
			Q(1, 1) = 1.0;
			Q(2, 2) = c1;
			Q(2, 0) = -s1;
			Q(0, 2) = s1;
			Q(0, 0) = c1;
			break;
		case 2:
			Q(2, 2) = 1.0;
			Q(0, 0) = c1;
			Q(0, 1) = -s1;
			Q(1, 0) = s1;
			Q(1, 1) = c1;
			break;
		default:
			break;
		}
		cout << "Transform affine matrix to vector. \n";
		// translation element
		Q(transAxis, k%3) = (1+k%3) * trans;
		qBar(0) = Q(0, 0);
		qBar(1) = Q(1, 0);
		qBar(2) = Q(2, 0);
		qBar(3) = Q(0, 1);
		qBar(4) = Q(1, 1);
		qBar(5) = Q(2, 1);
		qBar(6) = Q(0, 2);
		qBar(7) = Q(1, 2);
		qBar(8) = Q(2, 2);
		qBar(9) = Q(0, 3);
		qBar(10) = Q(1, 3);
		qBar(11) = Q(2, 3);

		q.block(12 * k, 0, 12, 1) = qBar;
	}


	/* Dijkstra distance */
	cout << "Set up distance \n";
	Eigen::VectorXd dist;
	Eigen::VectorXd handleId(numBasis);
	handleId(0) = id;
	Eigen::MatrixXd weight(V.rows(), numBasis);
	weight.setZero();
	dist.resize(V.rows());
	dist.setConstant(std::numeric_limits<double>::infinity());
	printf("Size: V: %d | AdV: %d | dist: %d \n", V.rows(), AdM.size(), dist.rows());

	double maxDist;

	Eigen::VectorXi::Index maxIndex;
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		if (k < (numBasis - 1)) {
			dist.maxCoeff(&maxIndex);
			handleId(k + 1) = maxIndex;
		}

		if (k == 0) {
			maxDist = dist.maxCoeff() / 2;
		}
	}

	cout << "Set up weight \n";
	double			aCoef = 2.0 / (pow(maxDist, 3));
	double			bCoef = -3.0 / (pow(maxDist, 2));
	cout << "Handles: " << handleId.transpose() << endl;

	dist.setConstant(std::numeric_limits<double>::infinity());
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		for (int i = 0; i < V.rows(); ++i) {
			if (dist(i) <= maxDist)
				weight(i,k) = aCoef*pow(dist(i), 3) + bCoef*pow(dist(i), 2) + 1.0;
		}
		dist.setConstant(std::numeric_limits<double>::infinity());
	}

	/* Partition of unity of the weight */
	cout << "partition of unity\n";
	cout << "_PreV: " << weight.row(0) << endl;
	for (int i = 0; i < V.rows(); ++i) {
		double rowSum = weight.row(i).sum();
		Eigen::VectorXd rowSumVector(numBasis);
		rowSumVector.setConstant(rowSum);
		weight.row(i) = weight.row(i).cwiseQuotient(rowSumVector);
	}
	cout << "_Post: " << weight.row(0) << endl;
	
	for (int k = 0; k < numBasis; ++k) {
		cout << "Set up the vectors of basis \n";
		/* Setting up the subspace UU*/
		for (int i = 0; i < V.rows(); ++i) {
			//if (dist(i) > maxDist) continue;
			/* For the first 3 vertex coordinates (i.e. x, y, and z) */
			for (int j = 0; j < 3; ++j) {
				double val = V(i, j);
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 3 * j + 0, weight(i, k)*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 3 * j + 1, weight(i, k)*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 3 * j + 2, weight(i, k)*val));
			}
			/* For the the translation/last column */
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 9, weight(i, k)));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 10, weight(i, k)));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 11, weight(i, k)));
		}
	}
	cout << "Set up basis matrix \n";
	UU.setFromTriplets(UTrip.begin(), UTrip.end());
	
	/* Setting position */
	Eigen::VectorXd pos;
	Eigen::MatrixXd Pos;
	cout << "Compute new position \n";
	pos = UU*q;

	cout << "Transform the new position \n";
	Pos.resize(V.rows(), V.cols());
	for (int i = 0; i < V.rows(); ++i) {
		for (int j = 0; j < V.cols(); ++j) {
			Pos(i, j) = pos(3*i+j);
		}
	}

	cout << "Init pos: " << V.row(0) << endl;
	cout << "After: " << Pos.row(0) << endl;

	Eigen::MatrixXd VCol;
	igl::jet(-weight.col(0), true, VCol);
	cout << "Show vertices \n";
	viewer.data().set_vertices(Pos);
	viewer.data().compute_normals();
	viewer.data().set_colors(VCol);

	viewer.data().points.resize(0,6);
	for(int i=0; i<numBasis; i++)
		viewer.data().add_points(Pos.row(handleId(i)), Eigen::RowVector3d(1.0, 0.1, 0.1));
}

void MLEigenSolver::LBSTest_vel(igl::opengl::glfw::Viewer &viewer, int numBasis, int id, double trans, int transAxis, double angle, int rotAxis)
{
	if (id >= V.rows()) id = 0;
	cout << "Set up angles and translation \n";
	double c1;
	double s1;
	Eigen::MatrixXd Q(3, 4); Q.setZero();
	Eigen::VectorXd qBar(12), q(12 * numBasis);
	Eigen::SparseMatrix<double> UU(3 * V.rows(), 12 * numBasis);
	vector<Eigen::Triplet<double>> UTrip;
	UTrip.reserve(12 * numBasis* V.rows());

	cout << "Set up affine transformaiton matrix \n";
	for (int k = 0; k < numBasis; ++k)
	{
		srand(time(NULL));
		c1 = cos(k*angle / 180.0 * M_PI);
		s1 = sin(k*angle / 180.0 * M_PI);

		//switch (rotAxis)
		switch (k % 3)
		{
		case 0:
			Q(0, 0) = 1.0;
			Q(1, 1) = c1;
			Q(1, 2) = -s1;
			Q(2, 1) = s1;
			Q(2, 2) = c1;
			break;
		case 1:
			Q(1, 1) = 1.0;
			Q(2, 2) = c1;
			Q(2, 0) = -s1;
			Q(0, 2) = s1;
			Q(0, 0) = c1;
			break;
		case 2:
			Q(2, 2) = 1.0;
			Q(0, 0) = c1;
			Q(0, 1) = -s1;
			Q(1, 0) = s1;
			Q(1, 1) = c1;
			break;
		default:
			break;
		}
		cout << "Transform affine matrix to vector. \n";
		Eigen::MatrixXd Id(3, 4);
		Id.setIdentity();
		Q = Q - Id;


		// translation element
		Q(transAxis, k % 3) = (1 + k % 3) * trans;
		qBar(0) = Q(0, 0);
		qBar(1) = Q(1, 0);
		qBar(2) = Q(2, 0);
		qBar(3) = Q(0, 1);
		qBar(4) = Q(1, 1);
		qBar(5) = Q(2, 1);
		qBar(6) = Q(0, 2);
		qBar(7) = Q(1, 2);
		qBar(8) = Q(2, 2);
		qBar(9) = Q(0, 3);
		qBar(10) = Q(1, 3);
		qBar(11) = Q(2, 3);

		q.block(12 * k, 0, 12, 1) = qBar;
	}


	/* Dijkstra distance */
	cout << "Set up distance \n";
	Eigen::VectorXd dist;
	Eigen::VectorXd handleId(numBasis);
	handleId(0) = id;
	Eigen::MatrixXd weight(V.rows(), numBasis);
	weight.setZero();
	dist.resize(V.rows());
	dist.setConstant(std::numeric_limits<double>::infinity());
	printf("Size: V: %d | AdV: %d | dist: %d \n", V.rows(), AdM.size(), dist.rows());

	double maxDist;

	Eigen::VectorXi::Index maxIndex;
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		if (k < (numBasis - 1)) {
			dist.maxCoeff(&maxIndex);
			handleId(k + 1) = maxIndex;
		}

		if (k == 0) {
			maxDist = dist.maxCoeff() / 2;
		}
	}

	cout << "Set up weight \n";
	double			aCoef = 2.0 / (pow(maxDist, 3));
	double			bCoef = -3.0 / (pow(maxDist, 2));
	cout << "Handles: " << handleId.transpose() << endl;

	dist.setConstant(std::numeric_limits<double>::infinity());
	for (int k = 0; k < numBasis; ++k) {
		ComputeDijkstra(V, handleId(k), AdM, dist);
		for (int i = 0; i < V.rows(); ++i) {
			if (dist(i) <= maxDist)
				weight(i, k) = aCoef*pow(dist(i), 3) + bCoef*pow(dist(i), 2) + 1.0;
		}
		dist.setConstant(std::numeric_limits<double>::infinity());
	}

	/* Partition of unity of the weight */
	cout << "partition of unity\n";
	cout << "_PreV: " << weight.row(0) << endl;
	for (int i = 0; i < V.rows(); ++i) {
		double rowSum = weight.row(i).sum();
		Eigen::VectorXd rowSumVector(numBasis);
		rowSumVector.setConstant(rowSum);
		weight.row(i) = weight.row(i).cwiseQuotient(rowSumVector);
	}
	cout << "_Post: " << weight.row(0) << endl;

	for (int k = 0; k < numBasis; ++k) {
		cout << "Set up the vectors of basis \n";
		/* Setting up the subspace UU*/
		for (int i = 0; i < V.rows(); ++i) {
			//if (dist(i) > maxDist) continue;
			/* For the first 3 vertex coordinates (i.e. x, y, and z) */
			for (int j = 0; j < 3; ++j) {
				double val = V(i, j);
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 3 * j + 0, (weight(i, k))*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 3 * j + 1, (weight(i, k))*val));
				UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 3 * j + 2, (weight(i, k))*val));
			}
			/* For the the translation/last column */
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 0, 12 * k + 9 , (weight(i, k))));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 1, 12 * k + 10, (weight(i, k))));
			UTrip.push_back(Eigen::Triplet<double>(3 * i + 2, 12 * k + 11, (weight(i, k))));
		}
	}
	cout << "Set up basis matrix \n";
	UU.setFromTriplets(UTrip.begin(), UTrip.end());

	/* Setting position */
	Eigen::VectorXd pos;
	Eigen::MatrixXd Pos;
	cout << "Compute new position \n";
	pos = UU*q;

	cout << "Transform the new position \n";
	Pos.resize(V.rows(), V.cols());
	for (int i = 0; i < V.rows(); ++i) {
		for (int j = 0; j < V.cols(); ++j) {
			Pos(i, j) = pos(3 * i + j);
		}
	}

	cout << "Init pos: " << V.row(0) << endl;
	cout << "After: " << V.row(0)+ Pos.row(0) << endl;

	Eigen::MatrixXd VCol;
	igl::jet(-weight.col(0), true, VCol);
	cout << "Show vertices \n";
	viewer.data().set_vertices(Pos+V);
	viewer.data().compute_normals();
	viewer.data().set_colors(VCol);

	viewer.data().points.resize(0, 6);
	for (int i = 0; i<numBasis; i++)
		viewer.data().add_points(Pos.row(handleId(i)), Eigen::RowVector3d(1.0, 0.1, 0.1));
}