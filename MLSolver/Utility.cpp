#include "Utility.h"

#include <igl/doublearea.h>

#include <random>
#include <fstream>

#include <chrono>				// For quite precise timing, needed (perhaps) in most sub-task

using namespace std;

/* Create a list of neighbors of every vertex *
** Input:
**		V: matrix of vertices
**		F: matrix of faces (e.g. triangles) OR tetrahedron
** Output:
**		AM: Adjacency matrix */
void initiateAdM(Eigen::MatrixXd &V, Eigen::MatrixXi &F, vector<set<int>> &AM) {
	AM.clear();
	AM.resize(V.rows());

	/* For triangle mesh / faces */
	if (F.cols() == 3) {
		cout << "Neighborhood structure for triangle mesh. \n";
		for (int i = 0; i < F.rows(); i++)
		{
			int a1 = F(i, 0);
			int a2 = F(i, 1);
			int a3 = F(i, 2);

			AM[a1].insert(a2);
			AM[a1].insert(a3);
			AM[a2].insert(a1);
			AM[a2].insert(a3);
			AM[a3].insert(a1);
			AM[a3].insert(a2);
		}
	} else if (F.cols() == 4)
	{
		cout << "Neighborhood structure for tet-mesh\n";
		for (int i = 0; i < F.rows(); i++)
		{
			for (int j = 0; j < 4; j++) {
				/* vertex-to-vertex neighborhood */
				for (int k = 0; k < 4; k++) {
					if (j == k) continue;
					AM[F(i, j)].insert(F(i, k));
				}
			}
		}
	}
}

/* Precompute the distance between a vertex and its neighbor(s), for a bit of speed up */
void initiateDistTableSpM(Eigen::MatrixXd &V, const vector<set<int>> &AM, Eigen::SparseMatrix<double> &DistanceTable)
{
	vector<Eigen::Triplet<double>> locTrip;
	locTrip.reserve(V.rows());
	
	DistanceTable.resize(V.rows(), V.rows());
	DistanceTable.reserve(V.rows() * 6 * 2);

	for (int i = 0; i < V.rows(); i++) {
		double distAB;
		for (std::set<int, double>::iterator it = AM[i].begin(); it != AM[i].end(); ++it) {
			distAB = (V.row(i) - V.row(*it)).norm();
			locTrip.push_back(Eigen::Triplet<double>(i, *it, distAB));
		}
	}
	DistanceTable.setFromTriplets(locTrip.begin(), locTrip.end());
}

/* Compute distance between two points */
void VtoVDist(const Eigen::VectorXd &V1, const Eigen::VectorXd &V2, double &dist)
{
	Eigen::VectorXd	  v = V2 - V1;
	dist				= v.norm();
}

/* Rotate the edge*/
void rotateVector2D(const Eigen::RowVector2d &e, const double &angle, Eigen::RowVector2d &e2)
{
	double a = cos(angle);
	double b = sin(angle);
	e2(0) = e(0)*a - e(1)*b;
	e2(1) = e(0)*b + e(1)*a;
}

/* Get the angle between two edges */
double VtoVangle(const Eigen::Vector3d &V1, const Eigen::Vector3d &V2, const Eigen::Vector3d &P)
{
	Eigen::Vector3d v1 = V1 - P;
	Eigen::Vector3d v2 = V2 - P;
	double angle_ = acos((v1.dot(v2)) / (v1.norm()*v2.norm()));
	return angle_;
}

/* Compute Areas of surface*/
double computeSurfaceArea(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
	Eigen::VectorXd dArea;
	igl::doublearea(V, F, dArea);

	double totalArea = 0.0;
	for (int i = 0; i < F.rows(); i++)
	{
		totalArea += dArea(i) / 2.0;
	}

	return totalArea;
}
/* Check accuracy*/
double checkAccuracy(const Eigen::SparseMatrix<double>& M, const Eigen::VectorXd& ref, const Eigen::VectorXd& approx)
{
	Eigen::VectorXd diff = ref - approx;
	double energy1 = diff.transpose()*M*diff;
	double energy2 = ref.transpose()*M*ref;
	double energyApp = approx.transpose()*M*approx;
	//printf("diff en: %.6f | ref en: %.6f | red en: %.6f (%.6f) \n", energy1, energy2, energyApp, sqrt(energy1/energy2));
	cout << "__Diff: " << abs(energy1);
	cout << " | ref: " << energy2;
	cout << " | red: " << energyApp;
	cout << " | rel err.: " << sqrt(abs(energy1 / energy2)) << endl; 
	return sqrt(abs(energy1 / energy2));
}



void generateRandomMatrix(int m, int n, Eigen::MatrixXd& Mat)
{
	Eigen::MatrixXd MTemp(m, n);
	Eigen::MatrixXd Id(m, n); Id.setIdentity();
	Mat.resize(m, n);

	/* Random number generator */
	std::random_device rd;								// Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd());								// Standard mersenne_twister_engine seeded with rd()

	std::uniform_int_distribution<> dist(0, m);			// From 0 to F.rows()-1

	/* Filling out each element by random numbers */
	for (int j = 0; j < n; j++) {						// column
		for (int i = 0; i < m; i++) {					// row
			double randNum = (double)dist(gen);
			MTemp(i, j) = 2.0 * (randNum / (double)m - 0.5);
		}
		//M.col(j).normalize();		
	}

	/* Use as it is */
	Mat = MTemp;

	/* Regular normalization */
	//for (int j = 0; j < n; j++) {
	//	M.col(j) = MTemp.col(j).normalized();
	//}

	/* QR orthonormalized*/
	//cout << "Householder QR :";
	//Eigen::HouseholderQR<Eigen::MatrixXd> qr(MTemp);
	//M = qr.householderQ()*Id;	
	//cout << "ColPivHouseholder QR :";
	//Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(MTemp);
	//cout << "its rank:" << qr.rank() << endl;
	//M = qr.householderQ()*Id;

	///cout << "Block: \n" << Mat.topLeftCorner(20, min(n, 10)) << endl;
}

void generateBatheInitVectors(int m, int n, const Eigen::SparseMatrix<double> M, Eigen::MatrixXd& Mat)
{
	/* Initiating the Bathe's proposed inite vectors */
	Mat.resize(m, n);
	Eigen::VectorXd e(m); e.setZero();	
	Mat.col(0) = M.diagonal();

	for (int i = 1; i < (n-1); ++i) {
		e(i) = 1.0;
		Mat.col(i) = e;
		e(i) = 0.0;
	}

	/* Last column **
	** Filling out each element by random numbers */
	std::random_device rd;								// Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd());								// Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> dist(0, m);			// From 0 to F.rows()-1
	
	for (int i = 0; i < m; ++i) {					// row
		double randNum = (double)dist(gen);
		Mat(i, n-1) = 2.0 * (randNum / (double)m - 0.5);
	}
	double mNorm = Mat.col(n-1).transpose()*M*Mat.col(n-1);
	mNorm = (double) 1.0 / sqrt(mNorm);
	Mat.col(n-1) = mNorm * Mat.col(n-1);
}

void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio)
{
	cout << "> Scaling the mesh to have a unit length \n";
	Eigen::Vector3d minV;
	Eigen::Vector3d maxV;
	Eigen::Vector3d length;
	Eigen::MatrixXd MatSubs(V.rows(), V.cols());
	Eigen::MatrixXd MatAdd(V.rows(), V.cols());
	double maxVal;
	double scalingRatio = scaleRatio; //dia-meter, not radius
	

	/* Get the min and max coefficients */
	for (int i = 0; i < V.cols(); i++)
	{
		minV(i) = V.col(i).minCoeff();
		maxV(i) = V.col(i).maxCoeff();
		length(i) = maxV(i) - minV(i);
		MatSubs.col(i).setConstant(minV(i));
	}

	maxVal = length.maxCoeff();

	/* Translate to the Origin */
	V = V - MatSubs;

	/* Scale w.r.t the longest axis */
	V = V * (scalingRatio / maxVal);

	for (int i = 0; i < V.cols(); i++) {
		maxV(i) = V.col(i).maxCoeff();
		MatAdd.col(i).setConstant(0.5*maxV(i));
	}

	/* Translate s.t. the center is in the Origin */
	V = V - MatAdd;

	for (int i = 0; i < V.cols(); i++)
	{
		minV(i) = V.col(i).minCoeff();
		maxV(i) = V.col(i).maxCoeff();
		length(i) = maxV(i) - minV(i);
	}
	maxVal = length.maxCoeff();
	cout << "____after scaling: " << maxVal << endl;
}

void printVector(const Eigen::VectorXd& V, bool isTranspose, int numDigit)
{
	std::string text = "%." + to_string(numDigit) + "lg";
	if (isTranspose) text += "\t";
	else text += "\n";

	for (int i = 0; i < V.rows(); ++i) {
		std::printf(text.c_str(), V(i));
	}

	std::cout << std::endl; 
}