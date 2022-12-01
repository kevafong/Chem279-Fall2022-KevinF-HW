#include <tuple>
#include <armadillo>
#include <cmath>
#include <iostream>

//--------------Problem 1--------------

double gaussian(double x, double center, int l, int exp_);
    // returns the Gaussian(x) with parameters of center, angular momentum (l), and exponent

double invoke (double x, double center, int l, int exp_, double (*func)(double, double, int, int));
    // Invokes a function with the argument x and the parameters of center, angular momentum (l), and exponent, with pointer to function.

double trapazoidApprox (double a, double b, double offset, int N, int l_a, int l_b, int alpha, int beta, double (*func)(double, double, int, int));
    // Trapizoidal Appromization of an integral of function from a to b into N divisions. 
    // As of now, approximation is hard coded as the product of 2 functions, potential point of modular optimization.

void approxConverge (double a, double b, double offset, int l_a, int l_b, int alpha, int beta, double (*func)(double, double, int, int));
    // Performs the trapzoidal approximation of the gaussian from 1 to 1000 divisions, should demonstrate that the approximation converges

//--------------Problem 2--------------

std::tuple<int, int> divide(int dividend, int divisor); 
    // returns quotient and remainder, can convert matrix index to row and column indices

int factorial (int n, int i);
    //Returns the factorial of a number, n, reducing by increment of i
    // if i=1, returns the factorial, if i=2, returns the double factorial, , keeping numbers of same parity

int combo (int d, int l) ;
    // Returns the number of combinations of selecting l objects from a set of d objects.

int comboRep (int d, int l) ;
    // Returns the number of combinations of selecting l objects from a set of d objects, with repetition

arma::imat comboRepMatrix (int d, int l);
    // Returns a matrix of the possible combinations of selecting l objects from a set of d objects, with repetition

arma::mat indexMat (int m , int n) ;
    // Creates an m x n matrix, where the content of a cell is the row major index.

void overlap3D (arma::vec centerA, arma::vec centerB, int l_a, int l_b, double alpha, double beta);
    // Displays a matrix (m x n) of overlaps of different functions given their centers, gaussian exponents, and angular momentums
    //      where m in is number of rows, or the number of functions for given l_a, and n is the number of columns or the number of functions for given l_b
    //      Gaussian A - center: centerA, angular momentum: l_a, exponent: alpha
    //      Gaussian B - center: centerB, angular momentum: l_b, exponent: beta 

double cellToIntegral(int i, int n, arma::vec centerA, arma::vec centerB, arma::imat l_combosA, arma::imat l_combosB, double alpha, double beta);
    // Takes in an index i of a matrix, converts it to coordinate indices given the number of columns
    //  Uses the row and column index to index into the comboRepMatrix for each system to get a combination of angular momemtums
    //  Calculates the S_AB given the centers and coefficients of the points, and the angular momentums of the functions accessed
    //      l_combosA(matrix): matrix of possble combinations of angular mometum given an l_a
    //      l_combosB(matrix): matrix of possble combinations of angular mometum given an l_b

double S_AB (arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta);
    // Returns the overlaps of primitive gaussians, with centers at centerA and center B, and angular momentums in 3 dimensions ls_a and ls_b in 3D.
    //      ls_a(matrix): one of the functions for a given l_a, a row of the matrix l_combosA
    //      ls_b(matrix): one of the functions for a given l_b, a row of the matrix l_combosB

double S_AB_1D  (int dim, arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta);
    // Returns the analytical integral of the overlap of primitive gaussians, in a single given dimension.
    //      dim(int): index of the dimension over which function is being integrated, used to accesss the components of the center or angular momentum.

double centerP(double xA, double xB, double alpha, double beta) ;
    /* Returns the coordinate of the center of the product of two Gaussians in 1D
    *   xA(double) = x coordinate of CenterA, can be indexed to y or z.
    *   xB(double) = x coordinate of CenterB, can be indexed to y or z.   */

double expPrefactor (double xA, double xB, double alpha, double beta);
    // Returns the exponetial prefactor of the integration of the overlap of Primitive Gaussians (S_AB_1D) with the associated square root

double doubleSumm(double xA, double xB, int l_A, int l_B, double alpha, double beta) ;
    // Returns the double summation portion of the integration of the  of the overlap of Primitive Gaussians (S_AB_1D)





