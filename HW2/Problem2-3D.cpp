#include <iostream>
#include <armadillo>
#include <cmath>
#include <tuple>

int factorial (int n)   {
    //Returns the factorial of a number
    int fac=1;
    while (n>0) {
        fac *= n;
        n--;
    }
    return fac;
}

int doubleFactorial (int n)   {
    //Returns the double factorial of a number, keeping odds
    int fac=1;
    while (n>0) {
        fac *= n;
        n-= 2;
    }
    return fac;
}

int combo (int d, int l) {
    // Returns the number of combinations of selecting l objects from a set of d objects.
    return factorial(d) / (factorial(l) * factorial(d-l));
}

int comboRep (int d, int l) {
    // Returns the number of combinations of selecting l objects from a set of d objects, with repetition
    return factorial(d+l-1) / (factorial(l) * factorial(d-1));
}

arma::imat comboRepMatrix (int d, int l) {
    // Returns a matrix of the possible combinations of selecting l objects from a set of d objects, with repetition
    if (d == 1)  {
        return arma::imat(1, 1, arma::fill::value(l));
    }
    else if (l == 0)  {
        return arma::imat(1, d, arma::fill::zeros);
    }
    else    {
        arma::imat base = comboRepMatrix(d-1, 0);
        arma::imat matrix= arma::join_rows(arma::imat(1, 1, arma::fill::value(l)), base);
        int l_counter = 1;
        while(l_counter <= l)   {
            arma::imat submat = comboRepMatrix(d-1, l_counter);
            submat= arma::join_rows(arma::imat(comboRep(d-1, l_counter), 1, arma::fill::value(l-l_counter)), submat);
            matrix= arma::join_cols(matrix, submat);
            l_counter++;
        }
        return matrix;
    }
}

double centerP(double xA, double xB, double alpha, double beta) {
    // Returns the coordinate of the center of the product of two Gaussians in 1D
    return (alpha * xA + beta * xB) / (alpha + beta);
}

double expPrefactor (double xA, double xB, double alpha, double beta) {
    // Returns the exponetial prefactor of the overlap of Primitive Gaussians with the associated square root
    return exp(-alpha*beta*pow(xA-xB, 2) / (alpha+beta)) * sqrt(M_PI/(alpha+beta));
}

double doubleSumm(double xA, double xB, int l_A, int l_B, double alpha, double beta)  {
    // Returns the double summation of the overlap of Primitive Gaussians
    double xP= centerP(xA, xB, alpha, beta);
    double sum=0;
    for (int i=0; i<l_A+1; i++) {
        for (int j=0; j<l_B+1; j++) {
            if ((i+j)%2==0)   {
                sum += combo(l_A, i) * combo(l_B, j) * doubleFactorial(i+j-1) * pow(xP-xA, l_A-i) * pow(xP-xB, l_B-j) / (pow(2 * (alpha+beta),(i+j) / 2));
            }
        }
    }
    return sum;
}

double S_AB_1D  (int dim, arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta)  {
    // Returns the overlaps of primitive gaussians, in a single given dimension.
    double xA = centerA(dim);
    double xB = centerB(dim);
    int l_A = ls_a(dim);
    int l_B = ls_b(dim);
    return expPrefactor(xA, xB, alpha, beta) * doubleSumm(xA, xB, l_A, l_B, alpha, beta);
}

double S_AB (arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta)  {
    // Returns the overlaps of primitive gaussians, with centers at centerA and center B, and angular momentums in 3 dimensions ls_a and ls_b in 3D.
    double S_ABx=S_AB_1D(0, centerA, centerB, ls_a, ls_b, alpha, beta);
    double S_ABy=S_AB_1D(1, centerA, centerB, ls_a, ls_b, alpha, beta);
    double S_ABz=S_AB_1D(2, centerA, centerB, ls_a, ls_b, alpha, beta);
    return S_ABx * S_ABy * S_ABz;
    //return S_AB_1D(0, centerA, centerB, ls_a, ls_b, alpha, beta) * S_AB_1D(1, centerA, centerB, ls_a, ls_b, alpha, beta) * S_AB_1D(2, centerA, centerB, ls_a, ls_b, alpha, beta);
}

std::tuple<int, int> divide(int dividend, int divisor) {
    // returns quotient and remainder, can convert matrix index to row and column
    return  std::make_tuple(dividend / divisor, dividend % divisor);
}

double cellToIntegral(int i, int n, arma::vec centerA, arma::vec centerB, arma::imat a_combos, arma::imat b_combos, double alpha, double beta)    {
    // Takes in an index of a matrix, converts it to coordinates given the number of columns
    //      Uses the row and column index to index into the comboRepMatrix for each system to get a combination of angular momemtums
    //      Calcullates the S_AB given the centers and coefficients of the points, and the angular momentums of the functions accessed
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    arma::imat ls_a = a_combos.row(n_row);
    arma::imat ls_b = b_combos.row(n_col);
    return S_AB (centerA, centerB, ls_a, ls_b, alpha, beta);
}

void overlap3D (arma::vec centerA, arma::vec centerB, int l_a, int l_b, double alpha, double beta)  {
    // Displays a matrix (m x n) of overlaps of different functions given their centers, gaussian exponents, and angular momentums
    //      where m in is number of rows, or the number of functions for given l_a, and n is the number of columns or the number of functions for given l_b
    arma::imat a_combos = comboRepMatrix(3,l_a);
    arma::imat b_combos = comboRepMatrix(3,l_b);
    int m = comboRep(3,l_a);
    int n = comboRep(3,l_b);
    std::cout<< "Shell 1 has "<< m <<" functions.\nThis shell info: R -"<<centerA.t()<<"with angular momentum: "<<l_a<<", coefficient: "<<alpha<<std::endl<<std::endl;
    std::cout<< "Shell 2 has "<< n <<" functions.\nThis shell info: R -"<<centerB.t()<<"with angular momentum: "<<l_b<<", coefficient: "<<beta<<std::endl<<std::endl;
    arma::vec display_v = arma::regspace(0, m*n - 1);
    arma::mat display = arma::mat(display_v);
    display.reshape(m,n);
    display.transform( [n, centerA, centerB, a_combos, b_combos, alpha, beta](double i) { 
        return (cellToIntegral(i, n, centerA, centerB, a_combos, b_combos, alpha, beta)); 
    } );
    std::cout<<display;
}

int main()  {
    // std::cout<<comboRepMatrix(3,2)<<std::endl;

    arma::vec centerA_ = arma::vec{0,0,0};
    arma::vec centerB_ = arma::vec{1,1,2};
    overlap3D (centerA_, centerB_, 1, 1, 1.0, 1.0);
}