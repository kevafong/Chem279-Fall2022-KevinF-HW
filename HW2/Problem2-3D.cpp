#include "HW2.h"

std::tuple<int, int> divide(int dividend, int divisor) {
    return  std::make_tuple(dividend / divisor, dividend % divisor);
}

int factorial (int n, int i)   {
    int fac=1;
    while (n>0) {
        fac *= n;
        n-= i;
    }
    return fac;
}

int combo (int d, int l) {
    return factorial(d,1) / (factorial(l,1) * factorial(d-l,1));
}

int comboRep (int d, int l) {
    return factorial(d+l-1,1) / (factorial(l,1) * factorial(d-1,1));
}

arma::imat comboRepMatrix (int d, int l) {
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
    return (alpha * xA + beta * xB) / (alpha + beta);
}

arma::mat indexMat (int m , int n)  {
    arma::mat display = arma::mat(arma::regspace(0, m*n - 1));
    display.reshape(m,n);
    return display;
}

double expPrefactor (double xA, double xB, double alpha, double beta) {
    return exp(-alpha*beta*pow(xA-xB, 2) / (alpha+beta)) * sqrt(M_PI/(alpha+beta));
}

double doubleSumm(double xA, double xB, int l_A, int l_B, double alpha, double beta)  {
    double xP= centerP(xA, xB, alpha, beta);
    double sum=0;
    for (int i=0; i<l_A+1; i++) {
        for (int j=0; j<l_B+1; j++) {
            if ((i+j)%2==0)   {
                sum += combo(l_A, i) * combo(l_B, j) * factorial(i+j-1, 2) * pow(xP-xA, l_A-i) * pow(xP-xB, l_B-j) / (pow(2 * (alpha+beta),(i+j) / 2));
            }
        }
    }
    return sum;
}

double S_AB_1D  (int dim, arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta)  {
    double xA = centerA(dim);   // coordinate of center in a given dimension
    double xB = centerB(dim);
    int l_A = ls_a(dim);        // angular momentum of function in a given dimension
    int l_B = ls_b(dim);
    return expPrefactor(xA, xB, alpha, beta) * doubleSumm(xA, xB, l_A, l_B, alpha, beta);
}

double S_AB (arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta)  {
    double S_ABx=S_AB_1D(0, centerA, centerB, ls_a, ls_b, alpha, beta); 
    double S_ABy=S_AB_1D(1, centerA, centerB, ls_a, ls_b, alpha, beta);
    double S_ABz=S_AB_1D(2, centerA, centerB, ls_a, ls_b, alpha, beta);
    return S_ABx * S_ABy * S_ABz;
}

double cellToIntegral(int i, int n, arma::vec centerA, arma::vec centerB, arma::imat l_combosA, arma::imat l_combosB, double alpha, double beta)    {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    arma::imat ls_a = l_combosA.row(n_row);
    arma::imat ls_b = l_combosB.row(n_col);
    return S_AB (centerA, centerB, ls_a, ls_b, alpha, beta);
}

void overlap3D (arma::vec centerA, arma::vec centerB, int l_a, int l_b, double alpha, double beta)  {
    arma::imat l_combosA = comboRepMatrix(3,l_a);
    arma::imat l_combosB = comboRepMatrix(3,l_b);
    int m = comboRep(3,l_a);
    int n = comboRep(3,l_b);
    std::cout<< "Shell 1 has "<< m <<" functions.\nThis shell info: R -"<<centerA.t()<<"with angular momentum: "<<l_a<<", coefficient: "<<alpha<<std::endl<<std::endl;
    std::cout<< "Shell 2 has "<< n <<" functions.\nThis shell info: R -"<<centerB.t()<<"with angular momentum: "<<l_b<<", coefficient: "<<beta<<std::endl<<std::endl;
    arma::mat display = indexMat(m,n);
    display.transform( [n, centerA, centerB, l_combosA, l_combosB, alpha, beta](double i) { 
        return (cellToIntegral(i, n, centerA, centerB, l_combosA, l_combosB, alpha, beta)); 
    } );
    std::cout<<display<<std::endl;
}

int main()  {
    std::cout<<"Combinations of 2 items in 3 spaces\n"<<comboRepMatrix(3,2)<<std::endl;

    arma::vec centerA_ = arma::vec{0,0,0};
    arma::vec centerB_ = arma::vec{1,1,1};

    overlap3D (centerA_, centerB_, 1, 1, 1.0, 1.0);

    std::cout<<"Combinations of 4 items in 3 spaces\n"<<comboRepMatrix(3,2)<<std::endl;
    overlap3D (centerA_, centerB_, 2, 2, 1.0, 1.0);

    overlap3D (centerA_, centerB_, 3, 3, 1.0, 1.0);
}