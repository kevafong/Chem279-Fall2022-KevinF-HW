#include "HW2.h"

double gaussian(double x, double center, int l, int exp_)  {
    // returns the Gaussian(x) with parameters of center, angular momentum (l), and exponent
    return pow(x-center, l)* exp(-exp_ * pow(x-center, 2));
}

double invoke (double x, double center, int l, int exp_, double (*func)(double, double, int, int)) {
    // Invokes a function with the argument x and the parameters of center, angular momentum (l), and exponent, with pointer to function.
    return func(x, center, l, exp_);
}

double trapazoidApprox (double a, double b, double offset, int N, int l_a, int l_b, int alpha, int beta, double (*func)(double, double, int, int))    {
    // Trapizoidal Appromization of an integral of function from a to b into N divisions. 
    // Right now, approximation is hard coded as the product of 2 functions, potential point of modular optimization.
    double center = (a+b)/2;
    arma::vec delimiter = arma::linspace(a, b, N+1);
    delimiter.transform( [func, center, offset, l_a, l_b, alpha, beta](double x) { 
        return (invoke(x, center, l_a, alpha, func) * invoke(x, center+offset, l_b, beta, func)); } );
    double sum = (2* arma::sum(delimiter) - (delimiter(0) + delimiter(delimiter.n_rows-1)))/2;
    sum = sum *(b-a)/N;
    return sum;
}

void approxConverge (double a, double b, double offset, int l_a, int l_b, int alpha, int beta, double (*func)(double, double, int, int))  {
    // Performs the trapzoidal approximation of the gaussian from 1 to 1000 divisions
    arma::vec ndivs = arma::logspace(0, 3, 7);
    ndivs.transform( [a, b, offset, l_a, l_b, alpha, beta, func](double N) { 
        return trapazoidApprox(a, b, offset, N, l_a, l_b, alpha, beta, *func); } );
    std::cout<< ndivs << std::endl;
}

int main() {
    std::cout<< "overlap of 2 s-type functions centered at origin \n";
    approxConverge(-5, 5, 0, 0, 0, 1, 1, gaussian);

    std::cout<< "overlap of s-type with p-type function centered at origin \n";
    approxConverge(-5, 5, 0, 0, 1, 1, 1, gaussian);

    std::cout<< "overlap of 2 s-type functions offset by 1 \n";
    approxConverge(-5, 6, 1, 0, 0, 1, 1, gaussian);

    std::cout<< "overlap of s-type with p-type function functions offset by 1 \n";
    approxConverge(-5, 6, 1, 0, 1, 1, 1, gaussian);
}

