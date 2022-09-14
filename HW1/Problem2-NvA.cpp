#include <iostream>
#include <armadillo>


double E_ij (double R_ij)  {
    // Determines the LeonardJones energy between two points of gold at distance R_ij
    double sigma_au=2.951;  // Angstroms
    double epsilon_au=5.29;    // kcal/mol
    double epsilon_ij = epsilon_au;
    double sigma_ij= sigma_au;

    return epsilon_ij*(pow(sigma_ij/R_ij,12)-2*pow(sigma_ij/R_ij,6));
}

double forwardDifferences(double R_ij, double h)    {
    // Returns the forward difference approximation of the force by Leonard Jones potential energy 
    return (E_ij(R_ij+h)-E_ij(R_ij))/h;
}

double centralDifferences(double R_ij, double h)    {
    // Returns the central difference approximation of the force by Leonard Jones potential energy 
    return (E_ij(R_ij+h)-E_ij(R_ij-h))/(2*h);
}

double analyticalForce(double R_ij)  {
    // Returns the analytical solution of the force by Leonard Jones potential energy
    double sigma_au=2.951;  // Angstroms
    double epsilon_au=5.29;    // kcal/mol
    double epsilon_ij = epsilon_au;
    double sigma_ij= sigma_au;

    return (12*epsilon_ij/sigma_ij)*(pow(sigma_ij/R_ij,12)-pow(sigma_ij/R_ij,6));
}

double slope(arma::mat x, arma::mat y)  {
    return (y(3)-y(0))/(x(3)-x(0));
}

int main() {
    std::cout << "At R = sigma, the derivative of the function is 0" << std::endl;
    std::cout << "Forward differences at h=0.01: \n \t Derivative = " << forwardDifferences(2.951, 0.01) << std::endl << std::endl;

    std::cout << "Central differences at h=0.01: \n \t Derivative = " << centralDifferences(2.951, 0.01) << std::endl << std::endl;

    std::cout << "Analytical solution: \n \t Derivative = " << analyticalForce(2.951) << std::endl << std::endl;

    std::cout << "Analytical solution at R=2.62904211723: \n \t Derivative = " << analyticalForce(2.62904211723) << std::endl << std::endl;
    

    double R= 2.951;
    double force = analyticalForce(R);
    arma::mat h = {{0.1, 0.01, 0.001, 0.0001}};
    arma::mat fd = h;
    fd.transform( [R](double val) { return (forwardDifferences(R, val)); } );
    arma::mat cd = h;
    cd.transform( [R](double val) { return (centralDifferences(R, val)); } );

    arma::mat values = abs(arma::join_cols(fd, cd) - force);
    arma::mat dataset = arma::join_cols(h, values);
    std::cout << "data: \n" << dataset << std::endl;
    dataset.transform(log10);
    std::cout << "log data: \n" << dataset << std::endl;

    std::cout << "slope of forward difference errors: " << slope(dataset.row(0), dataset.row(1)) << std::endl;
    std::cout << "slope of central difference errosr: " << slope(dataset.row(0), dataset.row(2)) << std::endl;
    std::cout << "As stepsize approaches 0, log(h) approaches -inf\nAs central differences has a greater slope for its log error, \n \tthe error descreases more dramatically as stepsize approaches 0" << std::endl;
    }


