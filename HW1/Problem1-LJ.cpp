#include <iostream>
#include <armadillo>
#include <string>

arma::mat load(std::string txtfile)   {
    arma::mat matrix;
    matrix.load(txtfile,arma::raw_ascii);
    int numAtoms = matrix(0,0);
    matrix.shed_row(0);
    int numRows = matrix.n_rows;
    if (numAtoms != numRows)    {
        throw "number of atoms does not match";
    }
    return matrix;
}

double distance(arma::mat i, arma::mat j)   {
    // Determines the distance of two points
    double sum = 0;
    for (int n=1; n<4; n++) {
        sum += pow(i(n)-j(n),2);
        }
    return sqrt(sum);
}

double E_ij (arma::mat i, arma::mat j)  {
    // Determines the LeonardJones energy between two points (of gold for now)
    double sigma_au=2.951;  // Angstroms
    double epsilon_au=5.29;    // kcal/mol
    double sigma_ag=2.955;  // Angstroms
    double epsilon_ag=4.56;    // kcal/mol
    
    // this code could be useless when implemented account for different elements
    double epsilon_i;
    double sigma_i;
    double epsilon_j;
    double sigma_j;
    if (i(0) == 79) {
        epsilon_i=epsilon_au;
        sigma_i=sigma_au;
    }
    else if (i(0) == 47) {
        epsilon_i=epsilon_ag;
        sigma_i=sigma_ag;
    }
    if (j(0) == 79) {
        epsilon_j=epsilon_au;
        sigma_j=sigma_au;
    }
    else if (j(0) == 47) {
        epsilon_j=epsilon_ag;
        sigma_j=sigma_ag;
    }
    double epsilon_ij = sqrt(epsilon_i*epsilon_j);
    double sigma_ij=sqrt(sigma_i*sigma_j);

    double R_ij = distance(i, j);
    return epsilon_ij*(pow(sigma_ij/R_ij,12)-2*pow(sigma_ij/R_ij,6));
}

void onlyGold (arma::mat system) {
    // verifies that all atoms in the system are gold atoms, Atomic No 79
    for (int i=0; i<(system.n_rows);i++) {
        if (system(i,0) != 79 && system(i,0) != 47) {
            std::cout << "element other than gold detected" << std::endl;
            throw;
        }
    }
}

double LJ_system (arma::mat system) {
    // determines the LeonardJones energy between all atoms in a system of gold atoms.
    onlyGold(system);
    int n_rows = system.n_rows;
    double energy=0;
    for (int i=0; i<(n_rows-1);i++) {
        for (int j=i+1; j<(n_rows);j++)
        energy += E_ij(system.row(i),system.row(j));
    }
    return energy;
}

int main() {
    arma::mat A = load("test1.txt");
	std::cout << "Test1: 2 particles at r = sigma\n" << A;
    double energyA = LJ_system(A);
    std::cout << "Total Energy: " << energyA << std::endl << std::endl;

    arma::mat B = load("test2.txt");
	std::cout << "Test2: 2 particles at r = sigma/2^(1/6)\n" << B ;
    double energyB = LJ_system(B);
    std::cout << "Total Energy: " << energyB << std::endl << std::endl;

    arma::mat C = load("input.txt");
	std::cout << "Test3: Random 5 atoms\n" << C ;
    double energyC = LJ_system(C);
    std::cout << "Total Energy: " << energyC << std::endl  << std::endl;

    arma::mat D = load("test5.txt");
	std::cout << "Test4: atoms other than gold\n" << D ;
    double energyD = LJ_system(D);
    std::cout << "Total Energy: " << energyD << std::endl << std::endl;
    }

