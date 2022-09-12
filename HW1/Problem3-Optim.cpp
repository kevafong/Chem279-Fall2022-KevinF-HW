#include <iostream>
#include <armadillo>

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
    // Determines the distance of two points, i and j
    double sum = 0;
    for (int n=0; n<3; n++) {
        sum += pow(i(n)-j(n),2);
        }
    return sqrt(sum);
}

arma::mat direction (arma::mat i, arma::mat j) {
    // Determines the direction from point i to point j, normalized
    return (j - i)/distance(i,j);
}

double E_ij (arma::mat i, arma::mat j)  {
    // Determines the LeonardJones energy between two points of gold at distance R_ij
    double sigma_au=2.951;  // Angstroms
    double epsilon_au=5.29;    // kcal/mol
    double epsilon_ij = epsilon_au;
    double sigma_ij= sigma_au;

    double R_ij = distance(i, j);
    return epsilon_ij*(pow(sigma_ij/R_ij,12)-2*pow(sigma_ij/R_ij,6));
}

double LJ_system (arma::mat system) {
    // determines the LeonardJones energy between all atoms in a system of gold atoms.
    arma::mat system2 = system;
    system2.shed_col(0);
    int n_rows = system2.n_rows;
    double energy=0;
    for (int i=0; i<(n_rows-1);i++) {
        for (int j=i+1; j<(n_rows);j++)
        energy += E_ij(system2.row(i),system2.row(j));
    }
    return energy;
}

double LJ_system2 (arma::mat i, arma::mat n) {
    // determines the LeonardJones energy experienced by atom i in a system of n atoms
    int n_rows = n.n_rows;
    double energy=0;
    for (int j=0; j<n_rows;j++) {
        energy += E_ij(i,n.row(j));
    }
    return energy;
}

double analyticalForce(double R_ij)  {
    // Returns the analytical solution of the force by Leonard Jones potential energy
    double sigma_au=2.951;  // Angstroms
    double epsilon_au=5.29;    // kcal/mol
    double epsilon_ij = epsilon_au;
    double sigma_ij= sigma_au;

    return (12*epsilon_ij/sigma_ij)*(pow(sigma_ij/R_ij,12)-pow(sigma_ij/R_ij,6));
}

arma::mat gradient(arma::mat i, arma::mat n)    {
    // Determines the gradient of an atom i , as the sum of all forces acting on atom i from all atoms in set n
    arma::mat gradient = {{0, 0, 0}};
    int n_rows = n.n_rows;
    for (int j=0; j<n_rows; j++)    {
        gradient += analyticalForce(distance(i, n.row(j))) * direction(i,n.row(j));
    }
    return gradient;
}

double norm(arma::mat gradient)  {
    //Determines the norm of a gradient vector
    return distance(gradient, arma::mat {{0, 0, 0}});
}

arma::mat steepestDescent (arma::mat i, arma::mat n, double h, double tol)     {
    // Determines the new position of atom i, using a stepsize of "h". Will terminate when step is below tolerance "tol"
    arma::mat gradient_ = gradient(i,n);

    int count = 0;
    while (norm(gradient_)> tol && count < 1e06)     {
        arma::mat i_ = i - h*gradient_;
        if (LJ_system2(i_,n) < (LJ_system2(i,n)))   {
            i=i_;
            gradient_ = gradient(i,n);
            h *= 1.2;
        }
        else    {
            h /= 2;
        }
        count += 1;
    }
    return i;
}

bool toleranceStop(arma::mat coords, double tol)    {
    // Returns 0 when gradients for all atoms are under tolerance. Will allow while-loop as long as any gradients are above tolernace, returns 1
    bool stable = 0;
    for (int i=0; i<coords.n_rows; i++) {
        arma::mat coords_shed= coords;
        coords_shed.shed_row(i);
        double norm_=norm(gradient(coords.row(i),coords_shed));
        // std::cout << "gradient norm at "<<i<<" is "<<norm_<<  std::endl;
        if (norm_ > tol)    {
            stable = 1;
        }
    }
    return stable;
} 

arma::mat SDloop (arma::mat system, double h, double tol) {
    // Performs Steepest Descent on each atom before checking tolerance. Will loop all atoms again until all atoms meet tolerance
    arma::mat coords = system.cols(1,3);
    int n = system.n_rows;
    int count = 0;
    arma::mat prev_coords =arma::randu(size(coords));
    while (toleranceStop(coords, tol) && count < 1e06 && (coords == prev_coords).is_zero())    {
        prev_coords=coords;
        for (int i=0; i<n; i++) {
            arma::mat coords_shed= coords;
            coords_shed.shed_row(i);
            coords.row(i) = steepestDescent(coords.row(i),coords_shed, h, tol);
        }
        count+=1;
    }
    system.cols(1,3)=coords;
    return system;
}

int main() {
    arma::mat B = load("test2.txt");
	std::cout << "Test2: 2 particles at r = sigma/2^(1/6)\n" << B;
    std::cout << "Total Energy: " << LJ_system(B) << " kcal/mol \n" << std::endl;

    double h=0.001;
    double tol= 1e-7;

    std::cout << "Performing Steepest Descent..." <<  std::endl;
    arma::mat B_ = SDloop (B, h, tol);
    std::cout << "B Stabilized:\n" << B_ ;
    std::cout << "New Total Energy: " << LJ_system(B_) << " kcal/mol \n\n" << std::endl;

    arma::mat C = load("test3.txt");
	std::cout << "Test3: 3 particles\n" << C;
    std::cout << "Total Energy: " << LJ_system(C) << " kcal/mol \n" << std::endl;

    std::cout << "Performing Steepest Descent..." <<  std::endl;
    arma::mat C_ = SDloop (C, h, tol);
    std::cout << "C Stabilized:\n" << C_ ;
    std::cout << "New Total Energy: " << LJ_system(C_) << " kcal/mol \n\n" << std::endl;

    arma::mat D = load("test4.txt");
	std::cout << "Test4: 4 particles\n" << D;
    std::cout << "Total Energy: " << LJ_system(D) << " kcal/mol \n" << std::endl;

    std::cout << "Performing Steepest Descent..." <<  std::endl;
    arma::mat D_ = SDloop (D, h, tol);
    std::cout << "D Stabilized:\n" << D_ ;
    std::cout << "New Total Energy: " << LJ_system(D_) << " kcal/mol \n" << std::endl;

    }