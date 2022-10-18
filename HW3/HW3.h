#include <armadillo>
#include <iostream>
#include <vector>
#include "HW2.h"

arma::mat loadfile(std::string filename);
    // Loads the contents of filename into a matrix, checks if the number in the first line matches the number of rows

int n_basisfns (arma::mat matrix);
    // Return the number of basis functions associate with the molecule, will exit if there are unpaired electrons.

arma::mat basis_fns (arma::mat input);
    // Creates a matrix of parameters for each basis functions
    //          Each row is contains the respective information about each basis function:
    // Atomic number | Center_x | Center_y | Center_z | Quantum_l | Quantum_m | Quantum_n | exponent1 | exponent2 | exponent3 |
    // contractive coeffient 1 | contractive coeffient 2 | contractive coeffient 3 | normalization constant 1 | normalization constant 1 | normalization constant 1

arma::mat process_basisfns (arma::mat input_matrix);
    // Adds normalization constants to a set of basis functions

arma::mat primitiveNorms (arma::mat functions);
    // Returns a maxtrix of the normalization constants for the each of the 3 primitive Gaussians for each basis function

std::tuple<double, double, double, arma::vec, arma::imat> Scomponents(arma::mat basis_fn, int j);
    // From a basis function, returns the necessary components of the jth primitive gaussian of a basis function:
    //      contractive coeffient | normalization constant | exponent | center | quantum numbers

double indexOverlap (int i, int n, arma::mat basis_fns);
    // Converts the index of a matrix into the its value of the overlap matrix, of the given basis functions
    //      Each value S is the sum of the overlaps of the 3 primitives of one basis function with the 3 primitives of another

arma::mat overlapMatrix (arma::mat basisfns);\
    // Constructs the overlap matrix for a given set of basis functions

std::tuple<int, int> Hcomponents(arma::mat basis_fn);
    // From a basis function, returns the necessary components of the Huckel Hamiltonian of a basis function:
    //      atomic number | angular momentum

double indexHamiltonian (int i, int n, arma::mat basis_fns, arma::mat overlap);
    // Converts the index of a matrix into the its value of the Huckel Hamiltonian of the given basis functions

arma::mat huckelHamiltonian (arma::mat basisfns, arma::mat overlap);
    // Constructs the Huckel Hamiltonian for a given set of basis functions


arma::vec generalizedEigenvalues (arma::mat S, arma::mat H);
    // Returns the eigenvalues for an associated overlap and huckel hamiltonian of a set of basis functions

double totalEnergy (arma::vec Evalues);
    // Returns the energy of a molecule given its eigenvalues
