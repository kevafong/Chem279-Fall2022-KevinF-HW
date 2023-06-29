#include <map>
#include <vector>
#include "utility.h"

class Model
{   // C++ implementation of the Complete Neclect of Differential Overlap method of Quantum Chemistry.
public:
    Model(std::string filename);                // Class constructor
    //arma::mat H_STO3G, C_STO3G, N_STO3G, O_STO3G, F_STO3G;  
    arma::cube STO3G;                           // Slater type orbitals of 3 primitive Gaussians
    arma::mat CNDO2params;                                  // CNDO/2 values for Ionization, Electronegativities, and gaussian exponents
    arma::mat atom_mtx, basis_fns, overlap_mtx;
    arma::vec z_orbitals, z_access, z_values;   //  turns atomic numbers into orbitals, key index for certain matrices/arrays, and z values

    void load_STO3G  (std::string filename, arma::cube &STO3G);
        // loads STO-3G sets, adds normalization constants

    void build_vectors(arma::vec &z_access, arma::vec &z_orbitals, arma::vec &z_values);
        // builds useful vectors from the atomic numbers
        // z_access is used to access certain rows/columns of matrices by atom type for certain constants
        // z_orbitals returns the number of orbitals of an atom
        // z_values returns the z value, number of valence electrons of an element

    // Functions for Normalization Constants
    void addnorms (arma::mat &elem);
        // Normalization function 1, adds normalization constants to the STO-3G basis sets, adds a spacer for hydrogen to make sets same size

    void addnorms2 (arma::mat &elem, arma::imat quantums);
        // Normalization function 2, determines normalization constants, and adds them to a new column of the STO-3G basis set

    // Function to build Basis Functions
    void build_basis_fns (arma::mat &basis_fns);
        // Creates a matrix of parameters for each basis functions
        //          Each row is contains the respective information about each basis function:
        // Atomic number | Center_x | Center_y | Center_z | Quantum_l | Quantum_m | Quantum_n | exponent1 | exponent2 | exponent3 |
        // contractive coeffient 1 | contractive coeffient 2 | contractive coeffient 3 | normalization constant 1 | normalization constant 1 | normalization constant 1
    
    // Function to build Overlap Matrix
    std::tuple<double, double, double, arma::vec, arma::imat> Scomponents(arma::mat basis_fn, int j);
        // From a basis function, returns the necessary components of the jth primitive gaussian of a basis function:
        //      contractive coeffient | normalization constant | exponent | center | quantum numbers

    double indexOverlap (int i, int n);
        // Converts the index of a matrix into the its value of the overlap matrix, of the given basis functions
        //      Each value S is the sum of the overlaps of the 3 primitives of one basis function with the 3 primitives of another

    void overlapMatrix (arma::mat &overlap_mtx);
        // Constructs the overlap matrix for a given set of basis functions

    double S_AB_1D  (double xA, double xB, int l_A, int l_B, double alpha, double beta);


    double S_AB (arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta);

    
};