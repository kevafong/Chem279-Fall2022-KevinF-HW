#include <map>
#include <vector>
#include "HW2.h"

arma::mat loadfile(std::string filename);
    // Loads the contents of filename into a matrix, checks if the number in the first line matches the number of rows
    // If the first column contains atomic symbols, they are converted to atomic numbers.

class CNDO2
{   // C++ implementation of the Complete Neclect of Differential Overlap method of Quantum Chemistry.
public:
    CNDO2(std::string filename);                // Class constructor
    arma::mat H_STO3G, C_STO3G, N_STO3G, O_STO3G, F_STO3G;  // Slater type orbitals of 3 primitive Gaussians
    arma::cube STO3G;
    arma::mat CNDO2params;                                  // CNDO/2 values for Ionization, Electronegativities, and gaussian exponents
    int p, q;                                   // p alpha electrons, q beta electrons
    arma::mat atom_mtx, basis_fns, overlap_mtx, gamma_mtx, h_mtx;   // 
    arma::mat g_a, g_b, f_a, f_b;                           // g matrices and fock matrices for alpha and beta
    arma::mat c_a, c_b, p_a, p_b, p_tot, pa_old, pb_old;    // eigenvector matrices, density matrices
    arma::vec eps_a, eps_b, p_tot_a;            //  eigenvalue vectors, total density vector
    arma::vec z_orbitals, z_access, z_values;   //  turns atomic numbers into orbitals, key index for certain matrices/arrays, and z values

    void load_STO3G  (arma::mat &elem, std::string filename, arma::cube &STO3G);
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
    void build_basis_fns (arma::mat &basis_fns, int &p, int &q);
        // Creates a matrix of parameters for each basis functions, detemines p and q on creation
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

    // Functions to build Gamma Matrix
    void build_gamma (arma::mat &gamma);
        // Builds the gamma matrix of 2 point electronic repulsion

    arma::uvec sigma_rows ();
        //  Returns the rows of basis function set that correspond to s-orbitals, as a uvec of indices

    double indexGamma(double i, int n);
        // Converts the index of a matrix into the its value of the gamma matrix.

    double sixdimint(double distance, int row, int col, int j,int k,int l,int m, arma::mat sigma_fns);
        // returns the six dimensional integral portion of the 2 point electonic repulsion

    // Functions for density matrices
    void save_density (arma::mat &pa_old, arma::mat &pb_old);
        // saves p_a and p_b to pa_old and pb_old, respectively
    
    void build_density (arma::mat &p_a, arma::mat &p_b, arma::mat &p_tot, arma::vec &p_tot_a, arma::mat c_a, arma::mat c_b);
        // Builds the density matrices for alpha and beta electrons, given eigenvectors c_a and c_b, determines p_tot

    void atomic_density(arma::vec &p_tot_a);
        // Builds a vector of atomic densities, given p_tot

    // Functions to build Fock Matrices
    void build_hmat (arma::mat &h);
        // Builds the Hamiltonian matrix

    double indexHmat(double i, int n);
        // Converts the index of a matrix into the its value of the Hamiltonian matrix

    void build_gmat (arma::mat &g, bool beta);
        // Builds the hamiltonian matrix for either alpha or beta electrons. 0: alpha, 1:beta

    double indexGmat(double i, int n, bool beta);
        // Converts the index of a matrix into the its value of the G matrix

    void build_fock (arma::mat &fock, arma::mat g);
        // Builds the Fock matrix

    int ofAtom (int func_i);
        // Returns the index of the atom to which a wave function belongs
    
    // Energy
    double indexVnuc(double i, int n);
        // Converts the index of a matrix into the nuclear repulsion energy between two atoms

    double E_CNDO2 ();
        // Returns the energy associated with the CNDO/2 model of a molecules 

    // Iteration and Output
    void initials  ();
        // Outputs the initial gamma, overlap, and h-core matrices, as well as p and q. These values do not mutate.

    void SCF (double tolerance);
        // Iterates throuhg the Self Consisting Field Model until a threshold of change is met.
        //      Prints out the Fock matrices, Eigenvectors, Eigenvalues, and Density Matrices at each step.
        //      Prints out energies of the model at the end of iterations.

    void SCFiterate();
        // Single iterate step of the Self Consisting Field: 
        //      save old densities, build fock matrices, solve eigenvalues, determine new densities

    //----------------------HW5--------
    arma::mat x_munu, y_AB, SR, gammaR, VnucR, gradientelec, gradient;

    void build_x(arma::mat &x);
    double indexbeta(double i, int n);

    void build_y(arma::mat &y);
    double index_y(double i, int n, arma::mat term4_full);

    void build_SR (arma::mat &SR);
    double indexSR (int i, int n, int d);
    double S_ABR(int d, arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta);
    double IR(int d, arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta);


    void build_gammaR (arma::mat &gammaR);
    double indexGammaR(double i, int n, int d);
    double sixdimintR(int d, arma::mat directionvector, int row, int col, int j,int k,int l,int m, arma::mat sigma_fns);

    void build_VnucR (arma::mat &VnucR);
    double indexVnucR(double i, int n, int d);

    void electronicgradient(arma::mat &gradientelec);
    void gradientsolve(arma::mat &gradient);
};