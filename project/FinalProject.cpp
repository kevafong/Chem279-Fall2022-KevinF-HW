#include <armadillo>
#include <cmath>
#include <iostream>

arma::mat Force_mat(arma::mat atom_mat, double (*energy)(arma::mat))  {
    int n = 3 * atom_mat.n_rows;
    arma::mat force = indexMat(n,n);
    force.transform( [n, atom_mat, energy](double i) { 
        return (indexForce (i,n, atom_mat, energy)); 
    } );
    return force;
}

double indexForce(double i, int n, arma::mat atom_mat, double (*energy)(arma::mat))  {
    int n_row, n_col, nA, dimA, nB, dimB;
    std::tie(n_row, n_col) = divide(i, n);
    std::tie(nA, dimA) = divide(n_row, 3);
    std::tie(nB, dimB) = divide(n_row, 3);
    double h = 1E-6;

    arma::mat c1 = atom_mat;
    c1(nA, dimA+1) += h;
    c1(nB, dimB+1) += h;
    arma::mat c2 = atom_mat;
    c2(nA, dimA+1) -= h;
    c2(nB, dimB+1) += h;
    arma::mat c3 = atom_mat;
    c3(nA, dimA+1) += h;
    c3(nB, dimB+1) -= h;
    arma::mat c4 = atom_mat;
    c4(nA, dimA+1) -= h;
    c4(nB, dimB+1) -= h;

    double e1=energy(c1);
    double e2=energy(c2);
    double e3=energy(c3);
    double e4=energy(c4);

    return (e1-e2-e3+e4)/(4*pow(h,2))
}

arma::vec accessor (arma::mat atom_mat)   {
    arma::mat z1 = atom_mtx.col(0);
    z1.transform ( [](int i) { 
        return ((i > 2) ? (i-5) : (i-1)); 
    } );
    return z1;
}

arma::mat G_mat(arma::mat atom_mat)  {
    int n = 3 * atom_mat.n_rows;
    arma::mat gmat = indexMat(n,n);
    arma::vec access = accessor(atom_mat)
    gmat.transform( [n, atom_mat, access](double i) { 
        return (indexG (i,n, atom_mat)); 
    } );
    return gmat;
}

double indexG(double i, int n, arma::mat atom_mat, arma::vec access)  {
    int n_row, n_col, nA, dimA, nB, dimB;
    std::tie(n_row, n_col) = divide(i, n);
    std::tie(nA, dimA) = divide(n_row, 3);
    std::tie(nB, dimB) = divide(n_row, 3);
    arma::vec masses = {1.008, 12.011, 14.007, 15.999, 18.998};
    return 1/(sqrt(masses(access(nA))*masses(access(nB))));
}

