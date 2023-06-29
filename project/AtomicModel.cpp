#include "AtomicModel.h"
#include "utility.h"

void Model::load_STO3G  (std::string filename, arma::cube &STO3G)    {
    arma::mat X_STO3G;
    X_STO3G.load(filename,arma::raw_ascii);
    addnorms(X_STO3G);
    STO3G=join_slices(STO3G,X_STO3G);
}

void Model::build_vectors(arma::vec &z_access, arma::vec &z_orbitals, arma::vec &z_values)  {
    arma::mat z1 = atom_mtx.col(0);
    arma::mat z2 = z1;
    arma::mat z3 = z1;
    z1.transform ( [](int i) { 
        return ((i > 2) ? (i-5) : (i-1)); 
    } );
    z_access=z1;
    z2.transform ( [](int i) { 
        return ((i > 2) ? (4) : (1)); 
    } );
    z_orbitals=z2;
    z3.transform ( [](int i) { 
        return ((i > 2) ? (i-2) : (i)); 
    } );
    z_values=z3;
}

void Model::addnorms (arma::mat &elem) {
    arma::imat quantums = {{0,0,0}};
    addnorms2(elem, quantums);
    if (elem.n_cols==4)  {
        quantums = {{0,0,1}};
        addnorms2(elem, quantums);
    }
    else {
        std::vector<double> spacer= {0,0,0};
        elem.insert_cols(elem.n_cols-1,arma::mat(spacer));
        elem.insert_cols(elem.n_cols,arma::mat(spacer));
    }
}

void Model::addnorms2 (arma::mat &elem, arma::imat quantums)    {
    std::vector<double> normalsvector;
    arma::vec center = {0,0,0};
    for (int j=0 ; j<3; j++)    {
        double exponent = elem(j,0);
        double norm= S_AB (center, center, quantums, quantums, exponent, exponent);
        normalsvector.push_back(1/sqrt(norm));
    }
    arma::mat n_mat (normalsvector);
    elem.insert_cols(elem.n_cols,n_mat);
}

void Model::build_basis_fns (arma::mat &basis_fns)   {
    arma::mat input = atom_mtx;
    arma::mat output_mat;
    for (int i=0; i<input.n_rows; i++)  {
        int orbitals;
        arma::mat quantum_mat;
        arma::mat coeffs_mat;
        if (input(i,0)<3)  {
            arma::mat X_STO3G=STO3G.slice(z_access(i));
            orbitals=1;
            quantum_mat=arma::conv_to<arma::mat>::from(comboRepMatrix(3,0));
            coeffs_mat=join_rows(trans(X_STO3G.col(0)), trans(X_STO3G.col(1)),trans(X_STO3G.col(3)));
        }
        else{
            arma::mat X_STO3G=STO3G.slice(z_access(i));
            orbitals=4;
            quantum_mat=arma::conv_to<arma::mat>::from(arma::join_cols(comboRepMatrix(3,0),comboRepMatrix(3,1)));
            arma::mat c_contr=join_cols(trans(X_STO3G.col(1)),arma::repmat(trans(X_STO3G.col(2)), 3, 1));
            arma::mat norms=join_cols(trans(X_STO3G.col(3)),arma::repmat(trans(X_STO3G.col(4)), 3, 1));
            coeffs_mat=join_rows(arma::repmat(trans(X_STO3G.col(0)), orbitals, 1), c_contr, norms);
        }
        arma::mat center_mat=arma::repmat(input.row(i), orbitals, 1);   
        arma::mat atom_mat = arma::join_rows(center_mat,quantum_mat,coeffs_mat);
        output_mat=arma::join_cols(output_mat,atom_mat);
    }
    basis_fns = output_mat;
}    

std::tuple<double, double, double, arma::vec, arma::imat> Model::Scomponents(arma::mat basis_fn, int j) {
    arma::vec center = arma::vectorise(basis_fn.cols(1,3));
    arma::imat quantums = arma::conv_to<arma::imat>::from(basis_fn.cols(4,6));
    return std::make_tuple(basis_fn(j+10) , basis_fn(j+13) , basis_fn(j+7) , center,quantums);
}

double Model::indexOverlap (int i, int n)  {
    int mu, nu;
    std::tie(mu, nu) = divide(i, n);
    arma::mat fn_mu = basis_fns.row(mu);
    arma::mat fn_nu = basis_fns.row(nu);
    double sum=0;
    for (int k=0;k<3;k++)   {
        for (int l=0;l<3;l++)   {
            double d_kmu, d_lnu, N_kmu, N_lnu, exp_kmu, exp_lnu; 
            arma::vec center_mu, center_nu;
            arma::imat quantums_mu, quantums_nu;
            std::tie(d_kmu, N_kmu, exp_kmu, center_mu, quantums_mu) = Scomponents(fn_mu, k);
            std::tie(d_lnu, N_lnu, exp_lnu, center_nu, quantums_nu) = Scomponents(fn_nu, l);
            sum +=  d_kmu * d_lnu * N_kmu * N_lnu * S_AB(center_mu, center_nu, quantums_mu, quantums_nu, exp_kmu, exp_lnu);
        }
    }
    return sum;
}

void Model::overlapMatrix (arma::mat &overlap_mtx) {
    int n=basis_fns.n_rows;
    arma::mat overlap= indexMat(n,n);
    overlap.transform( [n, this](double i) { 
        return (indexOverlap (i,n)); 
    } );
    overlap_mtx = overlap;
}

double Model::S_AB_1D  (double xA, double xB, int l_A, int l_B, double alpha, double beta)  {
    double expPrefactor = exp(-alpha*beta*pow(xA-xB, 2) / (alpha+beta)) * sqrt(M_PI/(alpha+beta));

    double xP= (alpha * xA + beta * xB) / (alpha + beta);
    double doubleSumm=0;
    for (int i=0; i<l_A+1; i++) {
        for (int j=0; j<l_B+1; j++) {
            if ((i+j)%2==0)   {
                doubleSumm += combo(l_A, i) * combo(l_B, j) * factorial(i+j-1, 2) * pow(xP-xA, l_A-i) * pow(xP-xB, l_B-j) / (pow(2 * (alpha+beta),(i+j) / 2));
            }
        }
    }

    return expPrefactor * doubleSumm;
}

double Model::S_AB (arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta)  {
    double overlap = 1.0;
    for (int i=0; i<3; i++)
        overlap *= S_AB_1D(centerA(i), centerB(i), ls_a(i), ls_b(i), alpha, beta); 
    return overlap;
}

Model::Model(std::string filename)  {
    load_STO3G("H_STO3G.txt", STO3G);
    load_STO3G("C_STO3G.txt", STO3G);
    load_STO3G("N_STO3G.txt", STO3G);
    load_STO3G("O_STO3G.txt", STO3G);
    load_STO3G("F_STO3G.txt", STO3G);


    atom_mtx=loadfile(filename);
    build_vectors(z_access,z_orbitals, z_values);
    build_basis_fns(basis_fns);
    overlapMatrix(overlap_mtx);
    //arma::mat zero=arma::mat(basis_fns.n_rows,basis_fns.n_rows,arma::fill::zeros);
    //build_density(p_a,p_b,p_tot,p_tot_a,zero,zero); 
    //initials();  
}
