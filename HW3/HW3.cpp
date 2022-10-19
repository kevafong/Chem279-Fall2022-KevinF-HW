#include "HW3.h"

arma::mat loadfile(std::string filename)    {
    std::ifstream infile(filename);
    std::ofstream buffer("buffer.txt");
    int n_atoms;
    if (infile.good())  {
        std::string header;
        getline(infile, header);
        std::string headerno = header.substr(0, header.find(' ')); 
        n_atoms = std::stoi(headerno);
        //std::cout<<"header:"<<header.front()<<std::endl;
        //std::cout<<"n:"<<n_atoms<<std::endl;
        while(!infile.eof())    {
            std::string row;
            getline(infile, row);
            buffer << row << std::endl;
        }
    }
    buffer.close();
    arma::mat matrix;
    matrix.load("buffer.txt");
    if (matrix.n_rows != n_atoms)   {
        printf("Error, wrong number of rows in header\n");
        exit(1);
    }
    int status = std::remove("buffer.txt");
    return matrix;
}

int n_basisfns (arma::mat matrix)   {
    arma::vec atomic_nos = matrix.col(0);
    atomic_nos.transform( [](int n) { 
        return n>2 ? n-2 : n; 
    } );
    int basisfns = sum(atomic_nos);
    if (basisfns % 2 != 0)  {
        printf("Error, odd number of electrons\n");
        exit(1);
    }
    return basisfns;
}

arma::mat primitiveNorms (arma::mat functions)   {
    std::vector<double> normalsvector;
    int rows=functions.n_rows;
    for (int i=0; i<rows; i++)    {
        arma::mat basisfunc = functions.row(i);
        arma::vec center = arma::vectorise(basisfunc.cols(1,3));
        arma::imat quantums = arma::conv_to<arma::imat>::from(basisfunc.cols(4,6));
        for (int j=0 ; j<3; j++)    {
            double exponent = basisfunc(j+7);
            double norm= S_AB (center, center, quantums, quantums, exponent, exponent);
            normalsvector.push_back(1/sqrt(norm));
        }
    }
    arma::mat n_mat (normalsvector);
    n_mat.reshape(3,rows);
    n_mat=n_mat.t();
    return n_mat;
}

arma::mat process_basisfns (arma::mat input_matrix)    {
    arma::mat norms_mat=primitiveNorms(input_matrix);
    arma::mat output=join_rows(input_matrix,norms_mat);
    return output;
}

arma::mat basis_fns (arma::mat input)   {
    arma::mat Hcoeffs = {   {3.42525091, 0.15432897},
                            {0.62391373, 0.53532814},
                            {0.16885540, 0.44463454}};
    arma::mat Ccoeffs = {   {2.94124940, -0.09996723, 0.15591627},
                            {0.68348310, 0.39951283, 0.60768372},
                            {0.22228990, 0.70011547, 0.39195739}};
    arma::mat output_mat;
    for (int i=0; i<input.n_rows; i++)  {
        int orbitals;
        arma::mat quantum_mat;
        arma::mat coeffs_mat;
        if (input(i,0)==1)  {
            orbitals=1;
            quantum_mat=arma::conv_to<arma::mat>::from(comboRepMatrix(3,0));
            coeffs_mat=join_rows(trans(Hcoeffs.col(0)), trans(Hcoeffs.col(1)));
        }
        else if (input(i,0)==6)  {
            orbitals=4;
            quantum_mat=arma::conv_to<arma::mat>::from(arma::join_cols(comboRepMatrix(3,0),comboRepMatrix(3,1)));
            arma::mat c_contr=join_cols(trans(Ccoeffs.col(1)),arma::repmat(trans(Ccoeffs.col(2)), 3, 1));
            coeffs_mat=join_rows(arma::repmat(trans(Ccoeffs.col(0)), orbitals, 1), c_contr);
        }
        arma::mat center_mat=arma::repmat(input.row(i), orbitals, 1);
        //center_mat.shed_col(0);     
        arma::mat atom_mat = arma::join_rows(center_mat,quantum_mat,coeffs_mat);
        output_mat=arma::join_cols(output_mat,atom_mat);
    }
    return process_basisfns(output_mat);
}

std::tuple<double, double, double, arma::vec, arma::imat> Scomponents(arma::mat basis_fn, int j) {
    arma::vec center = arma::vectorise(basis_fn.cols(1,3));
    arma::imat quantums = arma::conv_to<arma::imat>::from(basis_fn.cols(4,6));
    return std::make_tuple(basis_fn(j+10) , basis_fn(j+13) , basis_fn(j+7) , center,quantums);
}

double indexOverlap (int i, int n, arma::mat basis_fns)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    arma::mat fn_mu = basis_fns.row(n_row);
    arma::mat fn_nu = basis_fns.row(n_col);
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

arma::mat overlapMatrix (arma::mat basisfns) {
    int n=basisfns.n_rows;
    arma::mat overlap= indexMat(n,n);
    overlap.transform( [n, basisfns](double i) { 
        return (indexOverlap (i,n,basisfns)); 
    } );
    return overlap;
}

std::tuple<int, int> Hcomponents(arma::mat basis_fn) {
    arma::vec quantums = arma::vectorise(basis_fn.cols(4,6));
    return std::make_tuple(basis_fn(0) -1, sum(quantums));
}

double indexHamiltonian (int i, int n, arma::mat basis_fns, arma::mat overlap)  {
    int n_row, n_col, mu_A, mu_L;
    std::tie(n_row, n_col) = divide(i, n);
    arma::mat ionE = {  {-13.6,arma::datum::nan}, {-24.6,arma::datum::nan}, {-5.4,-3.5}, {-10.0,-6.0},
                        {-15.2,-8.5}, {-21.4,-11.4}, {-26.0,-13.4}, {-32.3,-14.8} };
    arma::mat fn_mu = basis_fns.row(n_row);
    std::tie(mu_A, mu_L) = Hcomponents(fn_mu);
    if (n_row==n_col)   {
        return ionE(mu_A, mu_L);
    }
    else{
        int nu_A, nu_L;
        arma::mat fn_nu = basis_fns.row(n_col);
        std::tie(nu_A, nu_L) = Hcomponents(fn_nu);
        double h_mu=ionE(mu_A, mu_L);
        double h_nu=ionE(nu_A, nu_L);
        return (1.75/2)*(h_mu+h_nu)*overlap(n_row, n_col);
    }
}

arma::mat huckelHamiltonian (arma::mat basisfns, arma::mat overlap) {
    int n=basisfns.n_rows;
    arma::mat hamiltonian= indexMat(n,n);
    hamiltonian.transform( [n, basisfns, overlap](double i) { 
        return (indexHamiltonian (i,n,basisfns, overlap)); 
    } );
    return hamiltonian;
}

arma::vec generalizedEigenvalues (arma::mat S, arma::mat H)  {
    arma::vec S_evals, epsilon;
    arma::mat S_evecs, C_prime;
    arma::eig_sym(S_evals,S_evecs,S);
    arma::mat X = S_evecs*arma::diagmat(arma::pow(S_evals, -0.5))*trans(S_evecs);
    arma::mat H_prime = arma::trans(X) * H * X;
    arma::eig_sym(epsilon,C_prime,H_prime);
    arma::mat C = X*C_prime;
    //std::cout<<C;
    //std::cout<<epsilon;
    return epsilon;
}

double totalEnergy (arma::vec Evalues)  {
    return 2*arma::sum(Evalues.head(Evalues.size()/2));
}

double openMoleculeFile (std::string filename, bool print)  {
    std::string molecname = filename;
    for (int i=0; i<4 ; i++)   molecname.pop_back();
    arma::mat matrix=loadfile(filename);

    int n = n_basisfns(matrix);
    arma::mat basisfuncs = basis_fns(matrix);
    if (print)  {
        std::cout<< "Basis functions for "<<molecname<<":"<<std::endl;
        std::cout<<basisfuncs<<std::endl;
    }
    arma::mat S= overlapMatrix(basisfuncs);
    if (print)  {
    std::cout<< "Overlap Matrix for "<<molecname<<":"<<std::endl;
    std::cout<<S<<std::endl;
    }

    arma::mat H= huckelHamiltonian(basisfuncs,S);
    if (print)  {
    std::cout<< "Huckel Hamiltonian for "<<molecname<<":"<<std::endl;
    std::cout<<H<<std::endl;
    }

    arma::vec Evalues = generalizedEigenvalues (S,H);
    double E=totalEnergy(Evalues);
    std::cout<<"Total energy of "<<molecname<<": "<<E<<" eV"<<std::endl<<std::endl;
    return E;
}

int main() {
    double h2 = openMoleculeFile("H2.txt",false);
    double c2h2 = openMoleculeFile("C2H2.txt",false);
    double c2h2k = openMoleculeFile("C2H2-Kevin.txt",false);
    double c2h4 = openMoleculeFile("C2H4.txt",true);
    double energyDif = c2h4 - c2h2 - h2;
    std::cout<<"\u0394H = "<<energyDif<< " eV."<<std::endl;
    double benzene = openMoleculeFile("benzene.txt",false);
    double c6h8ccc = openMoleculeFile("C6H8ccc.txt",false);
    double energyDif2 = benzene - c6h8ccc + h2;
    std::cout<<"\u0394H = "<<energyDif2<< " eV."<<std::endl;
    double c6h8ttt = openMoleculeFile("C6H8ttt.txt",false);
    double energyDif3 = benzene - c6h8ttt + h2;
    std::cout<<"\u0394H = "<<energyDif3<< " eV."<<std::endl;
    return 0;
}