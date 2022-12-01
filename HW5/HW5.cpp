#include "HW5.h"

arma::mat loadfile(std::string filename)    {
    std::ifstream infile(filename);
    std::ofstream buffer("buffer.txt");
    int n_atoms;
    if (infile.good())  {
        std::string header;
        getline(infile, header);
        std::string headerno = header.substr(0, header.find(' ')); 
        n_atoms = std::stoi(headerno);
        std::map<char, std::string> AtomicSymbol= {{'H',"1",},{'C',"6",},{'N',"7",},{'O',"8",},{'F',"9",},};
        while(!infile.eof())    {
            std::string row;
            getline(infile, row);
            if(isalpha(row.front()))
                row.replace(0,1,AtomicSymbol.at(row.front()));
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


void CNDO2::load_STO3G  (arma::mat &elem, std::string filename)    {
    elem.load(filename,arma::raw_ascii);
    addnorms(elem);
}

void CNDO2::build_vectors(arma::vec &z_access, arma::vec &z_orbitals, arma::vec &z_values)  {
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

void CNDO2::addnorms (arma::mat &elem) {
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

void CNDO2::addnorms2 (arma::mat &elem, arma::imat quantums)    {
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

void CNDO2::build_basis_fns (arma::mat &basis_fns, int &p, int &q)   {
    arma::mat input = atom_mtx;
    arma::mat output_mat;
    for (int i=0; i<input.n_rows; i++)  {
        int orbitals;
        arma::mat quantum_mat;
        arma::mat coeffs_mat;
        if (input(i,0)==1)  {
            orbitals=1;
            quantum_mat=arma::conv_to<arma::mat>::from(comboRepMatrix(3,0));
            coeffs_mat=join_rows(trans(H_STO3G.col(0)), trans(H_STO3G.col(1)),trans(H_STO3G.col(3)));
        }
        else if (input(i,0)==6)  {
            orbitals=4;
            quantum_mat=arma::conv_to<arma::mat>::from(arma::join_cols(comboRepMatrix(3,0),comboRepMatrix(3,1)));
            arma::mat c_contr=join_cols(trans(C_STO3G.col(1)),arma::repmat(trans(C_STO3G.col(2)), 3, 1));
            arma::mat norms=join_cols(trans(C_STO3G.col(3)),arma::repmat(trans(C_STO3G.col(4)), 3, 1));
            coeffs_mat=join_rows(arma::repmat(trans(C_STO3G.col(0)), orbitals, 1), c_contr, norms);
        }
        arma::mat center_mat=arma::repmat(input.row(i), orbitals, 1);   
        arma::mat atom_mat = arma::join_rows(center_mat,quantum_mat,coeffs_mat);
        output_mat=arma::join_cols(output_mat,atom_mat);
    }
    basis_fns = output_mat;
    p = ceil(basis_fns.n_rows/2);
    q = floor(basis_fns.n_rows/2);
}    

std::tuple<double, double, double, arma::vec, arma::imat> CNDO2::Scomponents(arma::mat basis_fn, int j) {
    arma::vec center = arma::vectorise(basis_fn.cols(1,3));
    arma::imat quantums = arma::conv_to<arma::imat>::from(basis_fn.cols(4,6));
    return std::make_tuple(basis_fn(j+10) , basis_fn(j+13) , basis_fn(j+7) , center,quantums);
}

double CNDO2::indexOverlap (int i, int n)  {
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

void CNDO2::overlapMatrix (arma::mat &overlap_mtx) {
    int n=basis_fns.n_rows;
    arma::mat overlap= indexMat(n,n);
    overlap.transform( [n, this](double i) { 
        return (indexOverlap (i,n)); 
    } );
    overlap_mtx = overlap;
}

void CNDO2::build_gamma (arma::mat &gamma) {
    int n=atom_mtx.n_rows;
    gamma= indexMat(n,n);
    gamma.transform( [n,this](double i) { 
        return (indexGamma (i,n)); 
    } );
}

arma::uvec CNDO2::sigma_rows () {
    arma::vec orbital = arma::shift(z_orbitals,1);
    orbital(0)=0;
    return arma::conv_to<arma::uvec>::from(arma::cumsum(orbital));
}

double CNDO2::indexGamma(double i, int n)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    double sum=0;
    arma::mat coords = atom_mtx;
    coords.shed_col(0);
    arma::mat sigma_fns= basis_fns.rows(sigma_rows());
    double dist = sqrt(arma::accu(arma::square(coords.row(n_row)-coords.row(n_col))));
    for (int j=7; j<10; j++) {
        for (int k=7; k<10; k++) {
            for (int l=7; l<10; l++) {
                for (int m=7; m<10; m++) {
                    double d_prime1=sigma_fns(n_row,j+3)*sigma_fns(n_row,j+6);
                    double d_prime2=sigma_fns(n_row,k+3)*sigma_fns(n_row,k+6);
                    double d_prime3=sigma_fns(n_col,l+3)*sigma_fns(n_col,l+6);
                    double d_prime4=sigma_fns(n_col,m+3)*sigma_fns(n_col,m+6);
                    sum += d_prime1*d_prime2*d_prime3*d_prime4*sixdimint(dist,n_row, n_col, j,k,l,m,sigma_fns);
                }
            }
        }
    }
    return sum;
}

double CNDO2::sixdimint(double distance, int row, int col, int j,int k,int l,int m, arma::mat sigma_fns)  {
    double sig_a=1/(sigma_fns(row,j) + sigma_fns(row,k));
    double sig_b=1/(sigma_fns(col,l) + sigma_fns(col,m));
    double U_a=pow(M_PI*sig_a,1.5);
    double U_b=pow(M_PI*sig_b,1.5);
    double V2=1/(sig_a + sig_b);
    if (row==col)   {
        return 2*U_a*U_b*sqrt(V2/M_PI)*27.2114;
    }
    else {
        double T = V2 * pow(distance,2);
        return U_a*U_b*sqrt(1/pow(distance,2))*erf(sqrt(T))*27.2114;
    }
}

void CNDO2::save_density (arma::mat &pa_old, arma::mat &pb_old)    {
    pa_old=p_a;
    pb_old=p_b;
}

void CNDO2::build_density (arma::mat &p_a, arma::mat &p_b, arma::mat &p_tot, arma::vec &p_tot_a, arma::mat c_a, arma::mat c_b) {
    p_a = c_a.cols(0,p-1)*trans(c_a.cols(0,q-1));
    p_b = c_b.cols(0,p-1)*trans(c_b.cols(0,q-1));
    p_tot=  p_a + p_b;
    atomic_density(p_tot_a);
}

void CNDO2::atomic_density(arma::vec &p_tot_a) {
    arma::vec diagonal=p_tot.diag();
    std::vector<double> vector_holder;

    int counter = 0;
    for (int i=0; i<z_orbitals.n_elem; i++)   {
        double sum=0;
        for (int j=0;j<z_orbitals(i);j++)  {
            sum += diagonal(counter);
            counter++;
        }
        vector_holder.push_back(sum);
    }
    p_tot_a=arma::vec(vector_holder);
    
}

void CNDO2::build_hmat (arma::mat &h) {
    int n=basis_fns.n_rows;
    h= indexMat(n,n);
    h.transform( [n,this](double i) { 
        return (indexHmat (i,n)); 
    } );
}

double CNDO2::indexHmat(double i, int n)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    int A = ofAtom(n_row);
    if (n_row==n_col)   {
        double repulsion =0;
        int angular = accu(basis_fns.submat(n_row,4,n_col,6));
        for (int b;b<atom_mtx.n_rows; b++)  {
            if (b != A) {
                repulsion += z_values(b)*gamma_mtx(A,b);
            }
        }            
        return -CNDO2params(angular,z_access(A)) - (z_values(A)-0.5)*gamma_mtx(A,A)-repulsion;
    }
    else    {
        int B = ofAtom(n_col);
        int beta_A=CNDO2params(2,z_access(A));
        int beta_B=CNDO2params(2,z_access(B));
        return -(beta_A+beta_B)*overlap_mtx(n_row, n_col)/2;
    }
}

void CNDO2::build_gmat (arma::mat &g, bool beta) {
    int n=basis_fns.n_rows;
    g= indexMat(n,n);
    g.transform( [n,this,beta](double i) { 
        return (indexGmat (i,n,beta)); 
    } );
    ;
}

double CNDO2::indexGmat(double i, int n, bool beta)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    int A = ofAtom(n_row);
    arma::mat density = beta ? p_b : p_a;
    arma::vec ptot_a=p_tot.diag();
    if (n_row==n_col)   {
        double exchange =0;
        for (int b;b<atom_mtx.n_rows; b++)  {
            if (b != A)
                exchange += p_tot_a(b)*gamma_mtx(A,b);
        }
        return (p_tot_a(A)-density(n_row,n_row))*gamma_mtx(A,A)+exchange;
    }
    else    {
        int B = ofAtom(n_col);
        return -density(n_row, n_col)*gamma_mtx(A,B);
    }
}      

void CNDO2::build_fock (arma::mat &fock, arma::mat g) {
    fock= h_mtx+g;
}

int CNDO2::ofAtom (int func_i) {
    int atom_index = 0;
    while (func_i >= 0)   {
        func_i -= z_orbitals(atom_index);
        if (func_i < 0) {
            return atom_index;
        }
        atom_index++;
    }
    return 0;
}

double CNDO2::indexVnuc(double i, int n)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    if (n_row==n_col)   {
        return 0;
    }
    else{
        double Za=z_values(n_row);
        double Zb=z_values(n_col);
        arma::mat coords = atom_mtx;
        coords.shed_col(0);
        double dist = sqrt(arma::accu(arma::square(coords.row(n_row)-coords.row(n_col))));
        return 27.2113961318*Za*Zb/dist;
    }
}

double CNDO2::E_CNDO2 ()   {
    int n=atom_mtx.n_rows;
    arma::mat Vnuc= indexMat(n,n);
    Vnuc.transform( [n,this](double i) { 
        return (indexVnuc (i,n)); 
    } );
    std::cout<<"Nuclear Repulsion Energy is "<<arma::accu(Vnuc)/2<<" eV"<<std::endl;
    std::cout<<"Electronic Energy is "<<arma::accu(p_a%(h_mtx+f_a))/2 + arma::accu(p_b%(h_mtx+f_b))/2<<" eV"<<std::endl;
    //return arma::accu(arma::mat(eps_a).rows(0,p-1))/2 + arma::accu(arma::mat(eps_b).rows(0,q-1))/2 + arma::accu(Vnuc)/2 + arma::accu(p_tot%h_mtx)/2;
    return arma::accu(p_a%(h_mtx+f_a))/2 + arma::accu(p_b%(h_mtx+f_b))/2 + arma::accu(Vnuc)/2;
}

void CNDO2::SCFiterate()  {
    save_density (pa_old,pb_old);
    build_gmat(g_a,0);
    build_gmat(g_b,1);
    build_fock(f_a,g_a);
    build_fock(f_b,g_b);
    arma::eig_sym(eps_a,c_a,f_a);
    arma::eig_sym(eps_b,c_b,f_b);
    build_density(p_a, p_b, p_tot, p_tot_a, c_a, c_b);
}    

void CNDO2::initials  ()  {
    std::cout<<"gamma\n"<<gamma_mtx<<std::endl;
    std::cout<<"Overlap\n"<<overlap_mtx<<std::endl;
    std::cout<<"p = "<<p<<" q = "<<q<<std::endl;
    std::cout<<"H_core\n"<<h_mtx<<std::endl;
}

void CNDO2::SCF (double tolerance)   {
    int iterations = 0;
    bool equala = arma::approx_equal(p_a,pa_old,"absdiff", tolerance);
    bool equalb = arma::approx_equal(p_b,pb_old,"absdiff", tolerance);
    
    while (!equala || !equalb)  {
        SCFiterate();
        equala = arma::approx_equal(p_a,pa_old,"absdiff", tolerance);
        equalb = arma::approx_equal(p_b,pb_old,"absdiff", tolerance);
        
        // std::cout<<"Fa\n"<<f_a<<std::endl;
        // std::cout<<"Fb\n"<<f_b<<std::endl;
        // std::cout<<"Ea\n"<<eps_a<<std::endl;
        // std::cout<<"Eb\n"<<eps_b<<std::endl;
        // std::cout<<"Ca\n"<<c_a<<std::endl;
        // std::cout<<"Cb\n"<<c_b<<std::endl;
        // std::cout<<"Pa_new\n"<<p_a<<std::endl;
        // std::cout<<"Pb_new\n"<<p_b<<std::endl;
        // std::cout<<"P_t\n"<<p_tot_a<<std::endl;
        iterations++;
    }
    // std::cout<<"iterations :"<<iterations<<std::endl;
    double energy = E_CNDO2();
    // std::cout<<"the energy of the molecule is "<<energy<<" eV"<<std::endl;
}

CNDO2::CNDO2(std::string filename)  {
    load_STO3G( H_STO3G, "H_STO3G.txt");
    load_STO3G( C_STO3G, "C_STO3G.txt");
    load_STO3G( N_STO3G, "N_STO3G.txt");
    load_STO3G( O_STO3G, "O_STO3G.txt");
    load_STO3G( F_STO3G, "F_STO3G.txt");
    CNDO2params = { {7.176, 14.051, 25.390, 32.272},
                    {arma::datum::nan, 5.572, 7.275, 9.111, 11.080},
                    {9,21,25,31,39}};
    atom_mtx=loadfile(filename);
    build_vectors(z_access,z_orbitals, z_values);
    build_basis_fns(basis_fns, p, q);
    overlapMatrix(overlap_mtx);
    build_gamma(gamma_mtx);
    build_hmat(h_mtx);
    arma::mat zero=arma::mat(basis_fns.n_rows,basis_fns.n_rows,arma::fill::zeros);
    build_density(p_a,p_b,p_tot,p_tot_a,zero,zero); 
    initials();  
}

// -----------------------------HW5------------------------------------------------------------------------------------------------------------------------------------------------

void CNDO2::build_x (arma::mat &x) {
    int n=basis_fns.n_rows;
    arma::mat betas= indexMat(n,n);
    betas.transform( [n,this](double i) { 
        return (indexbeta (i,n)); 
    } );
    x = betas % p_tot;
}

double CNDO2::indexbeta(double i, int n)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    int A = ofAtom(n_row);
    int B = ofAtom(n_col);
    int beta_A=CNDO2params(2,z_access(A));
    int beta_B=CNDO2params(2,z_access(B));
    return -(beta_A+beta_B);
}

void CNDO2::build_y (arma::mat &y) {
    arma::mat term1 = p_tot_a*trans(p_tot_a);
    arma::mat z_a_mat= arma::repmat(arma::mat(z_values),1,z_values.n_elem);
    arma::mat z_b_mat= trans(z_a_mat);
    arma::mat ptot_a_mat= arma::repmat(arma::mat(p_tot_a),1,p_tot_a.n_elem);
    arma::mat ptot_b_mat= trans(ptot_a_mat);
    arma::mat term4_full = arma::pow(p_a,2)+arma::pow(p_b,2);
    int n=atom_mtx.n_rows;
    arma::mat term4= indexMat(n,n);
    term4.transform( [n,this, term4_full](double i) { 
        return (index_y (i,n, term4_full)); 
    } );
    y= term1 - z_b_mat%ptot_a_mat - z_a_mat%ptot_b_mat - term4;
}

double CNDO2::index_y(double i, int n, arma::mat term4_full)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    double sum=0;
    arma::uvec sorbitals = sigma_rows();
    return accu(term4_full.submat(sorbitals(n_row),sorbitals(n_col),sorbitals(n_row)+ z_orbitals(n_row)-1,sorbitals(n_col)+ z_orbitals(n_col)-1));
}

void CNDO2::build_SR (arma::mat &SR) {
    int n=basis_fns.n_rows;
    arma::mat Srow;
    for (int d=0; d<3; d++) {
        arma::mat Srow= indexMat(n,n);
        Srow.transform( [d, n, this](double i) { 
            return (indexSR (i,n, d)); 
        } );
        SR= arma::join_cols(SR, trans(arma::mat(arma::vectorise(Srow))));
    }
}

double CNDO2::indexSR (int i, int n, int d)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    int A = ofAtom(n_row);
    int B = ofAtom(n_col);
    std::cout<<n_row<<","<<n_col<<std::endl;
    if (A == B) {
        return 0;
    }
    else {
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
                sum +=  d_kmu * d_lnu * N_kmu * N_lnu * S_ABR(d, center_mu, center_nu, quantums_mu, quantums_nu, exp_kmu, exp_lnu);
                std::cout<<"sum: "<<sum<<std::endl;
            }
        }
        return sum;
    }
    
}

double CNDO2::S_ABR (int d, arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta) {
    arma::imat ones = arma::imat(size(ls_a), arma::fill::ones);
    // std::cout<<"newint"<<std::endl;
    // std::cout<<"ones\n"<<ones<<std::endl;
    // std::cout<<"ls_a\n"<<ls_a<<std::endl;
    // std::cout<<"new lsa\n"<<ls_a - ones<<std::endl;
    // std::cout<<"new lsa\n"<<ls_a + ones<<std::endl;
    double term1 = -ls_a(d)*S_AB_1D(d, centerA, centerB, ls_a - ones, ls_b, alpha, beta);
    double term2 =  2.0*alpha*S_AB_1D(d, centerA, centerB, ls_a + ones, ls_b, alpha, beta);
    std::cout<<term1<<","<<term2<<std::endl;
    std::cout<<term1+term2<<std::endl;
    return term1 + term2;
}

void CNDO2::build_gammaR (arma::mat &gammaR) {
    int n=atom_mtx.n_rows;
    arma::mat gammarow;
    for (int d=0; d<3; d++) {
        arma::mat gammarow= indexMat(n,n);
        gammarow.transform( [d, n,this](double i) { 
        return (indexGammaR (i,n,d)); 
        } );
        gammaR= arma::join_cols(gammaR, trans(arma::mat(arma::vectorise(gammarow))));
    }
}

double CNDO2::indexGammaR(double i, int n, int d)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    if (n_row==n_col)   {
        return 0;
    }
    else  {
        double sum=0;
        arma::mat coords = atom_mtx;
        coords.shed_col(0);
        arma::mat sigma_fns= basis_fns.rows(sigma_rows());
        arma::mat directionvector = coords.row(n_row)-coords.row(n_col);
        for (int j=7; j<10; j++) {
            for (int k=7; k<10; k++) {
                for (int l=7; l<10; l++) {
                    for (int m=7; m<10; m++) {
                        double d_prime1=sigma_fns(n_row,j+3)*sigma_fns(n_row,j+6);
                        double d_prime2=sigma_fns(n_row,k+3)*sigma_fns(n_row,k+6);
                        double d_prime3=sigma_fns(n_col,l+3)*sigma_fns(n_col,l+6);
                        double d_prime4=sigma_fns(n_col,m+3)*sigma_fns(n_col,m+6);
                        sum += d_prime1*d_prime2*d_prime3*d_prime4*sixdimintR(d, directionvector,n_row, n_col, j,k,l,m,sigma_fns);
                    }
                }
            }
        }
        return sum;
    }
}

double CNDO2::sixdimintR(int d, arma::mat directionvector, int row, int col, int j,int k,int l,int m, arma::mat sigma_fns)  {
    double distance = sqrt(arma::accu(arma::square(directionvector)));
    double sig_a=1.0/(sigma_fns(row,j) + sigma_fns(row,k));
    double sig_b=1.0/(sigma_fns(col,l) + sigma_fns(col,m));
    double U_a=pow(M_PI*sig_a,1.5);
    double U_b=pow(M_PI*sig_b,1.5);
    double V2=1/(sig_a + sig_b);
    double T = V2 * pow(distance,2);
    return U_a*U_b*(2*sqrt(V2)*exp(-T)/sqrt(M_PI)-erf(sqrt(T))/distance)*directionvector(d)*27.2114/pow(distance,2);
}

void CNDO2::build_VnucR (arma::mat &VnucR) {
    int n=atom_mtx.n_rows;
    arma::mat row;
    for (int d=0; d<3; d++) {
        arma::mat row= indexMat(n,n);
        row.transform( [d, n,this](double i) { 
        return (indexVnucR (i,n,d)); 
        } );
        VnucR= arma::join_cols(VnucR, arma::sum(row));
    }
}

double CNDO2::indexVnucR(double i, int n, int d)  {
    int n_row, n_col;
    std::tie(n_row, n_col) = divide(i, n);
    if (n_row==n_col)   {
        return 0;
    }
    else{
        double Za=z_values(n_row);
        double Zb=z_values(n_col);
        arma::mat coords = atom_mtx;
        coords.shed_col(0);
        arma::mat directionvector = coords.row(n_row)-coords.row(n_col);
        double distance = sqrt(arma::accu(arma::square(directionvector)));
        return -27.2113961318*Za*Zb*directionvector(d)/pow(distance,3.0);
    }
}

void CNDO2::electronicgradient(arma::mat &gradientelec)    {
    int n = atom_mtx.n_rows;
    int n_wavfns = basis_fns.n_rows;
    arma::mat gradientoverlap, gradientgamma;
    arma::mat x_vectors = trans(arma::repmat(arma::vectorise(x_munu),1,3));
    arma::mat xSR = x_vectors % SR;
    
    for (int d=0; d<3; d++) {
        arma::mat matrixtoreduce = xSR.row(d);
        matrixtoreduce.reshape(n_wavfns,n_wavfns);
        arma::mat row= indexMat(n,n);
        row.transform( [n,this, matrixtoreduce](double i) { 
            return (index_y (i,n, matrixtoreduce)); 
        } );
        row=sum(row);
        gradientoverlap= arma::join_cols(gradientoverlap, row);
    }

    arma::mat y_vectors = trans(arma::repmat(arma::vectorise(y_AB),1,3));
    arma::mat ygammaR = y_vectors % gammaR;

    for (int d=0; d<3; d++) {
        arma::mat matrix = ygammaR.row(d);
        matrix.reshape(n,n);
        matrix=sum(matrix);
        gradientgamma= arma::join_cols(gradientgamma, matrix);
    }
    std::cout<<"gradientoverlap\n"<<gradientoverlap<<std::endl;
    std::cout<<"gradientgamma\n"<<gradientgamma<<std::endl;
    gradientelec=gradientoverlap+gradientgamma;
}

int main() {
    CNDO2 C2H2= CNDO2("H2.txt");
    C2H2.SCF(10e-6);

    C2H2.build_x(C2H2.x_munu);
    C2H2.build_y(C2H2.y_AB);
    std::cout<<C2H2.x_munu<<std::endl;
    std::cout<<C2H2.y_AB<<std::endl;
    C2H2.build_SR(C2H2.SR);
    std::cout<<"Overlap_R\n"<<C2H2.SR<<std::endl;
    C2H2.build_gammaR(C2H2.gammaR);
    std::cout<<"gammaAB_R\n"<<C2H2.gammaR<<std::endl;

    C2H2.build_VnucR(C2H2.VnucR);
    std::cout<<"gradient(nuclear)\n"<<C2H2.VnucR<<std::endl;

    C2H2.electronicgradient(C2H2.gradientelec);

    return 0;
}