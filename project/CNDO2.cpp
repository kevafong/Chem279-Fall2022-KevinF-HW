#include "AtomicModel.h"
#include <iomanip>

class CNDO2: public Model  {
    public:
    CNDO2 (std::string filename): Model(filename) {
        CNDO2params = { {7.176, 14.051, 25.390, 32.272},
                    {arma::datum::nan, 5.572, 7.275, 9.111, 11.080},
                    {9,21,25,31,39}};
        p = ceil(basis_fns.n_rows/2.0);
        q = floor(basis_fns.n_rows/2.0);
        build_vectors(z_access,z_orbitals, z_values);
        arma::mat zero=arma::mat(basis_fns.n_rows,basis_fns.n_rows,arma::fill::zeros);
        build_density(p_a,p_b,p_tot,p_tot_a,zero,zero); 
    }

    arma::mat CNDO2params;                                  // CNDO/2 values for Ionization, Electronegativities, and gaussian exponents
    int p, q;                                   // p alpha electrons, q beta electrons
    arma::mat gamma_mtx, h_mtx;   // 
    arma::mat g_a, g_b, f_a, f_b;                           // g matrices and fock matrices for alpha and beta
    arma::mat c_a, c_b, p_a, p_b, p_tot, pa_old, pb_old;    // eigenvector matrices, density matrices
    arma::vec eps_a, eps_b, p_tot_a;            //  eigenvalue vectors, total density vector

    void build_gamma (arma::mat &gamma) {
        int n=atom_mtx.n_rows;
        gamma= indexMat(n,n);
        gamma.transform( [n,this](double i) { 
            return (indexGamma (i,n)); 
        } );
    }

    arma::uvec sigma_rows () {
        arma::vec orbital = arma::shift(z_orbitals,1);
        orbital(0)=0;
        return arma::conv_to<arma::uvec>::from(arma::cumsum(orbital));
    }

    double indexGamma(double i, int n)  {
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

    double sixdimint(double distance, int row, int col, int j,int k,int l,int m, arma::mat sigma_fns)  {
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

    void save_density (arma::mat &pa_old, arma::mat &pb_old)    {
        pa_old=p_a;
        pb_old=p_b;
    }

    void build_density (arma::mat &p_a, arma::mat &p_b, arma::mat &p_tot, arma::vec &p_tot_a, arma::mat c_a, arma::mat c_b) {
        p_a = c_a.cols(0,p-1)*trans(c_a.cols(0,q-1));
        p_b = c_b.cols(0,p-1)*trans(c_b.cols(0,q-1));
        p_tot=  p_a + p_b;
        atomic_density(p_tot_a);
    }

    void atomic_density(arma::vec &p_tot_a) {
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

    void build_hmat (arma::mat &h) {
        int n=basis_fns.n_rows;
        h= indexMat(n,n);
        h.transform( [n,this](double i) { 
            return (indexHmat (i,n)); 
        } );
    }

    double indexHmat(double i, int n)  {
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

    void build_gmat (arma::mat &g, bool beta) {
        int n=basis_fns.n_rows;
        g= indexMat(n,n);
        g.transform( [n,this,beta](double i) { 
            return (indexGmat (i,n,beta)); 
        } );
        ;
    }

    double indexGmat(double i, int n, bool beta)  {
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

    void build_fock (arma::mat &fock, arma::mat g) {
        fock= h_mtx+g;
    }

    int ofAtom (int func_i) {
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

    double indexVnuc(double i, int n)  {
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

    double E_CNDO2 ()   {
        int n=atom_mtx.n_rows;
        arma::mat Vnuc= indexMat(n,n);
        Vnuc.transform( [n,this](double i) { 
            return (indexVnuc (i,n)); 
        } );
        // std::cout<<"Nuclear Repulsion Energy is "<<arma::accu(Vnuc)/2<<" eV"<<std::endl;
        // std::cout<<"Electronic Energy is "<<arma::accu(p_a%(h_mtx+f_a))/2 + arma::accu(p_b%(h_mtx+f_b))/2<<" eV"<<std::endl;
        //return arma::accu(arma::mat(eps_a).rows(0,p-1))/2 + arma::accu(arma::mat(eps_b).rows(0,q-1))/2 + arma::accu(Vnuc)/2 + arma::accu(p_tot%h_mtx)/2;
        return arma::accu(p_a%(h_mtx+f_a))/2 + arma::accu(p_b%(h_mtx+f_b))/2 + arma::accu(Vnuc)/2;
    }

    void SCFiterate()  {
        save_density (pa_old,pb_old);
        build_gmat(g_a,0);
        build_gmat(g_b,1);
        build_fock(f_a,g_a);
        build_fock(f_b,g_b);
        arma::eig_sym(eps_a,c_a,f_a);
        arma::eig_sym(eps_b,c_b,f_b);
        build_density(p_a, p_b, p_tot, p_tot_a, c_a, c_b);
    }

    void build_matrices(arma::mat &basis_fns, arma::mat &overlap_mtx, arma::mat &gamma_mtx, arma::mat &h_mtx)   {
        build_basis_fns(basis_fns);
        overlapMatrix(overlap_mtx);
        build_gamma(gamma_mtx);
        build_hmat(h_mtx);
    }

    double calculateEnergy (double tolerance)   {
        int iterations = 0;
        bool equala = arma::approx_equal(p_a,pa_old,"absdiff", tolerance);
        bool equalb = arma::approx_equal(p_b,pb_old,"absdiff", tolerance);
        build_matrices(basis_fns, overlap_mtx, gamma_mtx, h_mtx);
        
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
        std::cout<<"the energy of the molecule is "<<energy<<" eV"<<std::endl;
        return energy;
    }    
};

template <class T>
double indexForce(double i, int n, T molecule)  {
    int n_row, n_col, nA, dimA, nB, dimB;
    std::tie(n_row, n_col) = divide(i, n);
    std::tie(nA, dimA) = divide(n_row, 3);
    std::tie(nB, dimB) = divide(n_col, 3);
    double h = 0.001;
    //if (dimA==dimB) {
    std::cout<<nA<<dimA<<std::endl;
    std::cout<<nB<<dimB<<std::endl;
    //}
    int print = 0;
    //if (dimA==dimB)
    print = 1;
    double baseenergy = molecule.calculateEnergy(10e-6);



    //---Central differences
    molecule.atom_mtx(nA, dimA+1) += h;
    molecule.atom_mtx(nB, dimB+1) += h;
    if (dimA==dimB)
    std::cout<<molecule.atom_mtx<<std::endl;
    double e1=molecule.calculateEnergy(10e-6);
    molecule.atom_mtx(nA, dimA+1) -= 2*h;
    double e2=molecule.calculateEnergy(10e-6);
    molecule.atom_mtx(nB, dimB+1) -= 2*h;
    double e4=molecule.calculateEnergy(10e-6);
    molecule.atom_mtx(nA, dimA+1) += 2*h;
    double e3=molecule.calculateEnergy(10e-6);


    //---Forward differences
    // double e1=molecule.calculateEnergy(0);
    // molecule.atom_mtx(nA, dimA+1) += h;
    // double e2=molecule.calculateEnergy(0);
    // molecule.atom_mtx(nB, dimB+1) += h;
    // double e4=molecule.calculateEnergy(0);
    // molecule.atom_mtx(nA, dimA+1) -= h;
    // double e3=molecule.calculateEnergy(0);
    //if (print)  {
    std::cout<<"base:"<<std::setprecision(17) <<baseenergy<<std::endl;
    std::cout<<"e1:"<<std::setprecision(17) <<e1<<std::endl;
    std::cout<<"e2:"<<std::setprecision(17) <<e2<<std::endl;
    std::cout<<"e3:"<<std::setprecision(17) <<e3<<std::endl;
    std::cout<<"e4:"<<std::setprecision(17) <<e4<<std::endl;


    std::cout<<"preenergy:"<<(e1-e2-e3+e4)<<std::endl;
    std::cout<<"energy:"<<(e1-e2-e3+e4)/(4*pow(h,2))<<std::endl;
    // std::cout<<"preenergy:"<<(e1+e3-e2-e4)<<std::endl;
    // std::cout<<"energy:"<<(e1+e3-e2-e4)/(4*pow(h,2))<<std::endl;
    //}


    return (e1-e2-e3+e4)/(4*pow(h,2));
}

// restrain to x dimension, 

template <class T>
arma::mat Force_mat(T molecule)  {
    int n = 3 * molecule.atom_mtx.n_rows;
    std::cout<<n<<std::endl;
    arma::mat force = indexMat(n,n);
    force.transform( [n, molecule](double i) { 
        return (indexForce (i,n, molecule)); 
    } );
    return force;
}

template <class T>
double indexG(double i, int n, T molecule)  {
    int n_row, n_col, nA, dimA, nB, dimB;
    std::tie(n_row, n_col) = divide(i, n);
    std::tie(nA, dimA) = divide(n_row, 3);
    std::tie(nB, dimB) = divide(n_col, 3);
    arma::vec masses = {1.008, 12.011, 14.007, 15.999, 18.998};
    return 1/(sqrt(masses(molecule.z_access(nA))*masses(molecule.z_access(nB))));
}

template <class T>
arma::mat G_mat(T molecule)  {
    int n = 3 * molecule.atom_mtx.n_rows;
    arma::mat gmat = indexMat(n,n);
    gmat.transform( [n, molecule](double i) { 
        return (indexG (i,n, molecule)); 
    } );
    return gmat;
}

std::tuple<arma::vec, arma::mat> HessianEigenvalues (arma::mat FG)  {
    arma::vec q; 
    arma::mat U;
    arma::eig_sym(q,U,FG);
    return std::make_tuple(q,U);
}

int main(int argc, char **argv) {
    CNDO2 molecule= CNDO2(argv[1]);
    double energy = molecule.calculateEnergy(10e-6);
    arma::mat force = Force_mat(molecule);
    // //force = arma::trunc(Force_mat(molecule)*10E6) / 10E6;
    std::cout<<"Force matrix:\n"<<force<<std::endl;
    // arma::mat gmat = G_mat(molecule);
    // std::cout<<"G matrix:\n"<<gmat<<std::endl;
    // std::cout<<"F%G:\n"<<force%gmat<<std::endl;
    // arma::vec q;
    // arma::mat U;
    // std::tie(q,U) = HessianEigenvalues(force%gmat);
    // std::cout<<"eigenvalues:\n"<<q<<std::endl;
    // std::cout<<"eigenvectors:\n"<<U<<std::endl;
    return 0;
} 