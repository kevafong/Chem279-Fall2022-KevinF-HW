#include "AtomicModel.h"
#include <iomanip>

class Toy: public Model  {
    public:
    Toy (std::string filename): Model(filename) {
    }

    double energyoneatom (int i)    {
        arma::mat wavfn = atom_mtx.row(i);
        //std::cout<<wavfn;

        // toy function: x^2 + y^2/2 + 2*z^2
        return pow(wavfn(1),2) + pow(wavfn(2),2)/2 + 2*pow(wavfn(3),2);
    }

    double calculateEnergy ()  {
        double sum = 0;
        for (int i=0;i<atom_mtx.n_rows; i++)
            sum += energyoneatom(i);
        return sum;
    }
};

template <class T>
double indexForce(double i, int n, T molecule)  {
    int n_row, n_col, nA, dimA, nB, dimB;
    std::tie(n_row, n_col) = divide(i, n);
    std::tie(nA, dimA) = divide(n_row, 3);
    std::tie(nB, dimB) = divide(n_col, 3);
    double h = 0.1;
    int print = 0;
    if (dimA==dimB)
    print = 1;

    if (print)  {
    std::cout<<"Atom "<<nA<<", dimension "<<dimA<<std::endl;
    std::cout<<"Atom "<<nB<<", dimension "<<dimB<<std::endl;
    std::cout<<molecule.atom_mtx<<std::endl;
    }
    
    double baseenergy = molecule.calculateEnergy();



    //---Central differences
    molecule.atom_mtx(nA, dimA+1) += h;
    molecule.atom_mtx(nB, dimB+1) += h;
    
    double e1=molecule.calculateEnergy();
    molecule.atom_mtx(nA, dimA+1) -= 2*h;
    double e2=molecule.calculateEnergy();
    molecule.atom_mtx(nB, dimB+1) -= 2*h;
    double e4=molecule.calculateEnergy();
    molecule.atom_mtx(nA, dimA+1) += 2*h;
    double e3=molecule.calculateEnergy();


    //---Forward differences
    // double e1=molecule.calculateEnergy(0);
    // molecule.atom_mtx(nA, dimA+1) += h;
    // double e2=molecule.calculateEnergy(0);
    // molecule.atom_mtx(nB, dimB+1) += h;
    // double e4=molecule.calculateEnergy(0);
    // molecule.atom_mtx(nA, dimA+1) -= h;
    // double e3=molecule.calculateEnergy(0);
    if (print)  {
    std::cout<<"base:"<<std::setprecision(17) <<baseenergy<<std::endl;
    std::cout<<"e1:"<<std::setprecision(17) <<e1<<std::endl;
    std::cout<<"e2:"<<std::setprecision(17) <<e2<<std::endl;
    std::cout<<"e3:"<<std::setprecision(17) <<e3<<std::endl;
    std::cout<<"e4:"<<std::setprecision(17) <<e4<<std::endl;


    std::cout<<"pre-h energy:"<<(e1-e2-e3+e4)<<std::endl;
    std::cout<<"energy:"<<(e1-e2-e3+e4)/(4*pow(h,2))<<std::endl;
    // std::cout<<"preenergy:"<<(e1+e3-e2-e4)<<std::endl;
    // std::cout<<"energy:"<<(e1+e3-e2-e4)/(4*pow(h,2))<<std::endl;
    }
    double energy = (e1-e2-e3+e4)/(4*pow(h,2));
    // if (e1==e4 && e2==e3)
    //     energy = 0;

    return energy;

}
//x^2+y^2/2

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

arma::vec convert_epsilon (arma::vec eigenvalues)    {
    eigenvalues=arma::trunc(eigenvalues*10E6) / 10E6;   // eliminate near-zeros
    //currently in units of eV/(A^2 * Da)      Da = 0.931464 GeV/c^2
    eigenvalues *= pow(2.9979*10E8,2)/(0.9314941*10E9*pow(10E-10,2));
    return eigenvalues;
}

arma::vec epsilon_freq (arma::vec eigenvalues)    {
    arma::vec freq = arma::sqrt(eigenvalues)/2*M_PI;
    return freq;
}


 //toy function
 


//sclar to vector 
// units convert: epsilons will come out in A, masses in amu, energy eV, 
// dimensionless units before sqrt
// hertz or rads

int main(int argc, char **argv) {
    Toy molecule= Toy(argv[1]);
    double energy = molecule.calculateEnergy();
    arma::mat force = Force_mat(molecule);
    //force = arma::trunc(Force_mat(molecule)*10E6) / 10E6;
    std::cout<<"Force matrix:\n"<<force<<std::endl;
    arma::mat gmat = G_mat(molecule);
    std::cout<<"G matrix:\n"<<gmat<<std::endl;
    std::cout<<"F%G:\n"<<force%gmat<<std::endl;
    arma::vec q;
    arma::mat U;
    std::tie(q,U) = HessianEigenvalues(force%gmat);
    std::cout<<"eigenvalues:\n"<<q<<std::endl;
    std::cout<<"eigenvectors:\n"<<U<<std::endl;
    arma::vec freq = epsilon_freq(convert_epsilon(q));
    std::cout<<"frequencies:\n"<<freq<<std::endl;
    return 0;
} 
