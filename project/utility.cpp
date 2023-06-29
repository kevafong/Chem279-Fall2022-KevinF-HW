#include "utility.h"

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

std::tuple<int, int> divide(int dividend, int divisor) {
    return  std::make_tuple(dividend / divisor, dividend % divisor);
}

int factorial (int n, int i)   {
    int fac=1;
    while (n>0) {
        fac *= n;
        n-= i;
    }
    return fac;
}

int combo (int d, int l) {
    return factorial(d,1) / (factorial(l,1) * factorial(d-l,1));
}

int comboRep (int d, int l) {
    return factorial(d+l-1,1) / (factorial(l,1) * factorial(d-1,1));
}

arma::imat comboRepMatrix (int d, int l) {
    if (d == 1)  {
        return arma::imat(1, 1, arma::fill::value(l));
    }
    else if (l == 0)  {
        return arma::imat(1, d, arma::fill::zeros);
    }
    else    {
        arma::imat base = comboRepMatrix(d-1, 0);
        arma::imat matrix= arma::join_rows(arma::imat(1, 1, arma::fill::value(l)), base);
        int l_counter = 1;
        while(l_counter <= l)   {
            arma::imat submat = comboRepMatrix(d-1, l_counter);
            submat= arma::join_rows(arma::imat(comboRep(d-1, l_counter), 1, arma::fill::value(l-l_counter)), submat);
            matrix= arma::join_cols(matrix, submat);
            l_counter++;
        }
        return matrix;
    }
}

double centerP(double xA, double xB, double alpha, double beta) {
    return (alpha * xA + beta * xB) / (alpha + beta);
}

arma::mat indexMat (int m , int n)  {
    arma::mat display = arma::mat(arma::regspace(0, m*n - 1));
    display.reshape(m,n);
    return display;
}

// double expPrefactor (double xA, double xB, double alpha, double beta) {
//     return exp(-alpha*beta*pow(xA-xB, 2) / (alpha+beta)) * sqrt(M_PI/(alpha+beta));
// }

// double doubleSumm(double xA, double xB, int l_A, int l_B, double alpha, double beta)  {
//     double xP= centerP(xA, xB, alpha, beta);
//     double sum=0;
//     for (int i=0; i<l_A+1; i++) {
//         for (int j=0; j<l_B+1; j++) {
//             if ((i+j)%2==0)   {
//                 sum += combo(l_A, i) * combo(l_B, j) * factorial(i+j-1, 2) * pow(xP-xA, l_A-i) * pow(xP-xB, l_B-j) / (pow(2 * (alpha+beta),(i+j) / 2));
//             }
//         }
//     }
//     return sum;
// }

// double S_AB_1D  (int dim, arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta)  {
//     double xA = centerA(dim);   // coordinate of center in a given dimension
//     double xB = centerB(dim);
//     int l_A = ls_a(dim);        // angular momentum of function in a given dimension
//     int l_B = ls_b(dim);
//     return expPrefactor(xA, xB, alpha, beta) * doubleSumm(xA, xB, l_A, l_B, alpha, beta);
// }

// double S_AB (arma::vec centerA, arma::vec centerB, arma::imat ls_a, arma::imat ls_b, double alpha, double beta)  {
//     double S_ABx=S_AB_1D(0, centerA, centerB, ls_a, ls_b, alpha, beta); 
//     double S_ABy=S_AB_1D(1, centerA, centerB, ls_a, ls_b, alpha, beta);
//     double S_ABz=S_AB_1D(2, centerA, centerB, ls_a, ls_b, alpha, beta);
//     return S_ABx * S_ABy * S_ABz;
// }