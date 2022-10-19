#include <tuple>
#include <armadillo>
#include <cmath>
#include <iostream>

class moleculeMatrix { 
    public:
        moleculeMatrix(std::string name_, int initial_bond)    {
            this->outputmatrix = arma::mat({{6,0,0,0}});
            this->name = name_;
            arma::vec center=coords(0);
            add_bond(center,0,initial_bond);
        }

        arma::mat outputmatrix;

        void add_bond (arma::vec center, double rads, int bond_type)    {
            double bond_l=bond_lengths_bohr[bond_type];
            double x = center(0) + bond_l * cos(rads);
            double y = center(1) + bond_l * sin(rads);
            double z = 0;
            double a=6;
            if (bond_type==0)   {
                a=1;
            }
            arma::mat row = {{a, x, y, z}};
            outputmatrix=arma::join_cols(outputmatrix, row);
        };

        arma::vec coords(int n) {
            arma::mat row= outputmatrix.row(n);
            return arma::vectorise(row.cols(1,3));
        }

        double sp_hybridize (int center_n, int fixed_n) {
            arma::vec direction= coords(center_n)-coords(fixed_n);
            double theta= atan(direction(1)/direction(0));
            //std::cout<<direction<<std::endl;
            if (direction(0)>0) {
                theta += arma::datum::pi;
            }
            //std::cout<<theta<<std::endl;
            return theta+arma::datum::pi;
        };

        std::tuple<double, double> sp2_hybridize (int center_n, int fixed_n){
            arma::vec direction= coords(center_n)-coords(fixed_n);
            double theta= atan(direction(1)/direction(0));
            //std::cout<<direction<<std::endl;
            if (direction(0)>0) {
                theta += arma::datum::pi;
            }
            return std::make_tuple(theta+arma::datum::pi*2/3, theta+arma::datum::pi*4/3);
        };

        void center_coords (arma::mat atoms)    {
            arma::mat midpoint = mean(atoms);
            midpoint(0) =0;
            arma::mat subtmat = arma::repmat(midpoint, outputmatrix.n_rows, 1);
            this->outputmatrix -= subtmat;
        };

        void round5()    {
            outputmatrix.transform( [](double i) { 
                return (round( i * 100000.0 ) / 100000.0); 
                } );
        }

        void save() {
            this->outputmatrix.save("buffer.txt",arma::arma_ascii);
            std::ifstream infile("buffer.txt");
            std::ofstream outfile(this->name);
            if (infile.good())  {
                std::string header;
                getline(infile, header);
                while(!infile.eof())    {
                    std::string row;
                    getline(infile, row);
                    outfile << row << std::endl;
                }
            }
            outfile.close();
            int status = std::remove("buffer.txt");
        }

        std::string name;
        arma::vec bond_lengths_ang= {1.1, 1.54, 1.34, 1.21};
        arma::vec bond_lengths_bohr= bond_lengths_ang/0.52917706;
};

int main() {
    moleculeMatrix C2H2 ("C2H2-Kevin.txt", 3);
    C2H2.add_bond(C2H2.coords(0),C2H2.sp_hybridize(0,1),0);
    C2H2.add_bond(C2H2.coords(1),C2H2.sp_hybridize(1,0),0);
    C2H2.round5();
    C2H2.center_coords(C2H2.outputmatrix.row(2));
    std::cout<<C2H2.outputmatrix<<std::endl;
    C2H2.save();

    moleculeMatrix C2H4 ("C2H4-Kevin.txt", 2);
    double theta1, theta2;
    std::tie(theta1, theta2) = C2H4.sp2_hybridize(0,1);
    C2H4.add_bond(C2H4.coords(0),theta1,0);
    C2H4.add_bond(C2H4.coords(0),theta2,0);
    std::tie(theta1, theta2) = C2H4.sp2_hybridize(1,0);
    C2H4.add_bond(C2H4.coords(1),theta1,0);
    C2H4.add_bond(C2H4.coords(1),theta2,0);
    C2H4.round5();
    C2H4.center_coords(C2H4.outputmatrix.rows(0,1));
    std::cout<<C2H4.outputmatrix<<std::endl;
    C2H4.save();

    moleculeMatrix benzene ("benzene.txt", 2);      // C0=C1 double bond
    std::tie(theta1, theta2) = benzene.sp2_hybridize(0,1);
    benzene.add_bond(benzene.coords(0),theta2,0);   //add hydrogen to C0
    std::tie(theta1, theta2) = benzene.sp2_hybridize(1,0);
    benzene.add_bond(benzene.coords(1),theta1,0);   //add hydrogen to C1
    benzene.add_bond(benzene.coords(1),theta2,1);   //add -C to C1
    std::tie(theta1, theta2) = benzene.sp2_hybridize(4,1);
    benzene.add_bond(benzene.coords(4),theta1,0);   //add hydrogen to C2
    benzene.add_bond(benzene.coords(4),theta2,2);   //add =C to C2
    std::tie(theta1, theta2) = benzene.sp2_hybridize(6,4);
    benzene.add_bond(benzene.coords(6),theta1,0);   //add hydrogen to C3
    benzene.add_bond(benzene.coords(6),theta2,1);   //add -C to C3
    std::tie(theta1, theta2) = benzene.sp2_hybridize(8,6);
    benzene.add_bond(benzene.coords(8),theta1,0);   //add hydrogen to C4
    benzene.add_bond(benzene.coords(8),theta2,2);   //add =C to C4
    std::tie(theta1, theta2) = benzene.sp2_hybridize(10,8);
    benzene.add_bond(benzene.coords(10),theta1,0);   //add hydrogen to C5
    benzene.round5();
    arma::vec carbons = {0,1,4,6,8,10};
    benzene.center_coords(benzene.outputmatrix.rows(0,1));
    std::cout<<benzene.outputmatrix<<std::endl;
    benzene.save();

    moleculeMatrix C6H8ccc ("C6H8ccc.txt", 2);      // C0=C1 double bond
    std::tie(theta1, theta2) = C6H8ccc.sp2_hybridize(0,1);
    C6H8ccc.add_bond(C6H8ccc.coords(0),theta1,0);   //add hydrogen to C0
    C6H8ccc.add_bond(C6H8ccc.coords(0),theta2,0);   //add hydrogen to C0
    std::tie(theta1, theta2) = C6H8ccc.sp2_hybridize(1,0);
    C6H8ccc.add_bond(C6H8ccc.coords(1),theta1,0);   //add hydrogen to C1
    C6H8ccc.add_bond(C6H8ccc.coords(1),theta2,1);   //add -C to C1
    std::tie(theta1, theta2) = C6H8ccc.sp2_hybridize(5,1);
    C6H8ccc.add_bond(C6H8ccc.coords(5),theta1,0);   //add hydrogen to C2
    C6H8ccc.add_bond(C6H8ccc.coords(5),theta2,2);   //add =C to C2
    std::tie(theta1, theta2) = C6H8ccc.sp2_hybridize(7,5);
    C6H8ccc.add_bond(C6H8ccc.coords(7),theta1,0);   //add hydrogen to C3
    C6H8ccc.add_bond(C6H8ccc.coords(7),theta2,1);   //add -C to C3
    std::tie(theta1, theta2) = C6H8ccc.sp2_hybridize(9,7);
    C6H8ccc.add_bond(C6H8ccc.coords(9),theta1,0);   //add hydrogen to C4
    C6H8ccc.add_bond(C6H8ccc.coords(9),theta2,2);   //add =C to C4
    std::tie(theta1, theta2) = C6H8ccc.sp2_hybridize(11,9);
    C6H8ccc.add_bond(C6H8ccc.coords(11),theta1,0);   //add hydrogen to C5
    C6H8ccc.add_bond(C6H8ccc.coords(11),theta2,0);   //add hydrogen to C5
    C6H8ccc.round5();
    C6H8ccc.center_coords(C6H8ccc.outputmatrix.rows(0,1));
    std::cout<<C6H8ccc.outputmatrix<<std::endl;
    C6H8ccc.save();

    moleculeMatrix C6H8ttt ("C6H8ttt.txt", 2);      // C0=C1 double bond
    std::tie(theta1, theta2) = C6H8ttt.sp2_hybridize(0,1);
    C6H8ttt.add_bond(C6H8ttt.coords(0),theta1,0);   //add hydrogen to C0
    C6H8ttt.add_bond(C6H8ttt.coords(0),theta2,0);   //add hydrogen to C0
    std::tie(theta1, theta2) = C6H8ttt.sp2_hybridize(1,0);
    C6H8ttt.add_bond(C6H8ttt.coords(1),theta1,0);   //add hydrogen to C1
    C6H8ttt.add_bond(C6H8ttt.coords(1),theta2,1);   //add -C to C1
    std::tie(theta1, theta2) = C6H8ttt.sp2_hybridize(5,1);
    C6H8ttt.add_bond(C6H8ttt.coords(5),theta2,0);   //add hydrogen to C2
    C6H8ttt.add_bond(C6H8ttt.coords(5),theta1,2);   //add =C to C2
    std::tie(theta1, theta2) = C6H8ttt.sp2_hybridize(7,5);
    C6H8ttt.add_bond(C6H8ttt.coords(7),theta1,0);   //add hydrogen to C3
    C6H8ttt.add_bond(C6H8ttt.coords(7),theta2,1);   //add -C to C3
    std::tie(theta1, theta2) = C6H8ttt.sp2_hybridize(9,7);
    C6H8ttt.add_bond(C6H8ttt.coords(9),theta2,0);   //add hydrogen to C4
    C6H8ttt.add_bond(C6H8ttt.coords(9),theta1,2);   //add =C to C4
    std::tie(theta1, theta2) = C6H8ttt.sp2_hybridize(11,9);
    C6H8ttt.add_bond(C6H8ttt.coords(11),theta1,0);   //add hydrogen to C5
    C6H8ttt.add_bond(C6H8ttt.coords(11),theta2,0);   //add hydrogen to C5
    C6H8ttt.round5();
    C6H8ttt.center_coords(C6H8ttt.outputmatrix);
    std::cout<<C6H8ttt.outputmatrix<<std::endl;
    C6H8ttt.save();


    return 0;
}