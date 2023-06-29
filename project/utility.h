#pragma once
#include <tuple>
#include <armadillo>
#include <cmath>
#include <iostream>

arma::mat loadfile(std::string filename);

std::tuple<int, int> divide(int dividend, int divisor); 
    // returns quotient and remainder, can convert matrix index to row and column indices

int factorial (int n, int i);
    //Returns the factorial of a number, n, reducing by increment of i
    // if i=1, returns the factorial, if i=2, returns the double factorial, , keeping numbers of same parity

int combo (int d, int l) ;
    // Returns the number of combinations of selecting l objects from a set of d objects.

int comboRep (int d, int l) ;
    // Returns the number of combinations of selecting l objects from a set of d objects, with repetition

arma::imat comboRepMatrix (int d, int l);
    // Returns a matrix of the possible combinations of selecting l objects from a set of d objects, with repetition

arma::mat indexMat (int m , int n) ;
    // Creates an m x n matrix, where the content of a cell is the row major index.


