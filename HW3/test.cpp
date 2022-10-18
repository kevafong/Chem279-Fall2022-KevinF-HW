arma::cube matrixtocube (arma::mat input_matrix)   {
    int slices = input_matrix.n_rows;
    input_matrix=input_matrix.t();
    arma::cube output_cube(input_matrix.memptr(), 3,4, slices, false);
    return output_cube;
}

arma::cube primitiveNorms (arma::cube functions)   {
    std::vector<double> normalsvector;
    int slices=functions.n_slices;
    for (int i=0; i<slices; i++)    {
        arma::mat basisfunc = functions.slice(i);
        arma::vec center = basisfunc.col(0);
        arma::imat quantums = arma::conv_to<arma::imat>::from(basisfunc.col(1));
        for (int j=0 ; j<3; j++)    {
            double exponent = basisfunc(j,2);
            double norm= S_AB (center, center, quantums, quantums, exponent, exponent);
            normalsvector.push_back(1/sqrt(norm));
        }
    }
    arma::mat n_mat (normalsvector);
    arma::cube n_matslice(n_mat.memptr(), 3,1, slices, false);
    std::cout<<"matslice"<<std::endl<<n_matslice;
    return n_matslice;
}

arma::cube process_basisfns (arma::mat input_matrix)    {
    arma::cube output=matrixtocube(input_matrix);
    std::cout<<"basis:"<<std::endl<<output;

    arma::cube norms_mat=primitiveNorms(output);
    std::cout<<"norms:"<<std::endl<<norms_mat;
    std::cout<<"basis2:"<<std::endl<<output;
    output.insert_cols(4,norms_mat);
    
    std::cout<<"final:"<<std::endl<<output;
    return output;
}