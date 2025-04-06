# include "inversion.cuh"

void Inversion::set_parameters()
{
    set_main_parameters();

    set_wavelet();
    set_geometry();
    set_properties();
    set_coordinates();
    set_cerjan_dampers();

    input_folder = catch_parameter("inversion_input_folder", parameters);
    output_folder = catch_parameter("inversion_output_folder", parameters);

    auto frequencies = split(catch_parameter("multi_frequency", parameters),',');

    nFreqs = frequencies.size();

    freqs = new float[nFreqs]();
    for (int fId = 0; fId < nFreqs; fId++) 
        freqs[fId] = std::stof(frequencies[fId]);

    iteration = 0;
}

void Inversion::set_observed_data()
{



}

void Inversion::set_model_dimension()
{



}

void Inversion::forward_modeling()
{
    ABC = true;

    for (srcId = 0; srcId < geometry->nrel; srcId++)
    {
        show_information();
        
        initialization();
        forward_solver();
        set_seismogram();
    }
}

void Inversion::show_information()
{
    auto clear = system("clear");
    
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "                         \033[34mFull Waveform Inversion\033[0;0m\n";
    std::cout << "-------------------------------------------------------------------------------\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << 
                                  ", x = " << (nx - 1)*dx <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n";

    // Iteration number

    // Frequency

    // residuo
}

void Inversion::check_convergence()
{
    // float square_difference = 0.0f;

    // for (int i = 0; i < n_data; i++)
    //     square_difference += powf(dobs[i] - dcal[i], 2.0f);

    // residuo.push_back(sqrtf(square_difference));

    if ((iteration >= max_iteration))
    {
        // std::cout << "Final residuo: "<< residuo.back() <<"\n";
        converged = true;
    }
    else
    {
        iteration += 1;
        converged = false;
    }

}

void Inversion::compute_gradient()
{

 
    
}

void Inversion::optimization()
{



}

void Inversion::update_model()
{


}
